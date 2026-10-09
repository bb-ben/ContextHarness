"""Offline checks of the Gemini adapter against the installed google-genai types."""

import pytest
from google.genai import errors, types

from context_harness.models.base import ModelRequest, ModelSettings, QuotaExhausted, TextBlock, ToolSpec
from context_harness.models.gemini_adapter import GeminiAdapter, classify_rate_limit
from context_harness.tools import Finish

REQUEST = ModelRequest(
    phase="memory_update",
    system="CONTRACT",
    blocks=(TextBlock("EVENT", cache=True), TextBlock("DYNAMIC")),
    tools=(
        ToolSpec("read_file", "Read.", {"type": "object", "properties": {"path": {"type": "string", "enum": ["context.md"]}},
                                        "required": ["path"], "additionalProperties": False}),
        Finish().spec,
    ),
    metadata={"event_id": "S-3", "step": 0},
)


def response(parts, finish="STOP", usage=None, block=None):
    data = {
        "candidates": [{"content": {"role": "model", "parts": parts}, "finish_reason": finish}],
        "usage_metadata": usage or {"prompt_token_count": 1000, "cached_content_token_count": 600,
                                    "candidates_token_count": 50, "thoughts_token_count": 200},
        "model_version": "gemini-3.7-flash",
        "response_id": "resp_1",
    }
    if block:
        data["prompt_feedback"] = {"block_reason": block}
    return types.GenerateContentResponse.model_validate(data)


def rate_limit(details):
    return errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded",
                                              "details": details}})


class StubModels:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def adapter(outcomes, sleeps=None, **settings):
    client = type("Client", (), {})()
    client.models = StubModels(outcomes)
    return GeminiAdapter(ModelSettings(model="gemini-3.7-flash", **settings), client=client,
                         sleep=(sleeps.append if sleeps is not None else lambda s: None))


def test_request_is_one_stateless_user_turn_with_declared_tools():
    a = adapter([response([{"text": "ok"}])])
    a.complete(REQUEST)
    call = a.client.models.calls[0]
    assert call["model"] == "gemini-3.7-flash"
    contents = call["contents"]
    assert len(contents) == 1 and contents[0].role == "user" and [p.text for p in contents[0].parts] == ["EVENT", "DYNAMIC"]
    config = call["config"]
    assert config.system_instruction == "CONTRACT" and config.automatic_function_calling.disable
    decls = config.tools[0].function_declarations
    assert [d.name for d in decls] == ["read_file", "finish"]
    assert "additionalProperties" not in decls[0].parameters_json_schema
    assert decls[0].parameters_json_schema["properties"]["path"]["enum"] == ["context.md"]
    assert config.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM
    assert "metadata" not in str(call) and "S-3" not in str(call)  # bookkeeping stays local


def test_response_parsing_excludes_thoughts_and_maps_usage():
    r = adapter([response([
        {"thought": True, "text": "private reasoning"},
        {"text": "Recording the event."},
        {"function_call": {"id": "c1", "name": "read_file", "args": {"path": "context.md"}}},
        {"function_call": {"name": "finish", "args": {}}},
    ])]).complete(REQUEST)
    assert r.text == "Recording the event." and "private" not in r.text
    assert [(c.id, c.name, c.arguments) for c in r.tool_calls] == [
        ("c1", "read_file", {"path": "context.md"}), ("gemini_3", "finish", {})]
    assert r.stop_reason == "tool_use" and r.model == "gemini-3.7-flash"
    assert (r.usage.input_tokens, r.usage.cache_read_input_tokens, r.usage.output_tokens) == (400, 600, 250)
    assert r.extra["thought_tokens"] == 200


@pytest.mark.parametrize("kwargs,stop", [
    ({"finish": "SAFETY"}, "refusal"),
    ({"block": "PROHIBITED_CONTENT"}, "refusal"),
    ({"finish": "MAX_TOKENS"}, "max_tokens"),
    ({"finish": "MALFORMED_FUNCTION_CALL"}, "malformed_tool_call"),
])
def test_stop_reasons(kwargs, stop):
    assert adapter([response([{"text": "x"}], **kwargs)]).complete(REQUEST).stop_reason == stop


def test_per_minute_429_waits_for_the_suggested_delay_then_retries():
    sleeps = []
    retry = rate_limit([{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"}])
    r = adapter([retry, response([{"text": "ok"}])], sleeps=sleeps).complete(REQUEST)
    assert r.text == "ok" and sleeps == [7.0]


def test_daily_429_raises_quota_exhausted_without_retrying():
    daily = rate_limit([{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
        {"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
         "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}])
    a = adapter([daily, response([{"text": "never"}])])
    with pytest.raises(QuotaExhausted):
        a.complete(REQUEST)
    assert len(a.client.models.calls) == 1


def test_classify_rate_limit_handles_missing_details():
    assert classify_rate_limit({}) == ("transient", None)
    assert classify_rate_limit({"error": {"message": "Requests per day exceeded"}})[0] == "daily"


def test_pacing_spaces_requests_by_requests_per_minute():
    sleeps = []
    a = adapter([response([{"text": "1"}]), response([{"text": "2"}])], sleeps=sleeps, requests_per_minute=30)
    a.complete(REQUEST)
    a.complete(REQUEST)
    assert len(sleeps) == 1 and 1.9 < sleeps[0] <= 2.0


def test_rejects_effort_levels_gemini_does_not_have():
    with pytest.raises(ValueError, match="thinking levels"):
        adapter([], effort="xhigh")


def test_timeouts_and_dropped_connections_are_retried():
    import httpx

    sleeps = []
    r = adapter([httpx.ReadTimeout("no reply"), response([{"text": "ok"}])], sleeps=sleeps).complete(REQUEST)
    assert r.text == "ok" and sleeps == [5.0]
    assert r.extra["attempts"] == 2 and r.extra["retry_wait_s"] == 5.0


def test_client_is_created_with_the_request_timeout(monkeypatch):
    import google.genai

    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(google.genai, "Client", FakeClient)
    GeminiAdapter(ModelSettings(model="gemini-3.7-flash", timeout_s=240))
    assert captured["http_options"].timeout == 240_000  # milliseconds
