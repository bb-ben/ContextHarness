"""Offline checks of the OpenAI adapter and cost cap against the installed SDK's types."""

import httpx
import openai
import pytest
from openai.types.responses import Response

from context_harness.models.base import (
    ModelRequest, ModelSettings, ProviderUnavailable, QuotaExhausted, TextBlock, ToolSpec, Usage,
)
from context_harness.models.openai_adapter import OpenAIAdapter
from context_harness.models.pricing import CostCap, cost_usd
from context_harness.tools import Finish

REQUEST = ModelRequest(
    phase="memory_update",
    system="CONTRACT",
    blocks=(TextBlock("EVENT", cache=True), TextBlock("DYNAMIC")),
    tools=(ToolSpec("read_file", "Read.", {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
           Finish().spec),
    metadata={"event_id": "S-3", "step": 0},
)


def completion(content=None, tool_calls=None, finish=None, refusal=None, usage=(10_000, 6_000, 3_000, 2_000)):
    """A Responses API result. finish: None (completed) or an incomplete reason."""
    prompt, cached, out, reasoning = usage
    output = [{"type": "reasoning", "id": "rs_1", "summary": []}]
    parts = ([{"type": "output_text", "text": content, "annotations": []}] if content else [])
    parts += ([{"type": "refusal", "refusal": refusal}] if refusal else [])
    if parts:
        output.append({"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": parts})
    output += tool_calls or []
    return Response.model_validate({
        "id": "resp_1", "object": "response", "created_at": 0, "model": "gpt-5.4-mini-2026-03-17",
        "status": "incomplete" if finish else "completed",
        "incomplete_details": {"reason": finish} if finish else None,
        "output": output, "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "usage": {"input_tokens": prompt, "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": 0},
                  "output_tokens": out, "output_tokens_details": {"reasoning_tokens": reasoning},
                  "total_tokens": prompt + out},
    })


def call(i, name, args):
    return {"type": "function_call", "id": f"fc_{i}", "call_id": f"call_{i}", "name": name, "arguments": args,
            "status": "completed"}


class Stub:
    def __init__(self, outcome):
        self.outcome, self.kwargs = outcome, None
        self.responses = self

    def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def adapter(outcome, **settings):
    stub = Stub(outcome)
    return OpenAIAdapter(ModelSettings(model="gpt-5.4-mini", **settings), client=stub), stub


def test_request_shape():
    a, stub = adapter(completion(content="ok"))
    a.complete(REQUEST)
    k = stub.kwargs
    assert k["model"] == "gpt-5.4-mini" and k["store"] is False and k["reasoning"] == {"effort": "medium"}
    assert k["instructions"] == "CONTRACT"
    assert [p["text"] for p in k["input"][0]["content"]] == ["EVENT", "DYNAMIC"] and k["input"][0]["role"] == "user"
    assert [t["name"] for t in k["tools"]] == ["read_file", "finish"] and k["tool_choice"] == "auto"
    assert all(t["type"] == "function" and t["strict"] is False for t in k["tools"])
    assert k["max_output_tokens"] == 16000 and "temperature" not in k
    assert "S-3" not in str(k)  # bookkeeping metadata stays local


def test_parsing_tool_calls_text_and_usage():
    r = adapter(completion(content="Recording.", tool_calls=[call(1, "read_file", '{"path": "context.md"}'),
                                                              call(2, "finish", "{}")]))[0].complete(REQUEST)
    assert r.text == "Recording." and r.stop_reason == "tool_use" and r.model == "gpt-5.4-mini-2026-03-17"
    assert [(c.id, c.name, c.arguments) for c in r.tool_calls] == [("call_1", "read_file", {"path": "context.md"}),
                                                                   ("call_2", "finish", {})]
    assert (r.usage.input_tokens, r.usage.cache_read_input_tokens, r.usage.output_tokens) == (4000, 6000, 3000)
    assert r.extra["thought_tokens"] == 2000


@pytest.mark.parametrize("kwargs,stop", [
    ({"tool_calls": [call(1, "read_file", "{not json")]}, "malformed_tool_call"),
    ({"refusal": "I can't help with that."}, "refusal"),
    ({"content": "x", "finish": "content_filter"}, "refusal"),
    ({"content": "x", "finish": "max_output_tokens"}, "max_tokens"),
    ({"content": "done"}, "end_turn"),
])
def test_stop_reasons(kwargs, stop):
    assert adapter(completion(**kwargs))[0].complete(REQUEST).stop_reason == stop


def _status_error(cls, status, code):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return cls("error", response=httpx.Response(status, request=request), body={"code": code, "message": "error"})


def test_errors_pause_instead_of_failing():
    with pytest.raises(QuotaExhausted):
        adapter(_status_error(openai.RateLimitError, 429, "insufficient_quota"))[0].complete(REQUEST)
    with pytest.raises(ProviderUnavailable):
        adapter(_status_error(openai.RateLimitError, 429, "rate_limit_exceeded"))[0].complete(REQUEST)
    with pytest.raises(ProviderUnavailable):
        adapter(openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com")))[0].complete(REQUEST)
    with pytest.raises(ValueError, match="reasoning efforts"):
        adapter(completion(), effort="max")


def test_cost_and_cost_cap():
    usage = Usage(input_tokens=4000, output_tokens=3000, cache_read_input_tokens=6000)
    assert cost_usd(usage, "gpt-5.4-mini") == pytest.approx((4000 * 0.75 + 6000 * 0.075 + 3000 * 4.50) / 1e6)
    assert cost_usd(usage, "unknown-model") is None
    a, _ = adapter(completion(content="ok"))
    capped = CostCap(a, max_usd=0.03)  # each call costs ~$0.017
    capped.complete(REQUEST)
    capped.complete(REQUEST)
    with pytest.raises(QuotaExhausted, match="cost cap"):
        capped.complete(REQUEST)
    with pytest.raises(ValueError, match="No published price"):
        CostCap(OpenAIAdapter(ModelSettings(model="gpt-unknown"), client=Stub(None)), 1.0)


def test_gpt_models_route_to_openai(monkeypatch):
    from context_harness.models import make_model

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert isinstance(make_model(ModelSettings(model="gpt-5.4-mini")), OpenAIAdapter)
