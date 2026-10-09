"""Offline checks of the OpenAI adapter and cost cap against the installed SDK's types."""

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletion

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


def completion(content=None, tool_calls=None, finish="tool_calls", refusal=None,
               usage=(10_000, 6_000, 3_000, 2_000)):
    prompt, cached, out, reasoning = usage
    return ChatCompletion.model_validate({
        "id": "cc_1", "object": "chat.completion", "created": 0, "model": "gpt-5.4-mini-2026-03-17",
        "choices": [{"index": 0, "finish_reason": finish, "message": {
            "role": "assistant", "content": content, "refusal": refusal, "tool_calls": tool_calls}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": out, "total_tokens": prompt + out,
                  "prompt_tokens_details": {"cached_tokens": cached},
                  "completion_tokens_details": {"reasoning_tokens": reasoning}},
    })


def call(i, name, args):
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": args}}


class Stub:
    def __init__(self, outcome):
        self.outcome, self.kwargs = outcome, None
        self.chat = type("Chat", (), {})()
        self.chat.completions = self

    def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def adapter(outcome, **settings):
    stub = Stub(outcome)
    return OpenAIAdapter(ModelSettings(model="gpt-5.4-mini", **settings), client=stub), stub


def test_request_shape():
    a, stub = adapter(completion(content="ok", finish="stop"))
    a.complete(REQUEST)
    k = stub.kwargs
    assert k["model"] == "gpt-5.4-mini" and k["store"] is False and k["reasoning_effort"] == "medium"
    assert k["messages"][0] == {"role": "developer", "content": "CONTRACT"}
    assert [p["text"] for p in k["messages"][1]["content"]] == ["EVENT", "DYNAMIC"]
    assert [t["function"]["name"] for t in k["tools"]] == ["read_file", "finish"] and k["tool_choice"] == "auto"
    assert k["max_completion_tokens"] == 16000 and "temperature" not in k
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
    ({"refusal": "I can't help with that.", "finish": "stop"}, "refusal"),
    ({"content": "x", "finish": "length"}, "max_tokens"),
    ({"content": "done", "finish": "stop"}, "end_turn"),
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
    a, _ = adapter(completion(content="ok", finish="stop"))
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
