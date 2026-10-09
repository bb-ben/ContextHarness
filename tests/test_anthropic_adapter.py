"""Offline checks of the Anthropic adapter against the installed SDK's types."""

import inspect

import anthropic
from anthropic.resources.beta.messages import Messages as BetaMessages
from anthropic.resources.messages import Messages
from anthropic.types.beta import BetaMessage

from context_harness.models.anthropic_adapter import FALLBACK_BETA, AnthropicAdapter
from context_harness.models.base import ModelRequest, ModelSettings, TextBlock, ToolSpec

REQUEST = ModelRequest(
    phase="memory_update",
    system="SYSTEM",
    blocks=(TextBlock("SESSION", cache=True), TextBlock("DYNAMIC")),
    tools=(ToolSpec("read_file", "Read.", {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),),
    metadata={"session": 3, "step": 0},
)


class _StubBeta:
    def __init__(self, message):
        self.message, self.kwargs = message, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.message


class _StubClient:
    def __init__(self, message):
        self.beta = type("B", (), {})()
        self.beta.messages = _StubBeta(message)
        self.messages = _StubBeta(message)


def _message(**overrides):
    data = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [
            {"type": "text", "text": "Noting the move."},
            {"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {"path": "context.md"}},
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 40, "cache_read_input_tokens": 900,
                  "cache_creation_input_tokens": 0},
    }
    data.update(overrides)
    return BetaMessage.model_validate(data)


def test_params_bind_to_sdk_signatures():
    adapter = AnthropicAdapter(ModelSettings(model="claude-opus-5-5"), client=_StubClient(_message()))
    params = adapter.build_params(REQUEST)
    inspect.signature(BetaMessages.create).bind(None, **params, betas=[FALLBACK_BETA], fallbacks="default")
    inspect.signature(Messages.create).bind(None, **params)

    assert params["output_config"] == {"effort": "medium"}
    assert "temperature" not in params and "thinking" not in params
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    content = params["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "SESSION", "cache_control": {"type": "ephemeral"}}
    assert content[1] == {"type": "text", "text": "DYNAMIC"}
    assert "metadata" not in params and "session" not in str(params["messages"])  # bookkeeping stays local


def test_response_parsing_and_fallback_routing():
    client = _StubClient(_message())
    response = AnthropicAdapter(ModelSettings(model="claude-opus-5-5"), client=client).complete(REQUEST)
    assert client.beta.messages.kwargs["fallbacks"] == "default"
    assert client.beta.messages.kwargs["betas"] == [FALLBACK_BETA]
    assert response.text == "Noting the move."
    assert [(c.id, c.name, c.arguments) for c in response.tool_calls] == [("tu_1", "read_file", {"path": "context.md"})]
    assert response.usage.cache_read_input_tokens == 900 and response.usage.total_input_tokens == 1020
    assert response.stop_reason == "tool_use" and response.model == "claude-opus-5-5"


def test_no_fallbacks_uses_standard_endpoint():
    client = _StubClient(_message())
    AnthropicAdapter(ModelSettings(model="claude-opus-5-5", fallbacks=False, effort=None), client=client).complete(REQUEST)
    assert client.messages.kwargs is not None and client.beta.messages.kwargs is None
    assert "output_config" not in client.messages.kwargs


def test_refusal_is_reported_not_raised():
    message = _message(
        content=[], stop_reason="refusal",
        stop_details={"type": "refusal", "category": "cyber", "explanation": "declined"},
    )
    response = AnthropicAdapter(ModelSettings(model="claude-opus-5-5"), client=_StubClient(message)).complete(REQUEST)
    assert response.stop_reason == "refusal" and response.tool_calls == ()
    assert response.extra["stop_details"]["category"] == "cyber"


def test_sdk_version_supports_default_fallbacks():
    assert "fallbacks" in inspect.signature(BetaMessages.create).parameters
    assert anthropic.__version__.split(".")[0] >= "1"


def test_outages_after_sdk_retries_pause_the_run():
    import httpx2
    import pytest

    from context_harness.models.base import ProviderUnavailable

    class Failing:
        def __init__(self, exc):
            self.exc = exc

        def create(self, **kwargs):
            raise self.exc

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    for exc in (anthropic.OverloadedError("overloaded", response=httpx2.Response(529, request=request), body=None),
                anthropic.APIConnectionError(request=request)):
        client = _StubClient(_message())
        client.beta.messages = Failing(exc)
        with pytest.raises(ProviderUnavailable):
            AnthropicAdapter(ModelSettings(model="claude-opus-5-5"), client=client).complete(REQUEST)
