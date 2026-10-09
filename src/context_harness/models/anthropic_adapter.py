"""Anthropic Messages API adapter. All provider-specific code lives here."""

from __future__ import annotations

import time
from typing import Any

import anthropic

from .base import ModelRequest, ModelResponse, ModelSettings, ProviderUnavailable, ToolCall, Usage

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicAdapter:
    def __init__(self, settings: ModelSettings, client: anthropic.Anthropic | None = None):
        self.settings = settings
        self.client = client or anthropic.Anthropic(
            max_retries=settings.max_retries, timeout=settings.timeout_s
        )

    def build_params(self, request: ModelRequest) -> dict[str, Any]:
        s = self.settings
        # Render order is tools -> system -> messages. The tool list and system
        # prompt are fixed per phase, and the first user block (the session
        # transcript) is fixed within a session's step loop, so cache breakpoints
        # sit on the system prompt and on blocks marked cache=True.
        params: dict[str, Any] = {
            "model": s.model,
            "max_tokens": s.max_tokens,
            "system": [
                {"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": b.text}
                        | ({"cache_control": {"type": "ephemeral"}} if b.cache else {})
                        for b in request.blocks
                    ],
                }
            ],
        }
        if request.tools:
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in request.tools
            ]
        if s.effort is not None:
            params["output_config"] = {"effort": s.effort}
        if s.thinking is not None:
            params["thinking"] = s.thinking
        if s.temperature is not None:
            params["temperature"] = s.temperature
        return params

    def complete(self, request: ModelRequest) -> ModelResponse:
        params = self.build_params(request)
        start = time.perf_counter()
        try:  # the SDK has already retried these max_retries times
            if self.settings.fallbacks:
                message = self.client.beta.messages.create(**params, betas=[FALLBACK_BETA], fallbacks="default")
            else:
                message = self.client.messages.create(**params)
        except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError,
                anthropic.OverloadedError, anthropic.ServiceUnavailableError) as e:
            raise ProviderUnavailable(f"Anthropic unavailable after retries: {type(e).__name__}") from e
        latency = time.perf_counter() - start

        texts, calls, fallback_events = [], [], []
        for block in message.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
            elif block.type == "fallback":
                fallback_events.append({"from": block.from_.model, "to": block.to.model})

        u = message.usage
        extra: dict[str, Any] = {}
        if fallback_events:
            extra["fallbacks"] = fallback_events
        if message.stop_reason == "refusal" and getattr(message, "stop_details", None):
            extra["stop_details"] = {
                "category": message.stop_details.category,
                "explanation": message.stop_details.explanation,
            }
        return ModelResponse(
            text="".join(texts),
            tool_calls=tuple(calls),
            stop_reason=message.stop_reason or "",
            usage=Usage(
                input_tokens=u.input_tokens or 0,
                output_tokens=u.output_tokens or 0,
                cache_read_input_tokens=u.cache_read_input_tokens or 0,
                cache_creation_input_tokens=u.cache_creation_input_tokens or 0,
            ),
            model=message.model,
            latency_s=latency,
            request_id=getattr(message, "_request_id", None),
            extra=extra,
        )
