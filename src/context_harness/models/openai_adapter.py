"""OpenAI adapter (openai SDK, Chat Completions with function tools).

All OpenAI-specific code lives here. Each harness request is one stateless call:
a developer message (the contract) plus one user message. `store=False` keeps
completions out of OpenAI's stored-completions log.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .base import ModelRequest, ModelResponse, ModelSettings, ProviderUnavailable, QuotaExhausted, ToolCall, Usage

REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


class OpenAIAdapter:
    def __init__(self, settings: ModelSettings, client: Any = None, sleep=time.sleep):
        if settings.effort is not None and settings.effort not in REASONING_EFFORTS:
            raise ValueError(f"OpenAI reasoning efforts are {sorted(REASONING_EFFORTS)}; got {settings.effort!r}")
        self.settings = settings
        if client is None:
            import openai

            # Reads OPENAI_API_KEY. The SDK retries 408/409/429/5xx and connection errors itself.
            client = openai.OpenAI(max_retries=settings.max_retries, timeout=settings.timeout_s)
        self.client = client
        self._sleep = sleep
        self._last_start = 0.0

    def build_params(self, request: ModelRequest) -> dict[str, Any]:
        s = self.settings
        params: dict[str, Any] = {
            "model": s.model,
            "messages": [
                {"role": "developer", "content": request.system},
                {"role": "user", "content": [{"type": "text", "text": b.text} for b in request.blocks]},
            ],
            "max_completion_tokens": s.max_tokens,
            "store": False,
        }
        if request.tools:
            params["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
                for t in request.tools
            ]
            params["tool_choice"] = "auto"
        if s.effort is not None:
            params["reasoning_effort"] = s.effort
        if s.temperature is not None:
            params["temperature"] = s.temperature
        return params

    def _pace(self) -> None:
        if self.settings.requests_per_minute:
            wait = 60.0 / self.settings.requests_per_minute - (time.monotonic() - self._last_start)
            if wait > 0:
                self._sleep(wait)
        self._last_start = time.monotonic()

    def complete(self, request: ModelRequest) -> ModelResponse:
        import openai

        params = self.build_params(request)
        self._pace()
        start = time.perf_counter()
        try:  # the SDK has already retried these max_retries times
            completion = self.client.chat.completions.create(**params)
        except openai.RateLimitError as e:
            if getattr(e, "code", None) == "insufficient_quota":
                raise QuotaExhausted(f"OpenAI quota or billing limit reached: {e}") from e
            raise ProviderUnavailable(f"OpenAI rate limit persisted after retries: {e}") from e
        except (openai.APIConnectionError, openai.InternalServerError) as e:
            raise ProviderUnavailable(f"OpenAI unavailable after retries: {type(e).__name__}") from e
        return self._parse(completion, time.perf_counter() - start)

    def _parse(self, completion, latency: float) -> ModelResponse:
        choice = completion.choices[0]
        message = choice.message
        calls, malformed = [], False
        for tc in message.tool_calls or []:
            if getattr(tc, "type", "function") != "function":
                continue
            try:
                arguments = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                malformed = True
                continue
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments if isinstance(arguments, dict) else {}))

        finish = choice.finish_reason or ""
        if getattr(message, "refusal", None) or finish == "content_filter":
            stop = "refusal"
        elif malformed and not calls:
            stop = "malformed_tool_call"
        elif finish == "length":
            stop = "max_tokens"
        else:
            stop = "tool_use" if calls else "end_turn"

        u = completion.usage
        prompt = (u.prompt_tokens or 0) if u else 0
        cached = (getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0) if u and u.prompt_tokens_details else 0
        reasoning = (getattr(u.completion_tokens_details, "reasoning_tokens", 0) or 0) if u and u.completion_tokens_details else 0
        extra: dict[str, Any] = {"finish_reason": finish, "thought_tokens": reasoning}
        if getattr(message, "refusal", None):
            extra["refusal"] = message.refusal
        return ModelResponse(
            text=message.content or "",
            tool_calls=tuple(calls),
            stop_reason=stop,
            usage=Usage(
                input_tokens=prompt - cached,
                output_tokens=(u.completion_tokens or 0) if u else 0,
                cache_read_input_tokens=cached,
            ),
            model=completion.model,
            latency_s=latency,
            request_id=getattr(completion, "_request_id", None),
            extra=extra,
        )
