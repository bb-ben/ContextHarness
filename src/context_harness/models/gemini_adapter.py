"""Google Gemini adapter (google-genai SDK, stateless generate_content).

All Gemini-specific code lives here. Each harness request is one user turn, so
no conversation state is kept on Google's side.
"""

from __future__ import annotations

import re
import time
from typing import Any

from .base import ModelRequest, ModelResponse, ModelSettings, QuotaExhausted, ToolCall, Usage

THINKING_LEVELS = {"minimal", "low", "medium", "high"}
REFUSAL_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY"}


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop JSON Schema keywords Gemini function declarations may not accept."""
    if not isinstance(schema, dict):
        return schema
    out = {k: _clean_schema(v) if isinstance(v, dict) else v for k, v in schema.items() if k != "additionalProperties"}
    if "properties" in out:
        out["properties"] = {k: _clean_schema(v) for k, v in out["properties"].items()}
    return out


def classify_rate_limit(details: Any) -> tuple[str, float | None]:
    """Return ('daily' | 'transient', retry_delay_seconds) for a 429 response body."""
    error = details.get("error", details) if isinstance(details, dict) else {}
    kind, delay = "transient", None
    for item in error.get("details", []) if isinstance(error, dict) else []:
        for violation in item.get("violations", []) or []:
            if "PerDay" in str(violation.get("quotaId", "")):
                kind = "daily"
        if m := re.fullmatch(r"(\d+(?:\.\d+)?)s", str(item.get("retryDelay", ""))):
            delay = float(m.group(1))
    if kind == "transient" and "per day" in str(error.get("message", "")).lower():
        kind = "daily"
    return kind, delay


class GeminiAdapter:
    def __init__(self, settings: ModelSettings, client: Any = None, sleep=time.sleep):
        if settings.effort is not None and settings.effort not in THINKING_LEVELS:
            raise ValueError(f"Gemini thinking levels are {sorted(THINKING_LEVELS)}; got effort={settings.effort!r}")
        self.settings = settings
        if client is None:
            from google import genai

            client = genai.Client()  # reads GEMINI_API_KEY (or GOOGLE_API_KEY)
        self.client = client
        self._sleep = sleep
        self._last_start = 0.0

    def build_config(self, request: ModelRequest):
        from google.genai import types

        s = self.settings
        config: dict[str, Any] = {
            "system_instruction": request.system,
            "max_output_tokens": s.max_tokens,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if request.tools:
            config["tools"] = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=t.name, description=t.description, parameters_json_schema=_clean_schema(t.input_schema)
                        )
                        for t in request.tools
                    ]
                )
            ]
            config["tool_config"] = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="AUTO"))
        if s.effort is not None:
            config["thinking_config"] = types.ThinkingConfig(thinking_level=s.effort.upper())
        if s.temperature is not None:
            config["temperature"] = s.temperature
        return types.GenerateContentConfig(**config)

    def build_contents(self, request: ModelRequest):
        from google.genai import types

        return [types.Content(role="user", parts=[types.Part(text=b.text) for b in request.blocks])]

    def _pace(self) -> None:
        if self.settings.requests_per_minute:
            wait = 60.0 / self.settings.requests_per_minute - (time.monotonic() - self._last_start)
            if wait > 0:
                self._sleep(wait)
        self._last_start = time.monotonic()

    def complete(self, request: ModelRequest) -> ModelResponse:
        from google.genai import errors

        config, contents = self.build_config(request), self.build_contents(request)
        for attempt in range(self.settings.max_retries + 1):
            self._pace()
            start = time.perf_counter()
            try:
                response = self.client.models.generate_content(model=self.settings.model, contents=contents, config=config)
                return self._parse(response, time.perf_counter() - start)
            except errors.ClientError as e:
                if e.code != 429:
                    raise
                kind, delay = classify_rate_limit(e.details)
                if kind == "daily":
                    raise QuotaExhausted(f"Gemini daily quota exhausted for {self.settings.model}: {e.message}") from e
                if attempt == self.settings.max_retries:
                    raise
                self._sleep(delay if delay is not None else min(60.0, 2.0 ** attempt * 5))
            except errors.ServerError:
                if attempt == self.settings.max_retries:
                    raise
                self._sleep(min(60.0, 2.0 ** attempt * 5))
        raise RuntimeError("unreachable")

    def _parse(self, response, latency: float) -> ModelResponse:
        texts, calls = [], []
        candidate = response.candidates[0] if response.candidates else None
        for i, part in enumerate((candidate.content.parts or []) if candidate and candidate.content else []):
            if part.function_call is not None:
                fc = part.function_call
                calls.append(ToolCall(id=fc.id or f"gemini_{i}", name=fc.name, arguments=dict(fc.args or {})))
            elif part.text and not part.thought:
                texts.append(part.text)

        finish = str(getattr(candidate.finish_reason, "value", candidate.finish_reason)) if candidate else ""
        blocked = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
        if blocked or finish in REFUSAL_REASONS:
            stop = "refusal"
        elif finish == "MAX_TOKENS":
            stop = "max_tokens"
        elif finish == "MALFORMED_FUNCTION_CALL":
            stop = "malformed_tool_call"
        else:
            stop = "tool_use" if calls else "end_turn"

        u = response.usage_metadata
        prompt = (u.prompt_token_count or 0) if u else 0
        cached = (u.cached_content_token_count or 0) if u else 0
        thoughts = (u.thoughts_token_count or 0) if u else 0
        extra: dict[str, Any] = {"finish_reason": finish, "thought_tokens": thoughts}
        if blocked:
            extra["block_reason"] = str(blocked)
        return ModelResponse(
            text="".join(texts),
            tool_calls=tuple(calls),
            stop_reason=stop,
            usage=Usage(
                input_tokens=prompt - cached,
                output_tokens=((u.candidates_token_count or 0) if u else 0) + thoughts,
                cache_read_input_tokens=cached,
            ),
            model=getattr(response, "model_version", None) or self.settings.model,
            latency_s=latency,
            request_id=getattr(response, "response_id", None),
            extra=extra,
        )
