"""Deterministic stand-ins for an LLM, for credential-free runs and tests.

FakeModel reads only the request it is given, exactly as a real model would, so
it exercises replay, workspace semantics, logging, and scoring end to end. Its
answer heuristic (word overlap between options and memory) is label-free.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from .base import ModelRequest, ModelResponse, ModelSettings, ToolCall, Usage

_SESSION = re.compile(r'<session date="([^"]+)">(.*?)</session>', re.S)
_USER_LINE = re.compile(r"^User: (.*)$", re.M)
_WORKING = re.compile(r"<working_context[^>]*>\n(.*?)\n</working_context>", re.S)
_SUMMARY = re.compile(r"<current_summary>\n(.*?)\n</current_summary>", re.S)
_OPTION = re.compile(r"^([A-E]): (.*)$", re.M)
_QUESTION = re.compile(r"<question>.*?</question>", re.S)
_SEEN = re.compile(r"Sessions seen: (\d+)")
_WORD = re.compile(r"[a-z]{4,}")


def _session_facts(prompt: str) -> tuple[str, str]:
    date, body = _SESSION.search(prompt).groups()
    first_user = _USER_LINE.search(body)
    return date, first_user.group(1) if first_user else ""


def _usage(request: ModelRequest, text: str, calls: list[ToolCall]) -> Usage:
    out = len(text) + sum(len(json.dumps(c.arguments)) for c in calls)
    return Usage(input_tokens=len(request.prompt_text()) // 4, output_tokens=max(1, out // 4))


class FakeModel:
    def __init__(self, settings: ModelSettings | None = None):
        self.settings = settings or ModelSettings(model="fake", effort=None, fallbacks=False)
        self.requests: list[ModelRequest] = []
        self._ids = 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        handler = {
            "memory_update": self._memory_update,
            "summary_update": self._summary_update,
        }.get(request.phase, self._answer)
        text, calls = handler(request)
        return ModelResponse(
            text=text,
            tool_calls=tuple(calls),
            stop_reason="tool_use" if calls else "end_turn",
            usage=_usage(request, text, calls),
            model=self.settings.model,
            latency_s=0.0,
        )

    def _call(self, name: str, **arguments) -> ToolCall:
        self._ids += 1
        return ToolCall(id=f"fake_{self._ids}", name=name, arguments=arguments)

    def _memory_update(self, request: ModelRequest) -> tuple[str, list[ToolCall]]:
        if request.metadata.get("step", 0) > 0:
            return "Done with this session.", []
        prompt = request.prompt_text()
        date, first_user = _session_facts(prompt)
        working = _WORKING.search(prompt).group(1)
        # The session counter lives only in context.md, so it advances only if
        # edits to context.md really reach the next model call.
        seen = int(m.group(1)) + 1 if (m := _SEEN.search(working)) else 1
        context = (
            "# Context\n\n"
            f"Sessions seen: {seen}\n"
            f"Last session: {date[:10]}: {first_user[:160]}\n"
            "Details: user_state/observations.md\n"
        )
        note = f"- {date[:10]}: {first_user[:200]}"
        return "Recording this session.", [
            self._call("append_file", path="user_state/observations.md", text=note),
            self._call("write_file", path="context.md", content=context),
        ]

    def _summary_update(self, request: ModelRequest) -> tuple[str, list[ToolCall]]:
        prompt = request.prompt_text()
        date, first_user = _session_facts(prompt)
        current = _SUMMARY.search(prompt).group(1)
        lines = [] if current == "(empty)" else current.splitlines()
        lines = (lines + [f"- {date[:10]}: {first_user[:200]}"])[-30:]
        return "\n".join(lines), []

    def _answer(self, request: ModelRequest) -> tuple[str, list[ToolCall]]:
        tool_names = {t.name for t in request.tools}
        if request.metadata.get("step", 0) == 0 and "read_file" in tool_names:
            return "", [self._call("read_file", path="user_state/observations.md")]
        prompt = request.prompt_text()
        question = _QUESTION.search(prompt).group(0)
        memory_words = set(_WORD.findall(_QUESTION.sub("", prompt).lower()))
        options = _OPTION.findall(question)
        best = max(options, key=lambda o: (len(set(_WORD.findall(o[1].lower())) & memory_words), -ord(o[0])))
        return "", [self._call("submit_answer", letter=best[0])]


class ScriptedModel:
    """Test double: `script(request)` returns (text, [(tool_name, arguments), ...])."""

    def __init__(self, script: Callable[[ModelRequest], tuple[str, list[tuple[str, dict]]]]):
        self.settings = ModelSettings(model="scripted", effort=None, fallbacks=False)
        self.script = script
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        text, raw_calls = self.script(request)
        calls = [ToolCall(id=f"s{len(self.requests)}_{i}", name=n, arguments=a) for i, (n, a) in enumerate(raw_calls)]
        return ModelResponse(
            text=text,
            tool_calls=tuple(calls),
            stop_reason="tool_use" if calls else "end_turn",
            usage=_usage(request, text, calls),
            model="scripted",
            latency_s=0.0,
        )
