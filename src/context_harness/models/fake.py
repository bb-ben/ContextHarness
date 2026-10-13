"""Deterministic stand-ins for an LLM, for credential-free runs and tests.

FakeModel reads only the request it is given, as a real model would, and follows
the memory contract (agent.md) exactly, so a fake run should produce zero memory
violations: any violation it does produce is a bug in the checks or the layer.
Its answer heuristic (word overlap between options and memory) is label-free.
"""

from __future__ import annotations

import json
import re
from datetime import date as Date, timedelta
from typing import Callable

from .base import ModelRequest, ModelResponse, ModelSettings, ToolCall, Usage

_EVENT = re.compile(r'<event id="S-(\d+)" type="[^"]*" date="([^"]+)">(.*?)</event>', re.S)
_USER_LINE = re.compile(r"^User: (.*)$", re.M)
_WORKING = re.compile(r"<working_context[^>]*>\n(.*?)\n</working_context>", re.S)
_DUE = re.compile(r"<due>\n(.*?)\n</due>", re.S)
_SUMMARY = re.compile(r"<current_summary[^>]*>\n(.*?)\n</current_summary>", re.S)
_OPTION = re.compile(r"^([A-E]): (.*)$", re.M)
_QUESTION = re.compile(r"<question>.*?</question>", re.S)
_SEEN = re.compile(r"events seen: (\d+)")
_WORD = re.compile(r"[a-z]{4,}")


def _clean(text: str, n: int) -> str:
    return " ".join(text.replace("|", "/").split())[:n]


def _event_facts(prompt: str) -> tuple[int, str, str]:
    index, date, body = _EVENT.search(prompt).groups()
    first_user = _USER_LINE.search(body)
    return int(index), date, first_user.group(1) if first_user else ""


def _body(request: ModelRequest) -> str:
    """The user message only. The system prompt (agent.md) contains example tags."""
    return "\n".join(b.text for b in request.blocks)


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
        handler = {"memory_update": self._memory_update, "summary_update": self._summary_update,
                   "memory_review": lambda r: ("Nothing to fix.", [self._call("finish")]),
                   "profile_update": self._profile_update}.get(request.phase, self._answer)
        if request.phase.endswith("_shorten"):  # keep the first half of the over-long text
            handler = lambda r: (_body(r).split("\n", 1)[1].rsplit("\n</", 1)[0][: len(_body(r)) // 2], [])  # noqa: E731
        elif handler == self._answer and request.tools and not {"submit_answer", "read_file"} & {t.name for t in request.tools}:
            handler = self._fill_tool  # any other single-tool request (benchmark adapters)
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
            return "Done with this event.", []
        prompt = _body(request)
        k, date, first_user = _event_facts(prompt)
        working = _WORKING.search(prompt).group(1)
        # The event counter lives only in context.md, so it advances only if
        # edits to context.md really reach the next model call.
        seen = int(m.group(1)) + 1 if (m := _SEEN.search(working)) else 1
        review = (Date.fromisoformat(date) + timedelta(days=7)).isoformat()
        calls = [
            self._call(
                "append_file",
                path="memory/episodes.md",
                text=f"- E-{k} | {date} | source: S-{k} | Conversation: {_clean(first_user, 100)}\n"
                f"  - O-{k}.1 | behavior | Asked: {_clean(first_user, 140)}",
            ),
            self._call(
                "append_file",
                path="state/transient.md",
                text=f"- T-{k}.1 | active | {date} | review: {review} | basis: O-{k}.1 | "
                f"Currently working on: {_clean(first_user, 80)}",
            ),
        ]
        due = _DUE.search(prompt).group(1)
        due_lines = [] if due == "(none)" else due.splitlines()
        for line in due_lines:
            *head, statement = [s.strip() for s in line.split("|")]
            head[1] = "expired"
            calls.append(self._call("edit_file", path="state/transient.md", old_text=line,
                                    new_text=" | ".join(head + [f"ended: {date}", statement])))
        if due_lines:  # closed entries move to the archive so the state file stays bounded
            calls.append(self._call("archive", ids=[line.split("|")[0].strip(" -") for line in due_lines]))
        calls.append(
            self._call(
                "write_file",
                path="context.md",
                content=f"# Working context · as of {date} (S-{k}) · events seen: {seen}\n\n"
                f"## Now\n- Latest request: {_clean(first_user, 80)} (T-{k}.1)\n\n## Watch\n\n## Active\n",
            )
        )
        calls.append(self._call("finish"))
        return "Recording this event.", calls

    def _profile_update(self, request: ModelRequest) -> tuple[str, list[ToolCall]]:
        """Contract-following stand-in: one observation, and a profile that cites it."""
        prompt = _body(request)
        k, date, first_user = _event_facts(prompt)
        old = re.search(r"<profile[^>]*>\n(.*?)\n</profile>", prompt, re.S).group(1)
        line = f"- {date}: asked about {_clean(first_user, 60)} [O-{k}.1]"
        lines = [x for x in old.splitlines() if x.startswith("- ")][-20:] + [line]
        profile = "## How they want help\n" + "\n".join(lines)
        return "", [self._call("record", episode=f"Conversation: {_clean(first_user, 80)}",
                               observations=[{"kind": "behavior", "text": f"Asked: {_clean(first_user, 100)}"}],
                               profile=profile)]

    def _summary_update(self, request: ModelRequest) -> tuple[str, list[ToolCall]]:
        prompt = _body(request)
        _, date, first_user = _event_facts(prompt)
        current = _SUMMARY.search(prompt).group(1)
        lines = [] if current == "(empty)" else current.splitlines()
        lines = (lines + [f"- {date}: {first_user[:200]}"])[-30:]
        return "\n".join(lines), []

    def _fill_tool(self, request: ModelRequest) -> tuple[str, list[ToolCall]]:
        """Call the first tool with its required fields filled by type, from the prompt's opening words."""
        spec = request.tools[0]
        words = _clean(_body(request), 80)

        def value(schema: dict):
            if "enum" in schema:
                return schema["enum"][-1]
            return {"string": words, "number": 1, "integer": 1, "boolean": True, "array": [], "object": {}}.get(
                schema.get("type"), words)

        props = spec.input_schema.get("properties", {})
        return "", [self._call(spec.name, **{k: value(props.get(k, {})) for k in spec.input_schema.get("required", [])})]

    def _answer(self, request: ModelRequest) -> tuple[str, list[ToolCall]]:
        tool_names = {t.name for t in request.tools}
        if request.metadata.get("step", 0) == 0 and "read_file" in tool_names:
            return "", [self._call("read_file", path="memory/episodes.md")]
        prompt = _body(request)
        question = _QUESTION.search(prompt).group(0)
        # Overlap with memory only: the answer instruction (and any strategy) is harness text.
        memory = _QUESTION.sub("", prompt).split("You are now responding to the person")[0]
        memory_words = set(_WORD.findall(memory.lower()))
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
