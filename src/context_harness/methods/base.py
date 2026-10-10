"""Method interface and the shared stateless step loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from ..hb_compat import extract_letter
from ..models.base import ModelRequest, ModelResponse, TextBlock, ToolCall
from ..tools import ToolExecutor, ToolResult
from ..tracing import TracedModel, TraceWriter
from ..types import AnswerResult, Event, Probe, Question, Session
from ..workspace import GENERAL, Layout


@dataclass(frozen=True)
class MethodConfig:
    """Budgets shared across methods so that comparisons stay explicit.

    answer_budget_chars caps how much memory text any method can put in front of
    the model when answering: the agent's context.md plus everything it reads,
    the recent-history window, or the rolling summary.
    """

    layout: Layout = GENERAL
    contract_path: str | None = None  # agent.md by default
    preload_state: bool = True  # contract v2: show the layout's state files in full in every call
    answer_strategy: str = "default"  # a key of prompts.ANSWER_STRATEGIES
    episode_rollover_at: float = 0.8  # roll old episodes into the archive above this share of the limit...
    episode_rollover_to: float = 0.5  # ...until the episode log is back under this share
    max_update_steps: int = 8
    max_answer_steps: int = 6
    answer_budget_chars: int = 48000
    summary_max_chars: int = 8000
    action_log_max_chars: int = 60000


@dataclass
class MethodContext:
    model: TracedModel
    trace: TraceWriter
    item_dir: Path
    config: MethodConfig
    resume: bool = False  # reopen an existing workspace instead of creating one


class MemoryMethod(Protocol):
    name: str
    has_workspace: bool  # True if it keeps a state workspace (enables memory checks)

    def observe(self, event: Event) -> None: ...

    def answer(self, probe: Probe, question: Question, probe_event_id: str) -> AnswerResult: ...

    def memory_snapshot(self) -> dict[str, str]: ...

    def export_state(self) -> dict[str, Any]:
        """JSON-serializable state not already on disk, for pausing mid-item."""
        ...

    def restore_state(self, state: dict[str, Any], history: tuple[Session, ...]) -> None:
        """Restore after a pause; `history` holds the events already observed."""
        ...


class ActionLog:
    """What the model has done so far within one step loop, rendered as text.

    Read results are kept verbatim until the log exceeds its budget, at which
    point the oldest read results are elided (the model can read again).
    """

    ARG_PREVIEW = 300

    def __init__(self, max_chars: int):
        self.max_chars = max_chars
        self.entries: list[dict[str, Any]] = []

    def note(self, source: str, text: str) -> None:
        self.entries.append({"kind": "note", "source": source, "text": text.strip()})

    def call(self, call: ToolCall, result: ToolResult, show_result: bool) -> None:
        args = {
            k: (v if not isinstance(v, str) or len(v) <= self.ARG_PREVIEW else f"<{len(v)} chars>")
            for k, v in call.arguments.items()
        }
        self.entries.append(
            {
                "kind": "call",
                "name": call.name,
                "args": json.dumps(args, ensure_ascii=False),
                "ok": result.ok,
                "result": result.content,
                "show_result": show_result or not result.ok,
                "elided": False,
            }
        )

    def render(self) -> str:
        text = self._render()
        for entry in self.entries:
            if len(text) <= self.max_chars:
                break
            if entry["kind"] == "call" and entry["show_result"] and not entry["elided"]:
                entry["elided"] = True
                text = self._render()
        return text

    def _render(self) -> str:
        out = []
        for i, e in enumerate(self.entries, 1):
            if e["kind"] == "note":
                out.append(f"[{i}] ({e['source']} note) {e['text']}")
                continue
            status = "ok" if e["ok"] else "error"
            out.append(f"[{i}] {e['name']} {e['args']} -> {status}")
            if e["elided"]:
                out.append("<result elided to save space; call the tool again if needed>")
            elif e["show_result"]:
                out.append(f"<result>\n{e['result']}\n</result>")
            else:
                out.append(e["result"])
        return "\n".join(out)


@dataclass
class LoopOutcome:
    stop: str  # "done" | "stop_condition" | "max_steps" | "refusal"
    steps: int
    last: ModelResponse | None


def run_step_loop(
    *,
    model: TracedModel,
    executor: ToolExecutor,
    phase: str,
    system: str,
    fixed_text: str,
    render_dynamic: Callable[[str, int], str],
    max_steps: int,
    log: ActionLog,
    metadata: dict[str, Any],
    show_result: Callable[[ToolCall], bool] = lambda c: True,
    stop_when: Callable[[], bool] = lambda: False,
    on_no_tool_calls: Callable[[ModelResponse, ActionLog], bool] = lambda r, log: True,
) -> LoopOutcome:
    """Drive a multi-step tool loop as a series of fresh single-turn requests.

    Each request is [fixed_text (cached), render_dynamic(actions, calls_left)].
    The dynamic block is re-rendered from current state on every step.
    """
    last = None
    for step in range(max_steps):
        request = ModelRequest(
            phase=phase,
            system=system,
            blocks=(
                TextBlock(fixed_text, cache=True),
                TextBlock(render_dynamic(log.render(), max_steps - step - 1)),
            ),
            tools=executor.specs,
            metadata={**metadata, "step": step},
        )
        last = model.complete(request)
        if last.stop_reason == "refusal":
            return LoopOutcome("refusal", step + 1, last)
        if last.stop_reason == "malformed_tool_call":
            log.note("harness", "Your last tool call was malformed and did not run. Try again.")
            continue
        if last.text.strip():
            log.note("your", last.text)
        if not last.tool_calls:
            if on_no_tool_calls(last, log):
                return LoopOutcome("done", step + 1, last)
            continue
        for call in last.tool_calls:
            result = executor.execute(call, phase=phase, step=step, **metadata)
            log.call(call, result, show_result(call))
        if stop_when():
            return LoopOutcome("stop_condition", step + 1, last)
    return LoopOutcome("max_steps", max_steps, last)


def answer_with_tools(
    *,
    ctx: MethodContext,
    executor: ToolExecutor,
    submit,  # SubmitAnswer
    system: str,
    fixed_text: str,
    render_dynamic: Callable[[str, int], str],
) -> AnswerResult:
    """Shared answer phase: loop until submit_answer succeeds.

    If the model replies in plain text, HB's extract_letter is the fallback.
    """
    fallback: dict[str, str] = {}

    def on_text_only(response: ModelResponse, log: ActionLog) -> bool:
        letter = extract_letter(response.text)
        if letter:
            fallback["letter"] = letter
            return True
        log.note("harness", "No answer was submitted. Call submit_answer with one letter.")
        return False

    outcome = run_step_loop(
        model=ctx.model,
        executor=executor,
        phase="answer",
        system=system,
        fixed_text=fixed_text,
        render_dynamic=render_dynamic,
        max_steps=ctx.config.max_answer_steps,
        log=ActionLog(ctx.config.action_log_max_chars),
        metadata={},
        stop_when=lambda: submit.letter is not None,
        on_no_tool_calls=on_text_only,
    )
    raw = outcome.last.text if outcome.last else ""
    details = {"stop": outcome.stop, "steps": outcome.steps}
    if submit.letter is not None:
        return AnswerResult(submit.letter, "tool", raw, details)
    if fallback.get("letter"):
        return AnswerResult(fallback["letter"], "text_fallback", raw, details)
    return AnswerResult(None, "none", raw, details)
