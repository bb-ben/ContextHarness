"""Baselines behind the same model adapter and answer phase as the agent."""

from __future__ import annotations

import difflib
from typing import Any

from .. import prompts
from ..models.base import ModelRequest, TextBlock
from ..tools import SubmitAnswer, ToolExecutor
from ..types import AnswerResult, Event, Probe, Question, Session
from .base import MethodContext, answer_with_tools


def _no_dynamic(actions: str, calls_left: int) -> str:
    return prompts.render_actions(actions) + "\n\nCall submit_answer with your choice."


class RecentHistory:
    """Answers from the most recent raw sessions that fit in answer_budget_chars.

    Makes no model calls during replay.
    """

    name = "recent"
    has_workspace = False

    def __init__(self, ctx: MethodContext):
        self.ctx = ctx
        self.sessions: list[Session] = []
        self._window = ""

    def observe(self, event: Event) -> None:
        self.sessions.append(event)

    def build_window(self) -> tuple[str, int]:
        budget = self.ctx.config.answer_budget_chars
        parts: list[str] = []
        used = 0
        for s in reversed(self.sessions):
            text = prompts.render_session(s)
            remaining = budget - used
            if len(text) > remaining:
                if remaining > 200:  # keep the tail of the next-older session
                    parts.append("[earlier part of this session omitted]\n" + text[-remaining:])
                    used += remaining
                break
            parts.append(text)
            used += len(text)
        n_full = sum(1 for p in parts if not p.startswith("[earlier"))
        return "\n\n".join(reversed(parts)), n_full

    def answer(self, probe: Probe, question: Question, probe_event_id: str) -> AnswerResult:
        window, n_full = self.build_window()
        self._window = window
        self.ctx.trace.emit("history_window", sessions_in_full=n_full, chars=len(window))
        submit = SubmitAnswer(question.letters)
        fixed = (
            f"<recent_history>\n{window}\n</recent_history>\n\n"
            + prompts.render_probe(probe, probe_event_id)
            + "\n\n"
            + prompts.render_question(question)
        )
        return answer_with_tools(
            ctx=self.ctx,
            executor=ToolExecutor([submit], self.ctx.trace),
            submit=submit,
            system=prompts.RECENT_ANSWER_SYSTEM.format(
                budget=self.ctx.config.answer_budget_chars, answer_task=prompts.ANSWER_TASK
            ),
            fixed_text=fixed,
            render_dynamic=_no_dynamic,
        )

    def memory_snapshot(self) -> dict[str, str]:
        return {"recent_history_window.md": self._window}

    def export_state(self) -> dict[str, Any]:
        return {"observed": len(self.sessions)}

    def restore_state(self, state: dict[str, Any], history: tuple[Session, ...]) -> None:
        self.sessions = list(history[: state["observed"]])


class RollingSummary:
    """One fixed-prompt model call per session rewrites a bounded summary."""

    name = "summary"
    has_workspace = False

    def __init__(self, ctx: MethodContext):
        self.ctx = ctx
        self.summary = ""
        self.max_chars = ctx.config.summary_max_chars
        self.system = prompts.SUMMARY_UPDATE_SYSTEM.format(
            tracking_guidance=prompts.TRACKING_GUIDANCE, max_chars=self.max_chars
        )

    def observe(self, event: Event) -> None:
        session = event
        request = ModelRequest(
            phase="summary_update",
            system=self.system,
            blocks=(
                TextBlock(
                    f"<current_summary>\n{self.summary or '(empty)'}\n</current_summary>\n\n"
                    + prompts.render_session(session)
                    + "\n\nWrite the complete updated summary."
                ),
            ),
            metadata={"event_id": session.event_id},
        )
        response = self.ctx.model.complete(request)
        new = response.text.strip()
        if response.stop_reason == "refusal" or not new:
            self.ctx.trace.emit("summary_unchanged", event_id=session.event_id, reason=response.stop_reason or "empty")
            return
        truncated = len(new) > self.max_chars
        new = new[: self.max_chars]
        diff = "".join(
            difflib.unified_diff(
                self.summary.splitlines(keepends=True), new.splitlines(keepends=True), "a/summary", "b/summary"
            )
        )
        self.ctx.trace.emit(
            "summary_update", event_id=session.event_id, chars=len(new), truncated=truncated, diff=diff
        )
        self.summary = new

    def answer(self, probe: Probe, question: Question, probe_event_id: str) -> AnswerResult:
        submit = SubmitAnswer(question.letters)
        fixed = (
            f"<summary>\n{self.summary or '(empty)'}\n</summary>\n\n"
            + prompts.render_probe(probe, probe_event_id)
            + "\n\n"
            + prompts.render_question(question)
        )
        return answer_with_tools(
            ctx=self.ctx,
            executor=ToolExecutor([submit], self.ctx.trace),
            submit=submit,
            system=prompts.SUMMARY_ANSWER_SYSTEM.format(answer_task=prompts.ANSWER_TASK),
            fixed_text=fixed,
            render_dynamic=_no_dynamic,
        )

    def memory_snapshot(self) -> dict[str, str]:
        return {"summary.md": self.summary}

    def export_state(self) -> dict[str, Any]:
        return {"summary": self.summary}

    def restore_state(self, state: dict[str, Any], history: tuple[Session, ...]) -> None:
        self.summary = state["summary"]
