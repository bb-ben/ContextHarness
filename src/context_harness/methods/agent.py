"""The agent under test: keeps its own working context and state files under the
contract in agent.md."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .. import prompts
from ..dimensions import HYPOTHESES_FILE, PREFERENCES_FILE, dimension_rows, render_dimensions
from ..entries import Entry, parse_file
from ..tools import Finish, ReadBudget, SubmitAnswer, ToolExecutor, workspace_tools
from ..types import AnswerResult, Event, Probe, Question, Session
from ..workspace import ARCHIVE_FILE, CONTEXT_FILE, EPISODES_FILE, TRANSIENT_FILE, FileChange, Workspace, WorkspaceError
from .base import ActionLog, MethodContext, answer_with_tools, run_step_loop


class WorkspaceAgent:
    name = "agent"
    has_workspace = True

    def __init__(self, ctx: MethodContext):
        self.ctx = ctx
        cfg = ctx.config
        self._phase = "setup"
        self._event_id: str | None = None
        root = ctx.item_dir / "workspace"
        if ctx.resume:
            self.ws = Workspace(root, cfg.layout, on_change=self._on_change)
        else:
            self.ws = Workspace.create(root, cfg.layout, on_change=self._on_change)
            self.ws.save_snapshot(ctx.item_dir / "snapshots" / "initial")
        self.system = prompts.load_contract(cfg.contract_path)
        self.finish = Finish()
        self.update_executor = ToolExecutor(workspace_tools(self.ws, writable=True) + [self.finish], ctx.trace)

    def _on_change(self, change: FileChange) -> None:
        self.ctx.trace.emit("file_diff", phase=self._phase, event_id=self._event_id, **asdict(change))

    def due(self, date: str) -> list[tuple[Entry, str]]:
        """Active transient entries whose expires/review date has arrived (the layer's reminder)."""
        if self.ws.layout.spec(TRANSIENT_FILE) is None:
            return []
        text = self.ws.read(TRANSIENT_FILE)
        lines = text.splitlines()
        entries, _ = parse_file(TRANSIENT_FILE, text, ("T",))
        out = []
        for e in entries:
            end = e.fields.get("expires") or e.fields.get("review")
            if e.status == "active" and end and end <= date:
                out.append((e, lines[e.line - 1]))
        return out

    def preloaded(self) -> dict[str, str]:
        if not self.ctx.config.preload_state:
            return {}
        return {path: self.ws.read(path) for path in self.ws.layout.preloaded}

    def _render_state(self, actions: str) -> list[str]:
        limit = self.ws.limit(CONTEXT_FILE)
        blocks = [prompts.render_working_context(self.ws.read(CONTEXT_FILE), limit)]
        if state := self.preloaded():
            blocks.append(prompts.render_state(state))
            if PREFERENCES_FILE in state:  # the structured view, derived from pre-loaded state
                rows = dimension_rows(self.ws.layout, state[PREFERENCES_FILE], state.get(HYPOTHESES_FILE, ""))
                blocks.append(render_dimensions(rows))
        return blocks + [prompts.render_workspace(self.ws.list_files())]

    def _rollover_episodes(self) -> None:
        """Move the oldest episode blocks, unchanged, into the archive when the log nears its limit."""
        layout, cfg = self.ws.layout, self.ctx.config
        if layout.spec(EPISODES_FILE) is None or layout.spec(ARCHIVE_FILE) is None:
            return
        text, limit = self.ws.read(EPISODES_FILE), self.ws.limit(EPISODES_FILE)
        if len(text) <= cfg.episode_rollover_at * limit:
            return
        segments: list[list[str]] = []  # each is a block: an E line plus its indented lines, or a lone line
        for line in text.splitlines():
            if line.startswith("- E-") or not segments or not line.startswith((" ", "\t")):
                segments.append([line])
            else:
                segments[-1].append(line)
        target, size, moved = cfg.episode_rollover_to * limit, len(text), []
        for seg in segments:
            if size <= target:
                break
            if seg[0].startswith("- E-"):
                moved.append(seg)
                size -= sum(len(line) + 1 for line in seg)
        if not moved:
            return
        kept = [line for seg in segments if seg not in moved for line in seg]
        phase, self._phase = self._phase, "rollover"
        try:
            self.ws.append(ARCHIVE_FILE, "\n".join(line for seg in moved for line in seg), check_ids=False)
            self.ws.write(EPISODES_FILE, "\n".join(kept) + "\n", check_ids=False)
            self.ctx.trace.emit("episode_rollover", event_id=self._event_id, episodes=len(moved),
                                first=moved[0][0].split(" |")[0][2:], last=moved[-1][0].split(" |")[0][2:])
        except WorkspaceError as e:
            self.ctx.trace.emit("episode_rollover_failed", event_id=self._event_id, error=str(e))
        finally:
            self._phase = phase

    def observe(self, event: Event) -> None:
        self._phase, self._event_id = "memory_update", event.event_id
        self.finish.called = False
        date = event.date[:10]
        self._rollover_episodes()

        def render(actions: str, calls_left: int) -> str:
            return "\n\n".join(
                self._render_state(actions)
                + [
                    prompts.render_due(self.due(date)),
                    prompts.render_actions(actions),
                    f"{prompts.UPDATE_INSTRUCTION} Model calls remaining after this one: {calls_left}.",
                ]
            )

        outcome = run_step_loop(
            model=self.ctx.model,
            executor=self.update_executor,
            phase="memory_update",
            system=self.system,
            fixed_text=prompts.render_event(event),
            render_dynamic=render,
            max_steps=self.ctx.config.max_update_steps,
            log=ActionLog(self.ctx.config.action_log_max_chars),
            metadata={"event_id": event.event_id},
            stop_when=lambda: self.finish.called,
        )
        self.ctx.trace.emit("update_loop_end", event_id=event.event_id, stop=outcome.stop, steps=outcome.steps)

    def answer(self, probe: Probe, question: Question, probe_event_id: str) -> AnswerResult:
        self._phase, self._event_id = "answer", probe_event_id
        cfg = self.ctx.config
        # Everything shown without a read (context.md, pre-loaded state, the dimensions view)
        # counts against the shared answer budget; reads draw on what remains.
        always = sum(len(block) for block in self._render_state(""))
        budget = ReadBudget(max(0, cfg.answer_budget_chars - always))
        submit = SubmitAnswer(question.letters)
        executor = ToolExecutor(workspace_tools(self.ws, writable=False, read_budget=budget) + [submit], self.ctx.trace)

        def render(actions: str, calls_left: int) -> str:
            return "\n\n".join(
                self._render_state(actions)
                + [
                    prompts.render_actions(actions),
                    " ".join(filter(None, [
                        prompts.answer_instruction(budget.max_chars - budget.used),
                        prompts.ANSWER_STRATEGIES[cfg.answer_strategy],
                        f"Model calls remaining after this one: {calls_left}.",
                    ])),
                ]
            )

        result = answer_with_tools(
            ctx=self.ctx,
            executor=executor,
            submit=submit,
            system=self.system,
            fixed_text=prompts.render_probe(probe, probe_event_id) + "\n\n" + prompts.render_question(question),
            render_dynamic=render,
        )
        result.details["read_chars"] = budget.used
        return result

    def memory_snapshot(self) -> dict[str, str]:
        return self.ws.snapshot()

    def export_state(self) -> dict[str, Any]:
        return {}  # everything lives in the workspace files

    def restore_state(self, state: dict[str, Any], history: tuple[Session, ...]) -> None:
        pass
