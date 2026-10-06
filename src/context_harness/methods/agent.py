"""The test agent: maintains its own working context and user-state files."""

from __future__ import annotations

from dataclasses import asdict

from .. import prompts
from ..tools import ReadBudget, SubmitAnswer, ToolExecutor, workspace_tools
from ..types import AnswerResult, Probe, Question, Session
from ..workspace import CONTEXT_FILE, FileChange, Workspace
from .base import ActionLog, MethodContext, answer_with_tools, run_step_loop


class WorkspaceAgent:
    name = "agent"
    has_workspace = True

    def __init__(self, ctx: MethodContext):
        self.ctx = ctx
        cfg = ctx.config
        self._phase = "setup"
        self._session_index: int | None = None
        self.ws = Workspace.create(ctx.item_dir / "workspace", cfg.budgets, on_change=self._on_change)
        self.ws.save_snapshot(ctx.item_dir / "snapshots" / "initial")
        self.update_executor = ToolExecutor(workspace_tools(self.ws, writable=True), ctx.trace)
        self.update_system = prompts.agent_update_system(
            cfg.budgets.context_max_chars, cfg.max_update_steps
        )

    def _on_change(self, change: FileChange) -> None:
        self.ctx.trace.emit("file_diff", phase=self._phase, session=self._session_index, **asdict(change))

    def _dynamic(self, instruction: str):
        """Build the per-step block from current state: context.md is re-read every call."""

        def render(actions: str, calls_left: int) -> str:
            limit = self.ctx.config.budgets.context_max_chars
            return "\n\n".join(
                [
                    prompts.render_working_context(self.ws.read(CONTEXT_FILE), limit),
                    prompts.render_workspace(self.ws.list_files()),
                    prompts.render_actions(actions),
                    f"{instruction} Model calls remaining after this one: {calls_left}.",
                ]
            )

        return render

    def observe_session(self, session: Session) -> None:
        self._phase, self._session_index = "memory_update", session.index
        outcome = run_step_loop(
            model=self.ctx.model,
            executor=self.update_executor,
            phase="memory_update",
            system=self.update_system,
            fixed_text=prompts.render_session(session),
            render_dynamic=self._dynamic(
                "Update the memory files as needed for the session above, or reply without "
                "calling tools if nothing needs to change."
            ),
            max_steps=self.ctx.config.max_update_steps,
            log=ActionLog(self.ctx.config.action_log_max_chars),
            metadata={"session": session.index},
        )
        self.ctx.trace.emit("update_loop_end", session=session.index, stop=outcome.stop, steps=outcome.steps)

    def answer(self, probe: Probe, question: Question) -> AnswerResult:
        self._phase, self._session_index = "answer", None
        cfg = self.ctx.config
        # context.md is always visible; reads draw on what remains of the shared budget.
        budget = ReadBudget(max(0, cfg.answer_budget_chars - len(self.ws.read(CONTEXT_FILE))))
        submit = SubmitAnswer(question.letters)
        executor = ToolExecutor(
            workspace_tools(self.ws, writable=False, read_budget=budget) + [submit], self.ctx.trace
        )
        result = answer_with_tools(
            ctx=self.ctx,
            executor=executor,
            submit=submit,
            system=prompts.AGENT_ANSWER_SYSTEM.format(
                read_budget=budget.max_chars, answer_task=prompts.ANSWER_TASK
            ),
            fixed_text=prompts.render_probe(probe) + "\n\n" + prompts.render_question(question),
            render_dynamic=self._dynamic(
                "Read any memory files you need, then call submit_answer with your choice."
            ),
        )
        result.details["read_chars"] = budget.used
        return result

    def memory_snapshot(self) -> dict[str, str]:
        return self.ws.snapshot()
