"""Tool layer: tool definitions, a validating executor, and an approval seam.

Milestone 1 ships only workspace tools and `submit_answer`. Domain tools that act
in the world can later subclass `Tool` with `requires_approval=True`; the
executor routes those through an `ApprovalGate`. The default gate approves
nothing, so no automatic-approval behavior exists yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .models.base import ToolCall, ToolSpec
from .tracing import TraceWriter
from .workspace import WORKSPACE_FILES, Workspace, WorkspaceError


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str  # text returned to the model


class Tool:
    spec: ToolSpec
    kind: Literal["read", "write", "answer", "other"] = "other"
    requires_approval: bool = False

    def run(self, args: dict[str, Any]) -> ToolResult:
        raise NotImplementedError


@dataclass(frozen=True)
class ApprovalDecision:
    approved: bool
    reason: str = ""


class ApprovalGate(Protocol):
    def review(self, call: ToolCall, tool: Tool) -> ApprovalDecision: ...


class NoApprover:
    """Denies any tool that requires approval. Workspace tools don't require it."""

    def review(self, call: ToolCall, tool: Tool) -> ApprovalDecision:
        return ApprovalDecision(False, "No approver is configured for this tool.")


class ToolExecutor:
    def __init__(self, tools: list[Tool], trace: TraceWriter, approval_gate: ApprovalGate | None = None):
        self.tools = {t.spec.name: t for t in tools}
        self.trace = trace
        self.approval_gate = approval_gate or NoApprover()

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(t.spec for t in self.tools.values())

    def execute(self, call: ToolCall, **context: Any) -> ToolResult:
        tool = self.tools.get(call.name)
        if tool is None:
            result = ToolResult(False, f"Unknown tool {call.name!r}. Available: {', '.join(self.tools)}")
        elif (problem := _validate(call.arguments, tool.spec.input_schema)) is not None:
            result = ToolResult(False, f"Invalid arguments: {problem}")
        else:
            decision = (
                self.approval_gate.review(call, tool) if tool.requires_approval else ApprovalDecision(True)
            )
            if not decision.approved:
                result = ToolResult(False, f"Not approved: {decision.reason}")
            else:
                try:
                    result = tool.run(call.arguments)
                except WorkspaceError as e:
                    result = ToolResult(False, str(e))

        stats = self.trace.stats
        stats.tool_calls += 1
        if not result.ok:
            stats.tool_errors += 1
        elif tool is not None and tool.kind == "write":
            stats.file_ops += 1
        elif tool is not None and tool.kind == "read" and call.name == "read_file":
            stats.file_reads += 1
        self.trace.emit(
            "tool_call",
            **context,
            tool=call.name,
            call_id=call.id,
            arguments=call.arguments,
            ok=result.ok,
            result=result.content if len(result.content) <= 2000 else result.content[:2000] + "…",
            result_chars=len(result.content),
        )
        return result


def _validate(args: Any, schema: dict[str, Any]) -> str | None:
    if not isinstance(args, dict):
        return "arguments must be an object"
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in args:
            return f"missing required field {key!r}"
    for key, value in args.items():
        if key not in props:
            return f"unexpected field {key!r}"
        expected = props[key].get("type")
        if expected == "string" and not isinstance(value, str):
            return f"field {key!r} must be a string"
        if "enum" in props[key] and value not in props[key]["enum"]:
            return f"field {key!r} must be one of {props[key]['enum']}"
    return None


# ---------------------------------------------------------------- workspace tools

_PATH = {"type": "string", "enum": list(WORKSPACE_FILES), "description": "Workspace file path."}


def _schema(**props: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


class ReadBudget:
    """Caps the total characters an agent may read in one phase (answer-time parity)."""

    def __init__(self, max_chars: int | None):
        self.max_chars = max_chars
        self.used = 0

    def charge(self, n: int) -> None:
        if self.max_chars is not None and self.used + n > self.max_chars:
            raise WorkspaceError(
                f"Read budget exhausted: reading {n} more characters would exceed the "
                f"{self.max_chars}-character limit for this phase ({self.used} used)."
            )
        self.used += n


class ListFiles(Tool):
    kind = "read"

    def __init__(self, ws: Workspace):
        self.ws = ws
        self.spec = ToolSpec(
            "list_files",
            "List the memory files with their current size and character limit.",
            _schema(),
        )

    def run(self, args: dict[str, Any]) -> ToolResult:
        lines = [f"{f['path']}  ({f['chars']}/{f['limit']} chars)" for f in self.ws.list_files()]
        return ToolResult(True, "\n".join(lines))


class ReadFile(Tool):
    kind = "read"

    def __init__(self, ws: Workspace, budget: ReadBudget | None = None):
        self.ws, self.budget = ws, budget
        self.spec = ToolSpec("read_file", "Return the full contents of one memory file.", _schema(path=_PATH))

    def run(self, args: dict[str, Any]) -> ToolResult:
        content = self.ws.read(args["path"])
        if self.budget:
            self.budget.charge(len(content))
        return ToolResult(True, content)


class WriteFile(Tool):
    kind = "write"

    def __init__(self, ws: Workspace):
        self.ws = ws
        self.spec = ToolSpec(
            "write_file",
            "Replace the entire contents of one memory file. Rejected if over the file's limit.",
            _schema(path=_PATH, content={"type": "string"}),
        )

    def run(self, args: dict[str, Any]) -> ToolResult:
        change = self.ws.write(args["path"], args["content"])
        return ToolResult(True, f"Wrote {change.path} ({change.chars_after} chars).")


class EditFile(Tool):
    kind = "write"

    def __init__(self, ws: Workspace):
        self.ws = ws
        self.spec = ToolSpec(
            "edit_file",
            "Replace one exact, unique occurrence of old_text with new_text in a memory file.",
            _schema(path=_PATH, old_text={"type": "string"}, new_text={"type": "string"}),
        )

    def run(self, args: dict[str, Any]) -> ToolResult:
        change = self.ws.edit(args["path"], args["old_text"], args["new_text"])
        return ToolResult(True, f"Edited {change.path} ({change.chars_after} chars).")


class AppendFile(Tool):
    kind = "write"

    def __init__(self, ws: Workspace):
        self.ws = ws
        self.spec = ToolSpec(
            "append_file",
            "Append text to the end of a memory file (a newline is inserted if needed).",
            _schema(path=_PATH, text={"type": "string"}),
        )

    def run(self, args: dict[str, Any]) -> ToolResult:
        change = self.ws.append(args["path"], args["text"])
        return ToolResult(True, f"Appended to {change.path} ({change.chars_after} chars).")


def workspace_tools(ws: Workspace, writable: bool, read_budget: ReadBudget | None = None) -> list[Tool]:
    tools: list[Tool] = [ListFiles(ws), ReadFile(ws, read_budget)]
    if writable:
        tools += [WriteFile(ws), EditFile(ws), AppendFile(ws)]
    return tools


class SubmitAnswer(Tool):
    kind = "answer"

    def __init__(self, letters: tuple[str, ...]):
        self.letter: str | None = None
        self.spec = ToolSpec(
            "submit_answer",
            "Submit your final choice: the letter of the response that best fits the user.",
            _schema(letter={"type": "string", "enum": list(letters)}),
        )

    def run(self, args: dict[str, Any]) -> ToolResult:
        self.letter = args["letter"]
        return ToolResult(True, f"Answer {self.letter} recorded.")
