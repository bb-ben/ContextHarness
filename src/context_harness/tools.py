"""Tool layer: tool definitions, a validating executor, and an approval seam.

Workspace tools take their file list from the Layout. Every call records which
files it touched (an access record), which the memory checks aggregate.

Domain tools that act in the world (e.g. future FoodBody actions) can subclass
`Tool` with `requires_approval=True`; the executor routes those through an
`ApprovalGate`. The default gate approves nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .entries import CHANGE_FIELD, CLOSED_STATUSES, ID_RE, LIVE_STATUSES, VALUE_MAX_CHARS, parse_file
from .models.base import ToolCall, ToolSpec
from .tracing import TraceWriter
from .workspace import ARCHIVE_FILE, Workspace, WorkspaceError


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str  # text returned to the model
    paths: tuple[str, ...] = ()  # workspace files this call read or wrote


class Tool:
    spec: ToolSpec
    kind: Literal["read", "write", "answer", "control", "other"] = "other"
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
            decision = self.approval_gate.review(call, tool) if tool.requires_approval else ApprovalDecision(True)
            if not decision.approved:
                result = ToolResult(False, f"Not approved: {decision.reason}")
            else:
                try:
                    result = tool.run(call.arguments)
                except WorkspaceError as e:
                    result = ToolResult(False, str(e))

        kind = tool.kind if tool is not None else "other"
        paths = result.paths or tuple(p for p in [call.arguments.get("path")] if isinstance(p, str))
        stats = self.trace.stats
        stats.tool_calls += 1
        if not result.ok:
            stats.tool_errors += 1
        elif kind == "write":
            stats.file_ops += 1
        elif kind == "read" and call.name in ("read_file", "lookup"):
            stats.file_reads += 1
        if kind in ("read", "write"):
            stats.access.append({"tool": call.name, "kind": kind, "paths": list(paths), "ok": result.ok})
        self.trace.emit(
            "tool_call",
            **context,
            tool=call.name,
            call_id=call.id,
            arguments=call.arguments,
            ok=result.ok,
            paths=list(paths),
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
        if expected == "array" and not (isinstance(value, list) and all(isinstance(v, str) for v in value)):
            return f"field {key!r} must be a list of strings"
        if "enum" in props[key] and value not in props[key]["enum"]:
            return f"field {key!r} must be one of {props[key]['enum']}"
    return None


def _schema(required: tuple[str, ...] | None = None, **props: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": list(props) if required is None else list(required),
        "additionalProperties": False,
    }


class ReadBudget:
    """Caps the total characters an agent may read in one phase (answer-time parity)."""

    def __init__(self, max_chars: int | None):
        self.max_chars = max_chars
        self.used = 0

    def charge(self, n: int) -> None:
        if self.max_chars is not None and self.used + n > self.max_chars:
            raise WorkspaceError(
                f"Read budget exhausted: reading {n} more characters would exceed the "
                f"{self.max_chars}-character limit for this phase ({self.used} used). "
                "Use lookup to fetch specific entries instead."
            )
        self.used += n


# ---------------------------------------------------------------- workspace tools


def _path(ws: Workspace) -> dict[str, Any]:
    return {"type": "string", "enum": list(ws.layout.paths), "description": "Workspace file path."}


class ListFiles(Tool):
    kind = "other"

    def __init__(self, ws: Workspace):
        self.ws = ws
        self.spec = ToolSpec("list_files", "List the memory files with their current size and limit.", _schema())

    def run(self, args: dict[str, Any]) -> ToolResult:
        return ToolResult(True, "\n".join(f"{f['path']}  ({f['chars']}/{f['limit']} chars)" for f in self.ws.list_files()))


class ReadFile(Tool):
    kind = "read"

    def __init__(self, ws: Workspace, budget: ReadBudget | None = None):
        self.ws, self.budget = ws, budget
        self.spec = ToolSpec("read_file", "Return the full contents of one memory file.", _schema(path=_path(ws)))

    def run(self, args: dict[str, Any]) -> ToolResult:
        content = self.ws.read(args["path"])
        if self.budget:
            self.budget.charge(len(content))
        return ToolResult(True, content, (args["path"],))


class Lookup(Tool):
    """Fetch specific entries by ID from any file, without reading whole files."""

    kind = "read"

    def __init__(self, ws: Workspace, budget: ReadBudget | None = None):
        self.ws, self.budget = ws, budget
        self.spec = ToolSpec(
            "lookup",
            "Return the lines for the given entry IDs (e.g. P-27.1, O-14.2) from whichever files hold them. "
            "An observation is returned with the E line it sits under.",
            _schema(ids={"type": "array", "items": {"type": "string"}, "description": "Entry IDs to fetch."}),
        )

    def run(self, args: dict[str, Any]) -> ToolResult:
        wanted = {i.strip() for i in args["ids"] if ID_RE.fullmatch(i.strip())}
        found: dict[str, str] = {}
        paths = set()
        for spec in self.ws.layout.files:
            if not spec.entry_types:
                continue
            lines = self.ws.read(spec.path).splitlines()
            entries, _ = parse_file(spec.path, "\n".join(lines), spec.entry_types)
            by_id = {e.id: e for e in entries}
            for entry_id in wanted & by_id.keys():
                e = by_id[entry_id]
                text = lines[e.line - 1]
                if e.parent and e.parent in by_id:
                    text = lines[by_id[e.parent].line - 1] + "\n" + text
                found[entry_id] = f"[{spec.path}]\n{text}"
                paths.add(spec.path)
        missing = sorted(wanted - found.keys())
        body = "\n".join(found[i] for i in sorted(found)) or "(no matching entries)"
        if missing:
            body += f"\nNot found: {', '.join(missing)}"
        if self.budget:
            self.budget.charge(len(body))
        return ToolResult(True, body, tuple(sorted(paths)))


def _describe(change, verb: str) -> str:
    """Tool result text: what happened, including which entry IDs appeared or disappeared."""
    text = f"{verb} {change.path} ({change.chars_after} chars)."
    if change.added_ids:
        text += f" New entries: {', '.join(change.added_ids)}."
    if change.removed_ids:
        text += (f" Warning: this removed {', '.join(change.removed_ids)}. Entries are never deleted; "
                 "supersede, retract or archive them instead.")
    return text


class WriteFile(Tool):
    kind = "write"

    def __init__(self, ws: Workspace):
        self.ws = ws
        self.spec = ToolSpec(
            "write_file",
            "Replace the entire contents of one memory file. Rejected if over the file's limit.",
            _schema(path=_path(ws), content={"type": "string"}),
        )

    def run(self, args: dict[str, Any]) -> ToolResult:
        change = self.ws.write(args["path"], args["content"])
        return ToolResult(True, _describe(change, "Wrote"), (change.path,))


class EditFile(Tool):
    kind = "write"

    def __init__(self, ws: Workspace):
        self.ws = ws
        self.spec = ToolSpec(
            "edit_file",
            "Replace one exact, unique occurrence of old_text with new_text in a memory file.",
            _schema(path=_path(ws), old_text={"type": "string"}, new_text={"type": "string"}),
        )

    def run(self, args: dict[str, Any]) -> ToolResult:
        change = self.ws.edit(args["path"], args["old_text"], args["new_text"])
        return ToolResult(True, _describe(change, "Edited"), (change.path,))


class AppendFile(Tool):
    kind = "write"

    def __init__(self, ws: Workspace):
        self.ws = ws
        self.spec = ToolSpec(
            "append_file",
            "Append text to the end of a memory file (a newline is inserted if needed).",
            _schema(path=_path(ws), text={"type": "string"}),
        )

    def run(self, args: dict[str, Any]) -> ToolResult:
        change = self.ws.append(args["path"], args["text"])
        return ToolResult(True, _describe(change, "Appended to"), (change.path,))


class Archive(Tool):
    """Move closed entries, unchanged and with their notes, into memory/archive.md."""

    kind = "write"

    def __init__(self, ws: Workspace):
        self.ws = ws
        self.spec = ToolSpec(
            "archive",
            "Move closed entries (superseded, expired, refuted, achieved, ...) out of their state file into "
            "memory/archive.md, unchanged. Use it when a state file is getting full. Live entries are refused.",
            _schema(ids={"type": "array", "items": {"type": "string"}, "description": "Entry IDs to archive."}),
        )

    def run(self, args: dict[str, Any]) -> ToolResult:
        wanted = {i.strip() for i in args["ids"]}
        moves: dict[str, list[tuple[int, int]]] = {}  # path -> [(first line, last line)] (0-based, inclusive)
        found = set()
        for spec in self.ws.layout.files:
            if spec.path == ARCHIVE_FILE or not spec.entry_types or not set(spec.entry_types) - {"E", "O"}:
                continue
            text = self.ws.read(spec.path)
            lines = text.splitlines()
            entries, _ = parse_file(spec.path, text, spec.entry_types)
            for e in entries:
                if e.id not in wanted:
                    continue
                if e.status not in CLOSED_STATUSES:
                    raise WorkspaceError(f"{e.id} is {e.status}; only closed entries can be archived")
                end = e.line - 1
                while end + 1 < len(lines) and lines[end + 1].startswith((" ", "\t")) and lines[end + 1].strip():
                    end += 1  # indented notes travel with their entry
                moves.setdefault(spec.path, []).append((e.line - 1, end))
                found.add(e.id)
        if missing := sorted(wanted - found):
            raise WorkspaceError(f"Not found in any state file: {', '.join(missing)}")
        moved: list[str] = []
        for path, spans in moves.items():
            lines = self.ws.read(path).splitlines()
            drop = {i for start, end in spans for i in range(start, end + 1)}
            moved += [lines[i] for start, end in sorted(spans) for i in range(start, end + 1)]
            kept = [line for i, line in enumerate(lines) if i not in drop]
            self.ws.write(path, "\n".join(kept) + "\n", check_ids=False)
        self.ws.append(ARCHIVE_FILE, "\n".join(moved), check_ids=False)
        return ToolResult(True, f"Archived {len(found)} entries.", tuple(sorted(moves)) + (ARCHIVE_FILE,))


class Supersede(Tool):
    """Revise a belief in one step: write the new entry and close the old one, linked both ways.

    By hand this takes an edit and an append that must agree, and in the v3.2 runs one
    revision in six left the links one-sided or edited the old entry's date.
    """

    kind = "write"

    def __init__(self, ws: Workspace):
        self.ws = ws
        self.spec = ToolSpec(
            "supersede",
            "Replace a live entry with a new one. Give the old entry's ID and the complete new entry line; "
            "the new entry is added to the file for its type with supersedes: <old ID>, and the old one is "
            "marked superseded with until: and superseded_by:. A preference or fact replacing another needs "
            f"changed: <what is different>, and its statement must stay under {VALUE_MAX_CHARS} characters.",
            _schema(id={"type": "string", "description": "ID of the live entry being replaced, e.g. P-10.1"},
                    new_entry={"type": "string", "description": "The complete new entry line, starting with '- '"}),
        )

    def _state_files(self):
        return [s for s in self.ws.layout.files
                if s.path != ARCHIVE_FILE and s.entry_types and set(s.entry_types) - {"E", "O"}]

    def run(self, args: dict[str, Any]) -> ToolResult:
        old_id, line = args["id"].strip(), args["new_entry"].strip()
        old = old_spec = None
        for spec in self._state_files():
            entries, _ = parse_file(spec.path, self.ws.read(spec.path), spec.entry_types)
            old = next((e for e in entries if e.id == old_id), None)
            if old:
                old_spec = spec
                break
        if old is None:
            raise WorkspaceError(f"{old_id} is not in any state file")
        if old.status not in LIVE_STATUSES | {"paused"}:
            raise WorkspaceError(f"{old_id} is {old.status}; only live entries can be superseded")
        if "\n" in line:
            raise WorkspaceError("new_entry must be a single entry line")
        new_type = line.lstrip("- ")[:1]
        new_spec = next((s for s in self._state_files() if new_type in s.entry_types), None)
        if new_spec is None:
            raise WorkspaceError(f"No state file holds {new_type} entries")
        parsed, problems = parse_file(new_spec.path, line, new_spec.entry_types)
        if problems or len(parsed) != 1:
            raise WorkspaceError(f"new_entry did not parse as one entry: {problems[0].detail if problems else line[:80]}")
        new = parsed[0]
        if new.status not in LIVE_STATUSES:
            raise WorkspaceError(f"the new entry must be live (current, active or open), not {new.status}")
        if new.type in ("P", "F") and new.type == old.type:
            if not new.fields.get(CHANGE_FIELD):
                raise WorkspaceError(f"Add {CHANGE_FIELD}: <what is different from {old_id}> to the new entry")
            if len(new.statement) > VALUE_MAX_CHARS:
                raise WorkspaceError(
                    f"The statement is {len(new.statement)} characters; keep it under {VALUE_MAX_CHARS}. Describe "
                    "the person now: keep what still holds and drop what the new evidence replaces.")
        segments = [s.strip() for s in line.split("|")]
        if old_id not in new.refs().get("supersedes", []):
            segments.insert(len(segments) - 1, f"supersedes: {old_id}")
        new_line = " | ".join(segments)

        old_text = self.ws.read(old_spec.path)
        old_lines = old_text.splitlines()
        head = [s.strip() for s in old_lines[old.line - 1].split("|")]
        head[1] = "superseded"
        head = [s for s in head if not s.startswith(("until:", "superseded_by:"))]
        head[-1:-1] = [f"until: {new.date}", f"superseded_by: {new.id}"]
        old_lines[old.line - 1] = " | ".join(head)
        closed = "\n".join(old_lines) + "\n"
        if new_spec.path == old_spec.path:
            change = self.ws.write(old_spec.path, closed + new_line + "\n")
        else:
            change = self.ws.append(new_spec.path, new_line)  # fails first, before the old entry is touched
            self.ws.write(old_spec.path, closed, check_ids=False)
        return ToolResult(True, f"Superseded {old_id} with {new.id} in {new_spec.path}. {_describe(change, 'Wrote')}",
                          tuple(sorted({old_spec.path, new_spec.path})))


class Finish(Tool):
    """Ends the current event after this turn's other tool calls have run."""

    kind = "control"

    def __init__(self):
        self.called = False
        self.spec = ToolSpec(
            "finish",
            "Call this together with your last edits for the event to end it in the same turn.",
            _schema(required=(), note={"type": "string", "description": "Optional one-line note."}),
        )

    def run(self, args: dict[str, Any]) -> ToolResult:
        self.called = True
        return ToolResult(True, "Event finished.")


def workspace_tools(ws: Workspace, writable: bool, read_budget: ReadBudget | None = None) -> list[Tool]:
    tools: list[Tool] = [ListFiles(ws), ReadFile(ws, read_budget), Lookup(ws, read_budget)]
    if writable:
        tools += [WriteFile(ws), EditFile(ws), AppendFile(ws), Supersede(ws), Archive(ws)]
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
