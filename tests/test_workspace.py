"""Workspace isolation, allowlist, budgets, diffs, executor, and the raw log."""

import json
import os
import stat

import pytest
from conftest import run_config

from context_harness.models.base import ToolCall
from context_harness.models.fake import FakeModel
from context_harness.orchestrator import run
from context_harness.tools import Tool, ToolExecutor, ToolResult, workspace_tools
from context_harness.models.base import ToolSpec
from context_harness.tracing import RawSessionLog, TraceWriter
from context_harness.workspace import WORKSPACE_FILES, Workspace, WorkspaceBudgets, WorkspaceError


@pytest.fixture
def ws(tmp_path):
    changes = []
    w = Workspace.create(tmp_path / "item" / "workspace", WorkspaceBudgets(500, 1000), on_change=changes.append)
    w.changes = changes
    (tmp_path / "item" / "raw_sessions.jsonl").write_text("SECRET")
    return w


def test_templates_explain_purpose_without_content(ws):
    files = sorted(str(p.relative_to(ws.root)) for p in ws.root.rglob("*") if p.is_file())
    assert files == sorted(WORKSPACE_FILES)
    for path in WORKSPACE_FILES:
        text = ws.read(path)
        assert "Purpose:" in text and "Limit:" in text
        assert len(text.splitlines()) == 3  # title, blank, purpose comment: no pre-filled state


@pytest.mark.parametrize(
    "bad",
    ["../raw_sessions.jsonl", "/etc/passwd", "user_state/../context.md", "raw_sessions.jsonl",
     "trace.jsonl", "user_state/new.md", "", "USER_STATE/goals.md", "user_state"],
)
def test_allowlist_rejects_everything_else(ws, bad):
    with pytest.raises(WorkspaceError):
        ws.read(bad)
    with pytest.raises(WorkspaceError):
        ws.write(bad, "x")


def test_symlinked_file_is_refused(ws, tmp_path):
    target = ws.root / "user_state" / "goals.md"
    target.unlink()
    target.symlink_to(tmp_path / "item" / "raw_sessions.jsonl")
    with pytest.raises(WorkspaceError):
        ws.read("user_state/goals.md")


def test_edit_append_budget_and_diffs(ws):
    ws.write("user_state/goals.md", "alpha\nbeta\n")
    ws.edit("user_state/goals.md", "beta", "gamma")
    ws.append("user_state/goals.md", "delta")
    assert ws.read("user_state/goals.md") == "alpha\ngamma\ndelta"
    with pytest.raises(WorkspaceError, match="exactly once"):
        ws.edit("user_state/goals.md", "missing", "x")
    with pytest.raises(WorkspaceError, match="over its limit of 500"):
        ws.write("context.md", "x" * 501)
    assert [c.op for c in ws.changes] == ["write_file", "edit_file", "append_file"]
    assert "-beta" in ws.changes[1].diff and "+gamma" in ws.changes[1].diff


def test_executor_validation_and_read_only_answer_tools(ws, tmp_path):
    trace = TraceWriter(tmp_path / "trace.jsonl")
    read_only = ToolExecutor(workspace_tools(ws, writable=False), trace)
    assert {s.name for s in read_only.specs} == {"list_files", "read_file"}
    r = read_only.execute(ToolCall("1", "write_file", {"path": "context.md", "content": "x"}))
    assert not r.ok and "Unknown tool" in r.content
    r = read_only.execute(ToolCall("2", "read_file", {"path": "../raw_sessions.jsonl"}))
    assert not r.ok and "must be one of" in r.content
    r = read_only.execute(ToolCall("3", "read_file", {"path": "context.md", "extra": 1}))
    assert not r.ok
    assert trace.stats.tool_errors == 3 and trace.stats.tool_calls == 3


def test_tools_requiring_approval_are_denied_by_default(tmp_path):
    class Launch(Tool):
        requires_approval = True
        spec = ToolSpec("launch", "An external action.", {"type": "object", "properties": {}})

        def run(self, args):
            raise AssertionError("must not run without approval")

    executor = ToolExecutor([Launch()], TraceWriter(tmp_path / "t.jsonl"))
    result = executor.execute(ToolCall("1", "launch", {}))
    assert result == ToolResult(False, "Not approved: No approver is configured for this tool.")


def test_items_get_isolated_workspaces(items, tmp_path):
    model = FakeModel()
    run(items, model, tmp_path / "run", run_config("agent"), progress=lambda *_: None)
    item_dirs = sorted((tmp_path / "run" / "items").iterdir())
    assert len(item_dirs) == 2
    first_obs = (item_dirs[0] / "workspace" / "user_state" / "observations.md").read_text()
    second_obs = (item_dirs[1] / "workspace" / "user_state" / "observations.md").read_text()
    assert "Tokyo" in first_obs and "Tokyo" not in second_obs
    # The second item's first model call starts from the template, not item 1's memory.
    first_call_item2 = next(r for r in model.requests if "ferns" in r.blocks[0].text)
    assert "Sessions seen" not in first_call_item2.blocks[1].text


def test_raw_session_log_is_immutable_and_verifiable(evolved_item, tmp_path):
    path = tmp_path / "raw.jsonl"
    log = RawSessionLog(path)
    for s in evolved_item.episode.sessions:
        log.append(s)
    log.close()
    assert not os.stat(path).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    assert RawSessionLog.verify(path)
    with pytest.raises(FileExistsError):
        RawSessionLog(path)
    lines = path.read_text().splitlines()
    record = json.loads(lines[1])
    record["turns"][0]["content"] = "tampered"
    lines[1] = json.dumps(record)
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_text("\n".join(lines) + "\n")
    assert not RawSessionLog.verify(tampered)
