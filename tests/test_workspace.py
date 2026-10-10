"""Workspace layout, isolation, allowlist, limits, diffs, tools, executor, and the raw log."""

import json
import os
import stat

import pytest
from conftest import run_config

from context_harness.models.base import ToolCall, ToolSpec
from context_harness.models.fake import FakeModel
from context_harness.orchestrator import run
from context_harness.tools import Finish, Lookup, ReadBudget, Tool, ToolExecutor, ToolResult, workspace_tools
from context_harness.tracing import RawSessionLog, TraceWriter
from context_harness.workspace import CONTEXT_FILE, HORIZONBENCH, WITH_DECISIONS, Workspace, WorkspaceError


@pytest.fixture
def ws(tmp_path):
    changes = []
    layout = HORIZONBENCH.with_limit(CONTEXT_FILE, 500).with_limit("state/goals.md", 1000)
    w = Workspace.create(tmp_path / "item" / "workspace", layout, on_change=changes.append)
    w.changes = changes
    (tmp_path / "item" / "raw_sessions.jsonl").write_text("SECRET")
    return w


def test_layout_files_and_templates(ws):
    files = sorted(str(p.relative_to(ws.root)) for p in ws.root.rglob("*") if p.is_file())
    assert files == sorted(HORIZONBENCH.paths)
    assert "memory/decisions.md" not in HORIZONBENCH.paths and "memory/decisions.md" in WITH_DECISIONS.paths
    for path in HORIZONBENCH.paths:
        text = ws.read(path)
        assert not any(line.startswith("- ") for line in text.splitlines())  # no pre-filled entries
    assert "as of (no events yet)" in ws.read(CONTEXT_FILE)


@pytest.mark.parametrize(
    "bad",
    ["../raw_sessions.jsonl", "/etc/passwd", "state/../context.md", "raw_sessions.jsonl", "trace.jsonl",
     "state/new.md", "", "STATE/goals.md", "state", "user_state/goals.md", "memory/decisions.md"],
)
def test_allowlist_rejects_everything_else(ws, bad):
    with pytest.raises(WorkspaceError):
        ws.read(bad)
    with pytest.raises(WorkspaceError):
        ws.write(bad, "x")


def test_symlinked_file_is_refused(ws, tmp_path):
    target = ws.root / "state" / "goals.md"
    target.unlink()
    target.symlink_to(tmp_path / "item" / "raw_sessions.jsonl")
    with pytest.raises(WorkspaceError):
        ws.read("state/goals.md")


def test_edit_append_limits_and_diffs(ws):
    ws.write("state/goals.md", "alpha\nbeta\n")
    ws.edit("state/goals.md", "beta", "gamma")
    ws.append("state/goals.md", "delta")
    assert ws.read("state/goals.md") == "alpha\ngamma\ndelta"
    with pytest.raises(WorkspaceError, match="exactly once"):
        ws.edit("state/goals.md", "missing", "x")
    with pytest.raises(WorkspaceError, match="over its limit of 500"):
        ws.write(CONTEXT_FILE, "x" * 501)
    assert [c.op for c in ws.changes] == ["write_file", "edit_file", "append_file"]
    assert "-beta" in ws.changes[1].diff and "+gamma" in ws.changes[1].diff


def test_lookup_returns_entries_with_their_episode_and_charges_budget(ws, tmp_path):
    ws.write("memory/episodes.md", "# Episodes\n- E-3 | 2025-10-03 | source: S-3 | Chat.\n  - O-3.1 | stated | Likes tea.\n")
    ws.write("state/preferences.md", "# Preferences\n- P-3.1 | current | 2025-10-03 | topic: drink | basis: O-3.1 | Likes tea.\n")
    budget = ReadBudget(10_000)
    result = Lookup(ws, budget).run({"ids": ["O-3.1", "P-3.1", "H-9.9"]})
    assert "E-3 | 2025-10-03" in result.content and "O-3.1 | stated" in result.content
    assert "P-3.1 | current" in result.content and "Not found: H-9.9" in result.content
    assert set(result.paths) == {"memory/episodes.md", "state/preferences.md"}
    assert budget.used == len(result.content)


def test_executor_records_access_and_rejects_bad_calls(ws, tmp_path):
    trace = TraceWriter(tmp_path / "trace.jsonl")
    read_only = ToolExecutor(workspace_tools(ws, writable=False), trace)
    assert {s.name for s in read_only.specs} == {"list_files", "read_file", "lookup"}
    assert not read_only.execute(ToolCall("1", "write_file", {"path": CONTEXT_FILE, "content": "x"})).ok
    assert not read_only.execute(ToolCall("2", "read_file", {"path": "../raw_sessions.jsonl"})).ok
    assert not read_only.execute(ToolCall("3", "lookup", {"ids": "P-1.1"})).ok  # must be a list
    assert read_only.execute(ToolCall("4", "read_file", {"path": "state/goals.md"})).ok
    assert trace.stats.tool_errors == 3 and trace.stats.file_reads == 1
    assert trace.stats.access[-1] == {"tool": "read_file", "kind": "read", "paths": ["state/goals.md"], "ok": True}


def test_finish_tool_takes_optional_note(ws, tmp_path):
    finish = Finish()
    executor = ToolExecutor([finish], TraceWriter(tmp_path / "t.jsonl"))
    assert executor.execute(ToolCall("1", "finish", {})).ok and finish.called


def test_tools_requiring_approval_are_denied_by_default(tmp_path):
    class Launch(Tool):
        requires_approval = True
        spec = ToolSpec("launch", "An external action.", {"type": "object", "properties": {}})

        def run(self, args):
            raise AssertionError("must not run without approval")

    executor = ToolExecutor([Launch()], TraceWriter(tmp_path / "t.jsonl"))
    assert executor.execute(ToolCall("1", "launch", {})) == ToolResult(False, "Not approved: No approver is configured for this tool.")


def test_items_get_isolated_workspaces(items, tmp_path):
    model = FakeModel()
    run(items, model, tmp_path / "run", run_config("agent"), progress=lambda *_: None)
    item_dirs = sorted((tmp_path / "run" / "items").iterdir())
    first = (item_dirs[0] / "workspace" / "memory" / "episodes.md").read_text()
    second = (item_dirs[1] / "workspace" / "memory" / "episodes.md").read_text()
    assert "Tokyo" in first and "Tokyo" not in second
    first_call_item2 = next(r for r in model.requests if "ferns" in r.blocks[0].text)
    assert "events seen" not in first_call_item2.blocks[1].text  # starts from the template


def test_raw_session_log_is_immutable_verifiable_and_resumable(evolved_item, tmp_path):
    path = tmp_path / "raw.jsonl"
    log = RawSessionLog(path)
    for s in evolved_item.episode.sessions:
        log.append(s)
    log.close()
    assert not os.stat(path).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    assert RawSessionLog.verify(path)
    with pytest.raises(FileExistsError):
        RawSessionLog(path)
    # Resuming keeps a verified prefix and continues the hash chain.
    resumed = RawSessionLog.resume(path, keep=2)
    resumed.append(evolved_item.episode.sessions[2])
    resumed.close()
    assert len(path.read_text().splitlines()) == 3 and RawSessionLog.verify(path)
    lines = path.read_text().splitlines()
    record = json.loads(lines[1])
    record["turns"][0]["content"] = "tampered"
    lines[1] = json.dumps(record)
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_text("\n".join(lines) + "\n")
    assert not RawSessionLog.verify(tampered)


def test_archive_moves_closed_entries_with_notes_and_refuses_live_ones(tmp_path):
    from context_harness.checks import check_retention
    from context_harness.entries import parse_workspace
    from context_harness.tools import Archive

    ws = Workspace.create(tmp_path / "ws", HORIZONBENCH)
    ws.write("state/goals.md", "# Goals\n"
             "- G-1.1 | achieved | 2025-10-01 | basis: O-1.1 | Finish the report.\n  - 2025-10-01 (S-1): drafted.\n"
             "- G-2.1 | active | 2025-10-02 | basis: O-2.1 | Learn Spanish.\n")
    before = parse_workspace(ws.snapshot(), HORIZONBENCH)
    archive = Archive(ws)
    with pytest.raises(WorkspaceError, match="G-2.1 is active"):
        archive.run({"ids": ["G-1.1", "G-2.1"]})
    assert "G-1.1" in ws.read("state/goals.md")  # nothing moved on refusal
    result = archive.run({"ids": ["G-1.1"]})
    assert result.ok and set(result.paths) == {"state/goals.md", "memory/archive.md"}
    assert ws.read("state/goals.md") == "# Goals\n- G-2.1 | active | 2025-10-02 | basis: O-2.1 | Learn Spanish.\n"
    assert "G-1.1 | achieved" in ws.read("memory/archive.md") and "(S-1): drafted." in ws.read("memory/archive.md")
    after = parse_workspace(ws.snapshot(), HORIZONBENCH)
    assert after.entries["G-1.1"].path == "memory/archive.md" and after.entries["G-1.1"].notes == ["2025-10-01 (S-1): drafted."]
    assert check_retention(before, after) == []  # moved, not deleted
    with pytest.raises(WorkspaceError, match="Not found"):
        archive.run({"ids": ["G-9.9"]})


def test_duplicate_ids_are_rejected_and_results_name_new_entries(tmp_path):
    from context_harness.tools import AppendFile, WriteFile

    ws = Workspace.create(tmp_path / "ws", HORIZONBENCH)
    block = "- E-7 | 2025-10-05 | source: S-7 | Chat.\n  - O-7.1 | stated | Likes tea."
    first = AppendFile(ws).run({"path": "memory/episodes.md", "text": block})
    assert "New entries: E-7, O-7.1." in first.content
    with pytest.raises(WorkspaceError, match="E-7, O-7.1 would appear more than once"):
        ws.append("memory/episodes.md", block)  # the duplicate-episode bug from the OpenAI run
    with pytest.raises(WorkspaceError, match="already exist in memory/episodes.md"):
        ws.write("state/profile.md", "# Profile\n- O-7.1 | stated | Likes tea.\n")  # wrong file, same ID
    ws.write("state/goals.md", "# Goals\n- G-7.1 | active | 2025-10-05 | basis: O-7.1 | Learn tea.\n")
    removed = WriteFile(ws).run({"path": "state/goals.md", "content": "# Goals\n"})
    assert "Warning: this removed G-7.1" in removed.content
    ws.edit("memory/episodes.md", "Likes tea.", "Likes green tea.")  # edits that keep IDs unique are fine
