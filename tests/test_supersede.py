"""supersede revises a belief in one step: the new entry carries supersedes: and a
changed: note, the old one is closed with until: and superseded_by:, and nothing
else about the old entry changes."""

import pytest

from context_harness.checks import EventInfo, check_state
from context_harness.entries import parse_workspace
from context_harness.tools import Supersede
from context_harness.workspace import GENERAL, Workspace, WorkspaceError

OLD = "- P-3.1 | current | 2025-10-01 | topic: output-style | basis: O-3.1 | Dramatic, high-passion drafts."
NOTE = "  - kept for rallies"


@pytest.fixture
def ws(tmp_path):
    ws = Workspace.create(tmp_path / "ws", GENERAL)
    ws.write("state/preferences.md", f"# Preferences\n{OLD}\n{NOTE}\n")
    ws.write("state/transient.md", "# Transient state\n- T-4.1 | active | 2025-10-02 | review: 2025-10-09 | basis: O-4.1 | Vegetarian this month.\n")
    return ws


def test_supersede_writes_both_sides_of_the_link(ws):
    new = ("- P-20.1 | current | 2025-10-20 | topic: output-style | basis: O-20.1, O-22.1 | "
           "changed: drafts now diplomatic and authoritative, not dramatic | Wants drafts diplomatic and authoritative.")
    result = Supersede(ws).run({"id": "P-3.1", "new_entry": new})
    assert result.ok and "P-20.1" in result.content
    text = ws.read("state/preferences.md")
    old_line = next(line for line in text.splitlines() if line.startswith("- P-3.1"))
    assert old_line == ("- P-3.1 | superseded | 2025-10-01 | topic: output-style | basis: O-3.1 | until: 2025-10-20 | "
                        "superseded_by: P-20.1 | Dramatic, high-passion drafts.")  # date and statement untouched
    assert NOTE in text  # notes stay with their entry
    new_line = next(line for line in text.splitlines() if line.startswith("- P-20.1"))
    assert "supersedes: P-3.1" in new_line  # added for the agent
    parsed = parse_workspace({p: ws.read(p) for p in GENERAL.paths}, GENERAL)
    events = [EventInfo(i, f"S-{i}", "2025-10-20") for i in range(21)]
    assert not {p.code for p in check_state(parsed, events, 20)} & {
        "supersession_mismatch", "incomplete_supersession", "missing_change_note", "history_rewritten"}


@pytest.mark.parametrize("new,complaint", [
    ("- P-20.1 | current | 2025-10-20 | topic: output-style | basis: O-20.1 | Wants diplomatic drafts.", "changed:"),
    ("- P-20.1 | current | 2025-10-20 | topic: output-style | basis: O-20.1 | changed: x | " + "y" * 300, "characters"),
    ("- P-20.1 | superseded | 2025-10-20 | topic: output-style | basis: O-20.1 | changed: x | Diplomatic.", "live"),
    ("- P-20.1 | current | 2025-10-20 | basis O-20.1 | changed: x", "parse"),
])
def test_supersede_rejects_incomplete_revisions_without_touching_files(ws, new, complaint):
    before = ws.read("state/preferences.md")
    with pytest.raises(WorkspaceError, match=complaint):
        Supersede(ws).run({"id": "P-3.1", "new_entry": new})
    assert ws.read("state/preferences.md") == before


def test_supersede_across_files_and_refuses_closed_entries(ws):
    fact = "- F-9.1 | current | 2025-10-09 | topic: diet | basis: O-9.1 | Vegetarian."
    assert Supersede(ws).run({"id": "T-4.1", "new_entry": fact}).ok  # a transient that turned out to last
    assert "superseded_by: F-9.1" in ws.read("state/transient.md")
    assert "supersedes: T-4.1" in ws.read("state/profile.md")
    with pytest.raises(WorkspaceError, match="only live entries"):
        Supersede(ws).run({"id": "T-4.1", "new_entry": fact.replace("F-9.1", "F-9.2")})


def test_checks_flag_unexplained_and_overlong_values():
    snap = {p: "" for p in GENERAL.paths}
    snap["memory/episodes.md"] = "- E-0 | 2025-10-01 | source: S-0 | Chat.\n  - O-0.1 | stated | Likes long answers.\n"
    snap["state/preferences.md"] = (
        "- P-0.1 | superseded | 2025-10-01 | topic: detail-level | basis: O-0.1 | until: 2025-10-01 | superseded_by: P-0.2 | Long.\n"
        f"- P-0.2 | current | 2025-10-01 | topic: detail-level | basis: O-0.1 | supersedes: P-0.1 | {'Wants detail. ' * 20}\n")
    codes = {p.code for p in check_state(parse_workspace(snap, GENERAL), [EventInfo(0, "S-0", "2025-10-01")], 0)}
    assert {"missing_change_note", "long_value"} <= codes


@pytest.mark.parametrize("note,refused", [
    ("now also wants formal drafts", True),
    ("also wants citations", True),
    ("formal drafts now, no longer dramatic", False),
    ("adds carb counts per meal", False),
])
def test_supersede_refuses_purely_additive_change_notes(ws, note, refused):
    new = f"- P-20.1 | current | 2025-10-20 | topic: output-style | basis: O-20.1 | changed: {note} | Wants formal drafts."
    if refused:
        with pytest.raises(WorkspaceError, match="confirm it instead"):
            Supersede(ws).run({"id": "P-3.1", "new_entry": new})
    else:
        assert Supersede(ws).run({"id": "P-3.1", "new_entry": new}).ok
