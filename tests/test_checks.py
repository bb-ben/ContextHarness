"""Entry parsing and memory checks: each contract rule is shown to fire on a
workspace that breaks it, and to stay quiet on one that follows it."""

from collections import Counter

import pytest

from context_harness.checks import EventInfo, MemoryMonitor, check_retention, check_state
from context_harness.entries import parse_file, parse_workspace
from context_harness.workspace import HORIZONBENCH

EVENTS = [EventInfo(i, f"S-{i}", d) for i, d in enumerate(["2025-10-01", "2025-10-05", "2025-10-09", "2025-10-20"])]

CLEAN = {
    "context.md": "# Working context · as of 2025-10-09 (S-2)\n\n## Now\n- Wants short answers (P-2.1)\n\n## Watch\n- Stressed (T-1.1)\n",
    "memory/episodes.md": (
        "# Episodes\n"
        "- E-0 | 2025-10-01 | source: S-0 | Planning chat.\n"
        "  - O-0.1 | stated | Likes long, detailed answers.\n"
        "- E-1 | 2025-10-05 | source: S-1 | Exam week chat.\n"
        "  - O-1.1 | event | Has exams all week.\n"
        "- E-2 | 2025-10-09 | source: S-2 | Asked for brevity.\n"
        "  - O-2.1 | stated | From now on please keep answers short.\n"
    ),
    "state/profile.md": "# Profile\n",
    "state/preferences.md": (
        "# Preferences\n"
        "- P-0.1 | superseded | 2025-10-01 | topic: answer-length | basis: O-0.1 | until: 2025-10-09 | superseded_by: P-2.1 | Likes long, detailed answers.\n"
        "- P-2.1 | current | 2025-10-09 | topic: answer-length | basis: O-2.1 | supersedes: P-0.1 | Wants short answers.\n"
    ),
    "state/goals.md": "# Goals\n- G-1.1 | active | 2025-10-05 | basis: O-1.1 | Pass the exams.\n  - 2025-10-09 (S-2): studying daily.\n",
    "state/transient.md": "# Transient state\n- T-1.1 | active | 2025-10-05 | expires: 2025-10-12 | basis: O-1.1 | Stressed by exams.\n",
    "state/hypotheses.md": "# Hypotheses\n- H-1.1 | confirmed | 2025-10-05 | for: O-1.1, O-2.1 | resolved_into: P-2.1 | Exams may make them want brevity.\n",
}


def codes(snapshot, current=2, previous=None):
    parsed = parse_workspace(snapshot, HORIZONBENCH)
    problems = check_state(parsed, EVENTS, current) + check_retention(previous, parsed)
    return Counter(p.code for p in problems)


def with_change(path, old, new):
    snap = dict(CLEAN)
    assert old in snap[path]
    snap[path] = snap[path].replace(old, new)
    return snap


def test_clean_workspace_has_no_violations():
    assert codes(CLEAN) == Counter()


def test_parser_reads_entries_observations_and_progress_notes():
    entries, problems = parse_file("state/goals.md", CLEAN["state/goals.md"], ("G",))
    assert not problems and entries[0].notes == ["2025-10-09 (S-2): studying daily."]
    parsed = parse_workspace(CLEAN, HORIZONBENCH)
    assert parsed.entries["O-2.1"].parent == "E-2" and parsed.entries["O-2.1"].fields["kind"] == "stated"
    assert parsed.entries["P-2.1"].refs() == {"basis": ["O-2.1"], "supersedes": ["P-0.1"]}
    assert parsed.context.as_of == "S-2" and parsed.context.bullets[0][2] == ["P-2.1"]


@pytest.mark.parametrize("path,old,new,code", [
    # provenance
    ("state/goals.md", "G-1.1 | active | 2025-10-05", "G-1.1 | active | 2025-10-06", "date_mismatch"),
    ("state/goals.md", "G-1.1 | active | 2025-10-05", "G-1.1 | active | Oct 5", "bad_date"),
    ("state/goals.md", "- G-1.1 |", "- G-3.1 |", "future_id"),
    ("memory/episodes.md", "source: S-1 |", "source: S-7 |", "bad_source"),
    ("memory/episodes.md", "O-1.1 | event |", "O-1.1 | hunch |", "bad_kind"),
    ("memory/episodes.md", "  - O-0.1", "- O-0.1", "orphan_observation"),
    ("state/goals.md", " | basis: O-1.1 | Pass", " | Pass", "missing_basis"),
    ("state/goals.md", "basis: O-1.1 | Pass", "basis: O-9.9 | Pass", "dangling_ref"),
    ("state/goals.md", "basis: O-1.1 | Pass", "basis: H-1.1 | Pass", "basis_not_observation"),
    ("state/transient.md", " | expires: 2025-10-12", "", "missing_end_date"),
    ("state/preferences.md", "| topic: answer-length | basis: O-2.1", "| basis: O-2.1", "missing_topic"),
    ("state/goals.md", "- G-1.1 | active | 2025-10-05 | basis: O-1.1 | Pass the exams.", "- G-1.1 | active | Pass the exams.", "unparsed_entry"),
    ("state/goals.md", "# Goals\n", "# Goals\n- pass exams eventually\n", "unparsed_line"),
    ("state/goals.md", "- G-1.1 |", "- P-1.1 |", "wrong_file"),
    ("state/goals.md", "| active |", "| ongoing |", "bad_status"),
    # consistency
    ("state/preferences.md", "| superseded | 2025-10-01", "| current | 2025-10-01", "topic_conflict"),
    ("state/preferences.md", " | until: 2025-10-09 | superseded_by: P-2.1", "", "incomplete_supersession"),
    ("state/preferences.md", "| supersedes: P-0.1 |", "|", "supersession_mismatch"),
    ("state/hypotheses.md", " | resolved_into: P-2.1", "", "unresolved_confirmation"),
    ("context.md", "(S-2)", "(S-1)", "context_stale"),
    ("context.md", "- Stressed (T-1.1)", "- Stressed", "context_uncited"),
    ("context.md", "(P-2.1)", "(P-0.1)", "context_inactive"),
    ("context.md", "(P-2.1)", "(P-4.4)", "context_dangling"),
    ("memory/episodes.md", "- E-2 | 2025-10-09 | source: S-2 | Asked for brevity.\n", "", "missing_episode"),
])
def test_each_rule_fires(path, old, new, code):
    found = codes(with_change(path, old, new))
    assert found[code] >= 1, f"expected {code}, got {dict(found)}"


def test_overdue_transient_is_flagged_once_its_date_passes():
    snap = with_change("context.md", "(S-2)", "(S-3)")
    snap["memory/episodes.md"] += "- E-3 | 2025-10-20 | source: S-3 | Later chat.\n"
    assert codes(snap, current=3)["overdue_transient"] == 1
    resolved = dict(snap)
    resolved["state/transient.md"] = resolved["state/transient.md"].replace("| active |", "| expired |").replace(
        "| basis:", "| ended: 2025-10-20 | basis:")
    resolved["context.md"] = resolved["context.md"].replace("- Stressed (T-1.1)\n", "")
    assert codes(resolved, current=3) == Counter()


@pytest.mark.parametrize("path,old,new,code", [
    ("state/goals.md", "- G-1.1 | active | 2025-10-05 | basis: O-1.1 | Pass the exams.\n", "", "entry_deleted"),
    ("memory/episodes.md", "Has exams all week.", "Has one exam.", "history_rewritten"),
    ("state/goals.md", "basis: O-1.1 | Pass", "basis: O-2.1 | Pass", "history_rewritten"),
    ("state/preferences.md", "P-0.1 | superseded", "P-0.1 | current", "status_reverted"),
])
def test_retention_rules_fire_against_the_previous_snapshot(path, old, new, code):
    previous = parse_workspace(CLEAN, HORIZONBENCH)
    assert codes(with_change(path, old, new), previous=previous)[code] >= 1


def test_allowed_changes_are_not_retention_violations():
    previous = parse_workspace(CLEAN, HORIZONBENCH)
    grown = with_change("state/goals.md", "basis: O-1.1 | Pass", "basis: O-1.1, O-2.1 | Pass")  # evidence added
    grown["state/goals.md"] = grown["state/goals.md"].replace("| active |", "| achieved |")  # status change
    assert not {"history_rewritten", "entry_deleted"} & set(codes(grown, previous=previous))


def test_monitor_tracks_access_lifecycle_and_report():
    monitor = MemoryMonitor(HORIZONBENCH)
    early = {k: (v if k in ("state/profile.md",) else "") for k, v in CLEAN.items()}
    early["memory/episodes.md"] = "- E-0 | 2025-10-01 | source: S-0 | Planning chat.\n  - O-0.1 | stated | Likes long, detailed answers.\n"
    early["context.md"] = "# Working context · as of 2025-10-01 (S-0)\n"
    access = [{"tool": "append_file", "kind": "write", "paths": ["memory/episodes.md"], "ok": True},
              {"tool": "read_file", "kind": "read", "paths": ["state/goals.md"], "ok": False}]
    first = monitor.after_event(EVENTS[0], early, access)
    assert first["access"] == {"memory/episodes.md": {"writes": 1}, "state/goals.md": {"errors": 1}}
    assert first["violations"] == {}
    monitor.after_event(EVENTS[1], {**early, "context.md": "# Working context · as of 2025-10-05 (S-1)\n",
                                    "memory/episodes.md": ""}, [])
    report = monitor.report()
    assert report["violations_by_code"]["entry_deleted"] == 2 and report["violations_by_code"]["missing_episode"] == 1
    assert report["lifecycle"]["O-0.1"]["deleted"] == "S-1"
    assert report["access_totals"]["memory/episodes.md"] == {"writes": 1}


def test_archiving_a_live_entry_is_flagged():
    goal = "- G-1.1 | active | 2025-10-05 | basis: O-1.1 | Pass the exams.\n  - 2025-10-09 (S-2): studying daily.\n"
    snap = with_change("state/goals.md", goal, "")
    snap["memory/archive.md"] = "# Archive\n" + goal
    found = codes(snap)
    assert found["archived_live"] == 1 and "entry_deleted" not in found
