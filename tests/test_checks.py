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
        "- P-0.1 | superseded | 2025-10-01 | topic: detail-level | basis: O-0.1 | until: 2025-10-09 | superseded_by: P-2.1 | Likes long, detailed answers.\n"
        "- P-2.1 | current | 2025-10-09 | topic: detail-level | basis: O-2.1 | supersedes: P-0.1 | changed: now wants short answers, not long ones | Wants short answers.\n"
    ),
    "state/goals.md": "# Goals\n- G-1.1 | active | 2025-10-05 | review: 2025-10-23 | basis: O-1.1 | Pass the exams.\n  - 2025-10-09 (S-2): studying daily.\n",
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
    from context_harness.checks import check_preferences

    assert check_preferences(parse_workspace(CLEAN, HORIZONBENCH), HORIZONBENCH) == []  # v3 checks too


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
    ("state/preferences.md", "| topic: detail-level | basis: O-2.1", "| basis: O-2.1", "missing_topic"),
    ("state/goals.md", "- G-1.1 | active | 2025-10-05 | review: 2025-10-23 | basis: O-1.1 | Pass the exams.", "- G-1.1 | active | Pass the exams.", "unparsed_entry"),
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
    ("state/goals.md", "- G-1.1 | active | 2025-10-05 | review: 2025-10-23 | basis: O-1.1 | Pass the exams.\n", "", "entry_deleted"),
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
    early["memory/episodes.md"] = "- E-0 | 2025-10-01 | source: S-0 | Planning chat.\n  - O-0.1 | behavior | Asked for a long, detailed answer.\n"
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


@pytest.mark.parametrize("where,flagged", [
    (None, True),                                         # recorded, applied nowhere
    ("state/goals.md", False),                            # cited in a goal progress note
    ("state/hypotheses.md", False),                       # cited as hypothesis evidence
])
def test_stated_observations_must_be_applied(where, flagged):
    snap = with_change("memory/episodes.md", "  - O-1.1 | event | Has exams all week.\n",
                       "  - O-1.1 | event | Has exams all week.\n  - O-1.2 | stated | Prefers weighing pros and cons.\n")
    if where == "state/goals.md":
        snap = {**snap, where: snap[where].replace("(S-2): studying daily.", "(S-2, O-1.2): studying daily.")}
    elif where == "state/hypotheses.md":
        snap = {**snap, where: snap[where].replace("for: O-1.1, O-2.1", "for: O-1.1, O-1.2, O-2.1")}
    assert (codes(snap)["unused_stated_observation"] == 1) is flagged


def test_goals_need_a_review_date_and_are_flagged_once_it_passes():
    no_date = with_change("state/goals.md", " | review: 2025-10-23", "")
    assert codes(no_date)["missing_end_date"] == 1
    due_soon = with_change("state/goals.md", "review: 2025-10-23", "due: 2025-10-20")
    assert codes(due_soon, current=3)["overdue_goal"] == 0  # live through its due day
    passed = with_change("state/goals.md", "review: 2025-10-23", "review: 2025-10-19")
    assert codes(passed, current=3)["overdue_goal"] == 1
    lapsed = with_change("state/goals.md", "| active | 2025-10-05 | review: 2025-10-23",
                         "| lapsed | 2025-10-05 | review: 2025-10-19 | ended: 2025-10-20")
    assert not {"overdue_goal", "missing_end_date", "bad_status"} & set(codes(lapsed, current=3))


def test_dimension_view_lists_open_hypotheses_under_their_dimension():
    from context_harness.dimensions import dimension_rows, render_dimensions

    hyps = "- H-5.1 | open | 2025-10-09 | topic: output-style | for: O-5.1 | May want formal, persuasive writing in general.\n"
    text = render_dimensions(dimension_rows(HORIZONBENCH, "", hyps))
    block = text.split("output-style:")[1].split("\n", 2)
    assert 'open hypothesis: H-5.1 "May want formal, persuasive writing in general."' in "\n".join(block[:3])


@pytest.mark.parametrize("for_,review,current,flagged", [
    ("O-1.1", None, 3, False),          # one event of support: not ripe
    ("O-0.1, O-1.1", None, 1, True),    # became ripe during S-1: decide it in S-1
    ("O-0.1, O-1.1", None, 2, True),    # ripe since S-1, still open after S-2
    ("O-0.1, O-1.1", "2025-10-15", 2, False),  # deferred with a later review date
    ("O-0.1, O-1.1", "2025-10-05", 2, True),   # deferral already passed
])
def test_supported_hypotheses_must_be_decided(for_, review, current, flagged):
    line = f"- H-1.2 | open | 2025-10-05 | topic: tone | for: {for_} | " + (f"review: {review} | " if review else "") + "May like a playful tone."
    snap = dict(CLEAN, **{"state/hypotheses.md": CLEAN["state/hypotheses.md"] + line + "\n"})
    assert (codes(snap, current=current)["undecided_hypothesis"] == 1) is flagged


def test_archiving_a_live_entry_is_flagged():
    goal = "- G-1.1 | active | 2025-10-05 | review: 2025-10-23 | basis: O-1.1 | Pass the exams.\n  - 2025-10-09 (S-2): studying daily.\n"
    snap = with_change("state/goals.md", goal, "")
    snap["memory/archive.md"] = "# Archive\n" + goal
    found = codes(snap)
    assert found["archived_live"] == 1 and "entry_deleted" not in found


def test_memory_diff_reports_additions_changes_moves_and_removals():
    from context_harness.checks import diff_memory

    before = parse_workspace(CLEAN, HORIZONBENCH)
    after_snap = dict(CLEAN)
    after_snap["state/transient.md"] = "# Transient state\n"
    after_snap["memory/archive.md"] = "# Archive\n" + CLEAN["state/transient.md"].splitlines()[1].replace(
        "| active |", "| expired |") + "\n"
    after_snap["state/goals.md"] = CLEAN["state/goals.md"].replace("| active |", "| achieved |")
    after_snap["state/profile.md"] = "# Profile\n- F-2.1 | current | 2025-10-09 | topic: diet | basis: O-2.1 | Vegetarian.\n"
    after_snap["context.md"] = CLEAN["context.md"].replace("- Stressed (T-1.1)\n", "- Vegetarian (F-2.1)\n")
    after_snap["memory/episodes.md"] = CLEAN["memory/episodes.md"].replace("  - O-0.1 | stated | Likes long, detailed answers.\n", "")
    d = diff_memory(before, parse_workspace(after_snap, HORIZONBENCH))
    assert [a["id"] for a in d["state"]["added"]] == ["F-2.1"]
    assert {c["id"]: c["status"] for c in d["state"]["changed"]} == {"G-1.1": ["active", "achieved"], "T-1.1": ["active", "expired"]}
    assert d["state"]["moved"] == [{"id": "T-1.1", "from": "state/transient.md", "to": "memory/archive.md"}]
    assert [r["id"] for r in d["episodic"]["removed"]] == ["O-0.1"]
    wc = d["working_context"]
    assert wc["added"] == ["Vegetarian (F-2.1)"] and wc["removed"] == ["Stressed (T-1.1)"]
    assert wc["cited_added"] == ["F-2.1"] and wc["cited_removed"] == ["T-1.1"]
    first = diff_memory(None, before)  # first event: everything is new
    assert len(first["episodic"]["added"]) == 6 and first["working_context"]["removed"] == []


# ---------------------------------------------------------------- v3: dimensions and evidence

def _prefs(*lines):
    snap = dict(CLEAN)
    snap["state/preferences.md"] = "# Preferences\n" + "".join(l + "\n" for l in lines)
    snap["context.md"] = CLEAN["context.md"].replace("- Wants short answers (P-2.1)\n", "")
    return snap


def pref_codes(*lines):
    from context_harness.checks import check_preferences

    return Counter(p.code for p in check_preferences(parse_workspace(_prefs(*lines), HORIZONBENCH), HORIZONBENCH))


def test_seed_dimensions_and_declared_custom_dimensions_pass():
    assert pref_codes(
        "- P-2.1 | current | 2025-10-09 | topic: detail-level | basis: O-2.1 | Wants short answers.",
        "- P-2.2 | current | 2025-10-09 | topic: gift-ideas | new_dimension: none covers gifts | basis: O-2.1 | Likes practical gifts.",
    ) == Counter()


def test_undeclared_and_near_duplicate_dimensions_are_flagged():
    found = pref_codes(
        "- P-2.1 | current | 2025-10-09 | topic: answer-length | basis: O-2.1 | Wants short answers.",
        "- P-2.2 | current | 2025-10-09 | topic: collaborative-style | new_dimension: x | basis: O-2.1 | Likes to co-work.",
    )
    assert found["undeclared_dimension"] == 1  # answer-length has no reason
    assert found["near_duplicate_dimension"] >= 1  # collaborative-style vs seed collaboration-style


@pytest.mark.parametrize("basis,resolved,flagged", [
    ("O-1.1", False, True),          # one event, kind `event`, no hypothesis: unsupported
    ("O-2.1", False, False),         # `stated`: allowed directly
    ("O-0.1, O-1.1", False, False),  # two events: allowed
    ("O-1.1", True, False),          # promoted from a confirmed hypothesis: allowed
])
def test_unsupported_promotion(basis, resolved, flagged):
    from context_harness.checks import check_preferences

    snap = _prefs(f"- P-2.1 | current | 2025-10-09 | topic: detail-level | basis: {basis} | Wants short answers.")
    if not resolved:
        snap["state/hypotheses.md"] = "# Hypotheses\n"
    codes_ = Counter(p.code for p in check_preferences(parse_workspace(snap, HORIZONBENCH), HORIZONBENCH))
    assert (codes_["unsupported_promotion"] == 1) is flagged


def test_violations_are_counted_once_and_resolution_is_recorded():
    monitor = MemoryMonitor(HORIZONBENCH)
    monitor.events = list(EVENTS[:2])  # S-0 and S-1 already happened; we observe S-2 repeatedly
    broken = with_change("context.md", "(S-2)", "(S-1)")  # stale context
    snap0 = {**broken, "context.md": broken["context.md"]}
    monitor.after_event(EVENTS[2], snap0, [])  # S-2: context says S-1
    monitor.after_event(EVENTS[2], snap0, [])  # still stale
    report = monitor.report()
    assert report["violations_by_code"]["context_stale"] == 1 and report["unresolved_at_end"] == 1
    assert [r["violations"].get("context_stale") for r in report["per_event"]] == [1, 1]
    assert [bool(r["new_violations"]) for r in report["per_event"]] == [True, False]
    monitor.after_event(EVENTS[2], CLEAN, [])  # fixed
    report = monitor.report()
    stale = next(v for v in report["violations"] if v["code"] == "context_stale")
    assert stale["resolved_at"] == "S-2" and report["unresolved_at_end"] == 0


def test_dimension_view_shows_values_custom_reasons_and_challenges():
    from context_harness.dimensions import dimension_rows, render_dimensions

    prefs = ("- P-2.1 | current | 2025-10-09 | topic: decision-style | basis: O-2.1 | Decides quickly.\n"
             "- P-2.2 | current | 2025-10-09 | topic: gift-ideas | new_dimension: none covers gifts | basis: O-2.1 | Practical gifts.\n"
             "- P-0.1 | superseded | 2025-10-01 | topic: tone | basis: O-0.1 | until: 2025-10-09 | superseded_by: P-2.3 | Formal.\n")
    hyps = "- H-2.1 | open | 2025-10-09 | confidence: low | for: O-2.1 | challenges: P-2.1 | May now prefer to deliberate.\n"
    text = render_dimensions(dimension_rows(HORIZONBENCH, prefs, hyps))
    assert 'decision-style: How they like to reach decisions' in text
    assert 'current: P-2.1 "Decides quickly."' in text and 'challenged by: H-2.1 (open)' in text
    assert "Dimensions you added:" in text and "gift-ideas: none covers gifts" in text
    assert "tone: How to talk with them" in text and "Formal" not in text  # superseded values are not current
    assert text.count("current: (none yet)") == len(HORIZONBENCH.dimensions) - 1


def test_preferences_that_may_not_last_carry_a_review_date():
    from context_harness.dimensions import dimension_rows, render_dimensions

    prefs = "- P-2.1 | current | 2025-10-09 | topic: detail-level | basis: O-2.1 | review: 2025-10-16 | Wants three-line answers.\n"
    assert 'current: P-2.1 "Wants three-line answers." (review 2025-10-16)' in render_dimensions(
        dimension_rows(HORIZONBENCH, prefs, ""))
    snap = with_change("state/preferences.md", "| supersedes: P-0.1 |", "| supersedes: P-0.1 | review: 2025-10-15 |")
    assert codes(snap, current=2)["overdue_preference_review"] == 0
    assert codes(snap, current=3)["overdue_preference_review"] == 1  # 2025-10-20: review reached, undecided


def test_preference_hypotheses_must_not_resolve_into_transient_entries():
    snap = with_change("state/hypotheses.md", "| resolved_into: P-2.1 |", "| topic: detail-level | resolved_into: T-1.1 |")
    assert codes(snap)["preference_into_transient"] == 1
    assert codes(CLEAN)["preference_into_transient"] == 0


def test_tagged_signals_surface_until_a_belief_takes_them_up():
    from context_harness.dimensions import dimension_rows, render_dimensions

    episodes = ("- E-5 | 2025-07-10 | source: S-5 | Dinner.\n"
                "  - O-5.1 | behavior | topic: constraints | Asked for a dinner they can make one-handed.\n"
                "- E-6 | 2025-07-16 | source: S-6 | Batch cooking.\n"
                "  - O-6.1 | behavior | topic: constraints | Asked for freezer meals, one pan if possible.\n")
    entries, problems = parse_file("memory/episodes.md", episodes, ("E", "O"))
    assert not problems and entries[1].fields == {"kind": "behavior", "topic": "constraints"}
    text = render_dimensions(dimension_rows(HORIZONBENCH, "", "", episodes))
    assert ('signals not yet in any belief (2 events): O-5.1 "Asked for a dinner they can make one-handed."; '
            'O-6.1 "Asked for freezer meals, one pan if possible."') in text

    snap = dict(CLEAN, **{"memory/episodes.md": CLEAN["memory/episodes.md"] + episodes.replace("E-5", "E-1").replace(
        "E-6", "E-2").replace("O-5.1", "O-1.2").replace("O-6.1", "O-2.2").replace("2025-07-10", "2025-10-05").replace(
        "2025-07-16", "2025-10-09")})
    assert codes(snap)["uncollected_signals"] == 1
    taken_up = dict(snap, **{"state/hypotheses.md": CLEAN["state/hypotheses.md"]
                             + "- H-2.2 | open | 2025-10-09 | topic: constraints | for: O-1.2, O-2.2 | May want one-pan, low-effort meals.\n"})
    assert codes(taken_up)["uncollected_signals"] == 0
    hyps = "- H-2.2 | open | 2025-10-09 | topic: constraints | for: O-5.1 | May want one-pan meals.\n"
    assert "signals not yet in any belief (1 event): O-6.1" in render_dimensions(dimension_rows(HORIZONBENCH, "", hyps, episodes))
