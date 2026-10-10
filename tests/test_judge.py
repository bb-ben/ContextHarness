"""The memory judge sees ground truth only after a run, grades beliefs rather than
raw episodes, and aggregates per-attribute grades."""

import json

import fixture_data

from context_harness.cli import main
from context_harness.judge import ground_truth, judge_memory, memory_text, render_judge_input, score_grades
from context_harness.models.fake import ScriptedModel
from context_harness.replay import find_snapshot, load_source_item


def _finished_item(tmp_path):
    assert main(["run", "--model", "fake", "--dataset", str(fixture_data.FIXTURE_PATH), "--out", str(tmp_path),
                 "--run-name", "src"]) == 0
    for item in sorted((tmp_path / "src" / "items").iterdir()):
        if load_source_item(item).gold.preference_evolution:
            return item
    raise AssertionError("fixture has no item with ground-truth attributes")


def test_memory_text_keeps_beliefs_and_drops_raw_episodes(tmp_path):
    snap = tmp_path / "snap"
    (snap / "memory").mkdir(parents=True)
    (snap / "state").mkdir()
    (snap / "context.md").write_text("# Working context\n- Wants short answers (P-2.1)\n")
    (snap / "state" / "preferences.md").write_text("# Preferences\n- P-2.1 | current | 2025-10-09 | topic: detail-level | basis: O-2.1 | Wants short answers.\n")
    (snap / "memory" / "episodes.md").write_text("- E-2 | 2025-10-09 | source: S-2 | Asked for brevity.\n  - O-2.1 | stated | Keep it short.\n")
    (snap / "memory" / "archive.md").write_text("# Archive\n- E-0 | 2025-10-01 | source: S-0 | Old chat.\n- P-0.1 | superseded | 2025-10-01 | topic: detail-level | basis: O-0.1 | until: 2025-10-09 | superseded_by: P-2.1 | Likes long answers.\n")
    text = memory_text(snap)
    assert "P-2.1" in text and "P-0.1" in text and "Wants short answers" in text
    assert "O-2.1 | stated" not in text and "E-0" not in text and "E-2" not in text


def test_scores_count_missing_grades_as_absent():
    truth = {"preference_domain": "d", "life_events": [], "attributes": [
        {"attribute": "speed", "current": "deliberate", "original": "moderate", "changed": True},
        {"attribute": "evidence", "current": "True", "original": "False", "changed": True},
        {"attribute": "consensus", "current": "medium", "changed": False},
    ]}
    grades = [
        {"attribute": "speed", "memory_value": "careful", "current_match": "matches",
         "original_still_current": False, "change_recorded": True, "evidence": "P-15.1"},
        {"attribute": "evidence", "memory_value": "evidence-backed", "current_match": "partial",
         "original_still_current": True, "change_recorded": False, "evidence": "P-15.1"},
    ]
    s = score_grades(truth, grades)
    assert s["changed_fidelity"] == 0.75 and s["current_fidelity"] == 0.5
    assert s["stale"] == 1 and s["changes_recorded"] == 1 and s["ungraded"] == ["consensus"]


def test_judge_sees_ground_truth_and_memory_and_submits_grades(tmp_path):
    item_dir = _finished_item(tmp_path)
    item = load_source_item(item_dir)
    truth = ground_truth(item.gold)
    seen = []

    def script(req):
        seen.append(req)
        return "", [("submit_grades", {"grades": [
            {"attribute": a["attribute"], "memory_value": "nothing", "current_match": "absent",
             "original_still_current": False, "change_recorded": False, "evidence": ""} for a in truth["attributes"]]})]

    result = judge_memory(ScriptedModel(script), item.gold, find_snapshot(item_dir))
    text = seen[0].blocks[0].text
    assert seen[0].phase == "judge" and "<ground_truth" in text and "<memory>" in text
    assert all(a["attribute"] in text for a in truth["attributes"])
    assert result["current_fidelity"] == 0.0 and not result["ungraded"]


def test_cli_writes_a_judgement_next_to_the_item(tmp_path, monkeypatch):
    item_dir = _finished_item(tmp_path)
    truth = ground_truth(load_source_item(item_dir).gold)

    def script(req):
        return "", [("submit_grades", {"grades": [
            {"attribute": a["attribute"], "memory_value": "x", "current_match": "matches",
             "original_still_current": False, "change_recorded": True, "evidence": ""} for a in truth["attributes"]]})]

    monkeypatch.setattr("context_harness.cli.make_model", lambda settings: ScriptedModel(script))
    assert main(["judge-memory", str(item_dir), "--model", "fake"]) == 0
    saved = json.loads((item_dir / "memory_judgement.json").read_text())
    assert saved["current_fidelity"] == 1.0 and saved["changes_recorded"] == saved["n_changed"]


def test_repeats_combine_by_majority_with_ties_going_low():
    from context_harness.judge import combine_repeats

    truth = {"preference_domain": "d", "life_events": [], "attributes": [
        {"attribute": "speed", "current": "deliberate", "original": "moderate", "changed": True},
        {"attribute": "consensus", "current": "medium", "changed": False}]}

    def run(speed, consensus, recorded):
        g = lambda a, m, r: {"attribute": a, "memory_value": m, "current_match": m,  # noqa: E731
                             "original_still_current": False, "change_recorded": r, "evidence": ""}
        s = score_grades(truth, [g("speed", speed, recorded), g("consensus", consensus, False)])
        return {"item_id": "i", "judge_model": "m", "memory_chars": 1, "estimated_cost_usd": 0.01, **s}

    j = combine_repeats([run("matches", "partial", True), run("matches", "absent", True), run("partial", "matches", False)])
    by = {a["attribute"]: a for a in j["attributes"]}
    assert by["speed"]["current_match"] == "matches" and by["speed"]["change_recorded"] is True
    assert by["consensus"]["current_match"] == "absent"  # three-way tie: the lowest score wins
    assert j["agreement"] == round((2 / 3 + 1 / 3) / 2, 3) and j["current_fidelity_range"] == [0.5, 0.75]
