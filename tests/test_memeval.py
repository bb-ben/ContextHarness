"""The evaluation framework grades memory against the person's own messages and a
reference checklist, grades replies against the checklist, and keeps the majority of
repeated judgments."""

import json
from pathlib import Path

import fixture_data

from context_harness.cli import main
from context_harness.memeval import CRITERIA, evaluate_memory, evaluate_reply, person_messages
from context_harness.models.fake import ScriptedModel
from context_harness.replay import load_source_item

CHECKLIST = {"must_contain": ["Is vegetarian", "Wants short answers"], "must_not_assert": ["Wants long answers"],
             "reply_should": ["Is short", "Is vegetarian"]}


def _item(tmp_path):
    assert main(["run", "--model", "fake", "--method", "profile", "--dataset", str(fixture_data.FIXTURE_PATH),
                 "--out", str(tmp_path), "--run-name", "src"]) == 0
    return sorted((tmp_path / "src" / "items").iterdir())[0]


def _memory_args(m1, n1, score):
    return {"must_contain": [{"id": "M1", "mark": m1}, {"id": "M2", "mark": "present"}],
            "must_not_assert": [{"id": "N1", "mark": n1}],
            "criteria": [{"criterion": k, "score": score} for k in CRITERIA]}


def test_memory_judge_sees_only_the_persons_words_and_keeps_the_majority(tmp_path):
    item = load_source_item(_item(tmp_path))
    answers = iter([_memory_args("present", "asserted", 2), _memory_args("missing", "asserted", 1),
                    _memory_args("present", "absent", 2)])
    seen = []
    result = evaluate_memory(ScriptedModel(lambda r: (seen.append(r) or "", [("submit_evaluation", next(answers))])),
                             item, "## Facts\n- Is vegetarian [O-1.1]", CHECKLIST)
    text = seen[0].blocks[0].text
    assert "<evidence>" in text and "assistant:" not in text.split("</evidence>")[0].lower()
    assert "M1. Is vegetarian" in text and "N1. Wants long answers" in text
    marks = {r["id"]: r["mark"] for r in result["checklist"]}
    assert marks == {"M1": "present", "M2": "present", "N1": "asserted"}  # 2 of 3 votes each
    assert result["contain_rate"] == 1.0 and result["stale_rate"] == 1.0
    assert result["criteria"]["currency"] == 2 and 0 < result["agreement"] < 1


def test_ties_go_to_the_worse_mark_and_replies_are_graded(tmp_path):
    item = load_source_item(_item(tmp_path))
    answers = iter([{"reply_should": [{"id": "R1", "mark": "follows"}, {"id": "R2", "mark": "violates"}], "asks_for_known": True},
                    {"reply_should": [{"id": "R1", "mark": "partial"}, {"id": "R2", "mark": "violates"}], "asks_for_known": False}])
    result = evaluate_reply(ScriptedModel(lambda r: ("", [("submit_evaluation", next(answers))])), item, "A long reply.",
                            CHECKLIST, repeats=2)
    marks = {r["id"]: r["mark"] for r in result["checklist"]}
    assert marks == {"R1": "partial", "R2": "violates"}  # R1 tied 1-1: the worse mark
    assert result["follow_rate"] == 0.0 and result["violate_rate"] == 0.5 and result["asks_for_known"] is False


def test_evaluate_cli_grades_memory_and_reply(tmp_path, monkeypatch):
    item_dir = _item(tmp_path)
    item = load_source_item(item_dir)
    checklists = tmp_path / "checklists.json"
    checklists.write_text(json.dumps({item.gold.item_id: CHECKLIST}))

    def script(req):
        if req.phase == "eval_memory":
            return "", [("submit_evaluation", _memory_args("present", "absent", 2))]
        if req.phase == "eval_reply":
            return "", [("submit_evaluation", {"reply_should": [{"id": "R1", "mark": "follows"},
                                                                {"id": "R2", "mark": "follows"}], "asks_for_known": False})]
        return "", [("respond", {"text": "Short vegetarian idea."})]

    monkeypatch.setattr("context_harness.cli.make_model", lambda settings: ScriptedModel(script))
    assert main(["evaluate", str(item_dir), "--checklists", str(checklists), "--model", "fake",
                 "--out", str(tmp_path), "--run-name", "ev"]) == 0
    row = json.loads((tmp_path / "ev" / "summary.json").read_text())["rows"][0]
    assert row["method"] == "profile" and row["memory"]["contain_rate"] == 1.0 and row["memory"]["stale_rate"] == 0.0
    assert row["reply"]["follow_rate"] == 1.0 and row["reply"]["text"] == "Short vegetarian idea."
    assert person_messages(item).startswith("[S-0 ")


def test_scenario_checklists_cover_every_scenario():
    root = Path(__file__).parent.parent / "scenarios"
    ids = {json.loads(line)["id"] for line in (root / "foodbody.jsonl").read_text().splitlines()}
    checklists = json.loads((root / "foodbody_checklists.json").read_text())
    assert set(checklists) == ids
    assert all(c["must_contain"] and c["must_not_assert"] and c["reply_should"] for c in checklists.values())
