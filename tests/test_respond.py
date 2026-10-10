"""Generated replies: the agent writes from its saved memory (and says which entries it
used), baselines write from plain text or nothing, and the judge's grades combine by
majority. Answers also record the entries they rest on."""

import json

import fixture_data

from conftest import ev, run_config
from context_harness.cli import main
from context_harness.judge import ground_truth
from context_harness.models.fake import ScriptedModel
from context_harness.orchestrator import run
from context_harness.replay import find_snapshot, load_source_item
from context_harness.respond import judge_response, respond_with_agent, respond_with_text


def _finished_item(tmp_path):
    assert main(["run", "--model", "fake", "--dataset", str(fixture_data.FIXTURE_PATH), "--out", str(tmp_path),
                 "--run-name", "src"]) == 0
    for item in sorted((tmp_path / "src" / "items").iterdir()):
        if load_source_item(item).gold.preference_evolution:
            return item
    raise AssertionError("fixture has no item with ground-truth attributes")


def _grades(truth, fit):
    return [("submit_grades", {"grades": [{"attribute": a["attribute"], "fit": fit, "reason": "r"}
                                          for a in truth["attributes"]]})]


def test_agent_writes_a_reply_from_memory_without_seeing_options(tmp_path):
    item_dir = _finished_item(tmp_path)
    item = load_source_item(item_dir)
    seen = []

    def script(req):
        seen.append(req)
        return "", [("respond", {"text": "Here is a plan.", "applied": ["T-0.1"]})]

    reply = respond_with_agent(item, find_snapshot(item_dir), ScriptedModel(script), run_config("agent"), tmp_path / "r")
    assert reply["text"] == "Here is a plan." and reply["applied"] == ["T-0.1"]
    text = "\n".join(b.text for b in seen[0].blocks)
    assert seen[0].phase == "respond" and "<current_conversation" in text and "<question>" not in text
    assert "<preference_dimensions>" in text and "respond with it" in text


def test_plain_text_replies_with_and_without_memory(tmp_path):
    item = load_source_item(_finished_item(tmp_path))
    seen = []
    model = ScriptedModel(lambda req: (seen.append(req) or "", [("respond", {"text": "ok"})]))
    respond_with_text(item, "Likes short answers.", "a rolling summary", model, tmp_path / "a")
    respond_with_text(item, None, "", model, tmp_path / "b")
    assert "<memory" in seen[0].blocks[0].text and "Likes short answers." in seen[0].blocks[0].text
    assert all("<memory" not in b.text for b in seen[1].blocks) and "know nothing" in seen[1].system


def test_response_judge_majority_and_shares(tmp_path):
    item = load_source_item(_finished_item(tmp_path))
    truth = ground_truth(item.gold)
    fits = iter(["current", "current", "original"])
    graded = judge_response(ScriptedModel(lambda req: ("", _grades(truth, next(fits)))), item.gold, item, "reply")
    assert all(r["fit"] == "current" and r["votes"] == ["current", "current", "original"] for r in graded["attributes"])
    assert graded["all_fit_current"] == 1.0


def test_respond_eval_cli(tmp_path, monkeypatch):
    item_dir = _finished_item(tmp_path)
    truth = ground_truth(load_source_item(item_dir).gold)

    def script(req):
        if req.phase == "judge_response":
            return "", _grades(truth, "current" if "planned" in req.blocks[0].text else "neutral")
        return "", [("respond", {"text": "A planned reply." if "<state>" in req.blocks[-1].text else "A reply.",
                                 "applied": []})]

    monkeypatch.setattr("context_harness.cli.make_model", lambda settings: ScriptedModel(script))
    assert main(["respond-eval", str(item_dir), "--model", "fake", "--condition", "memory", "--condition", "none",
                 "--out", str(tmp_path), "--run-name", "resp"]) == 0
    rows = json.loads((tmp_path / "resp" / "summary.json").read_text())["rows"]
    by = {r["condition"]: r for r in rows}
    assert by["memory"]["all_fit_current"] == 1.0 and by["none"]["all_fit_current"] == 0.0
    assert by["memory"]["utilization"] is not None and by["none"]["utilization"] is None


def test_answers_record_the_entries_they_rest_on(evolved_item, tmp_path):
    def script(req):
        if req.phase == "answer":
            return "", [("submit_answer", {"letter": "A", "applied": ["T-0.1", "P-9.9"]})]
        if ev(req) == 0:
            return "", [("append_file", {"path": "state/transient.md",
                                         "text": "- T-0.1 | active | 2025-10-02 | expires: 2026-01-05 | basis: O-0.1 | Busy."}),
                        ("finish", {})]
        return "", [("finish", {})]

    results, _ = run([evolved_item], ScriptedModel(script), tmp_path / "run", run_config("agent"), progress=lambda *_: None)
    used = results[0]["utilization"]
    assert used["applied"] == ["T-0.1", "P-9.9"] and used["live"] == ["T-0.1"] and used["unknown"] == ["P-9.9"]
