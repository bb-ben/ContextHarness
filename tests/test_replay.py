"""Answer-only replay restores a saved memory, reruns just the answer step, and
applies the chosen strategy."""

import json

import fixture_data

from context_harness.cli import main
from context_harness.models.fake import ScriptedModel
from context_harness.prompts import ANSWER_STRATEGIES
from context_harness.replay import find_snapshot, load_source_item, replay_answer
from conftest import run_config


def _finished_item(tmp_path):
    assert main(["run", "--model", "fake", "--dataset", str(fixture_data.FIXTURE_PATH), "--out", str(tmp_path),
                 "--run-name", "src"]) == 0
    return sorted((tmp_path / "src" / "items").iterdir())[0]


def test_replay_restores_memory_and_answers_once_per_repeat(tmp_path):
    source = _finished_item(tmp_path)
    item, snapshot = load_source_item(source), find_snapshot(source)
    seen = []

    def script(req):
        seen.append(req)
        assert req.phase == "answer"  # nothing is replayed except the answer step
        return "", [("submit_answer", {"letter": "B"})]

    cfg = run_config("agent", answer_strategy="dimension-first")
    result = replay_answer(item, snapshot, ScriptedModel(script), cfg, tmp_path / "replay" / "r1")
    assert result["predicted_letter"] == "B" and result["correct"] and result["strategy"] == "dimension-first"
    assert ANSWER_STRATEGIES["dimension-first"] in seen[0].blocks[1].text
    restored = (tmp_path / "replay" / "r1" / "workspace" / "context.md").read_text()
    assert restored == (snapshot / "context.md").read_text()  # same memory the run ended with
    assert (source / "result.json").exists() and not (source / "replay").exists()  # source untouched


def test_default_strategy_adds_no_instruction(tmp_path):
    source = _finished_item(tmp_path)
    seen = []
    replay_answer(load_source_item(source), find_snapshot(source),
                  ScriptedModel(lambda r: (seen.append(r) or "", [("submit_answer", {"letter": "A"})])),
                  run_config("agent", answer_strategy="default"), tmp_path / "replay" / "d1")
    assert "Work in this order" not in seen[0].blocks[1].text


def test_cli_replays_each_strategy_n_times(tmp_path):
    source = _finished_item(tmp_path)
    assert main(["replay-answer", str(source), "--model", "fake", "--strategy", "default",
                 "--strategy", "dimension-first", "--repeats", "2", "--out", str(tmp_path), "--run-name", "rep"]) == 0
    summary = json.loads((tmp_path / "rep" / "summary.json").read_text())
    assert set(summary["strategies"]) == {"default", "dimension-first"}
    assert all(len(s["answers"]) == 2 for s in summary["strategies"].values())
    assert sorted(p.name for p in (tmp_path / "rep").iterdir() if p.is_dir()) == [
        "default_1", "default_2", "dimension-first_1", "dimension-first_2"]
