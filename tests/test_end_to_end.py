"""Full fake-model runs through the CLI: outputs, determinism, resume."""

import json

import fixture_data
import pytest

from context_harness.cli import main


def _run(tmp_path, name, *extra):
    argv = ["run", "--model", "fake", "--dataset", str(fixture_data.FIXTURE_PATH),
            "--out", str(tmp_path), "--run-name", name, *extra]
    assert main(argv) == 0
    return tmp_path / name


@pytest.mark.parametrize("method", ["agent", "recent", "summary"])
def test_cli_run_writes_all_outputs(tmp_path, method):
    run_dir = _run(tmp_path, method, "--method", method)
    config = json.loads((run_dir / "run_config.json").read_text())
    assert config["method"] == method and config["model_settings"]["model"] == "fake"
    assert config["method_config"]["answer_budget_chars"] == 48000

    results = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines()]
    assert len(results) == 2
    for r in results:
        assert r["predicted_letter"] in "ABCDE" and r["predicted_letter"]
        assert r["error"] is None and isinstance(r["correct"], bool)
        assert {"model_calls", "usage", "file_ops", "file_reads"} <= set(r["stats"])
        assert r["latency_s"] >= 0

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["n_items"] == 2 and summary["evolved"]["n"] == 1 and summary["static"]["n"] == 1

    item_dir = sorted((run_dir / "items").iterdir())[0]
    events = [json.loads(l)["event"] for l in (item_dir / "trace.jsonl").read_text().splitlines()]
    assert events[0] == "item_start" and events[-1] == "evaluation"
    assert events.count("event_start") == 4  # future session dropped
    assert "question_start" in events and "answer" in events
    assert events.index("question_start") > max(i for i, e in enumerate(events) if e == "event_end")
    assert (item_dir / "snapshots" / "final").is_dir()
    if method == "agent":
        assert "file_diff" in events and (item_dir / "snapshots" / "initial" / "context.md").exists()
        result = json.loads((item_dir / "result.json").read_text())
        diag = result["evolution_diagnostics"]
        assert diag[0]["replayed_sessions"] == [2] and diag[0]["memory_writes"] >= 2
        # The fake follows agent.md exactly, so the checks must find nothing.
        report = json.loads((item_dir / "memory_report.json").read_text())
        assert report["violations_by_code"] == {} and report["events"] == 4
        assert report["final_metrics"]["provenance_share"] == 1.0
        assert {"memory/episodes.md", "state/transient.md", "context.md"} <= set(report["access_totals"])
        chain = result["evidence_chain"][0]
        assert chain["stage"] == "in_state" and chain["episodes"] == ["E-2"] and chain["observations"] == ["O-2.1"]
        assert "memory_check" in events and not (item_dir / "checkpoint.json").exists()
    else:  # no agent-managed files, so the file-op diagnostic does not apply
        assert json.loads((item_dir / "result.json").read_text())["evolution_diagnostics"] is None
    assert summary["shortest_option_accuracy"] is not None


def test_fake_runs_are_deterministic(tmp_path):
    a = _run(tmp_path, "a")
    b = _run(tmp_path, "b")
    strip = lambda d: [(r["item_key"], r["predicted_letter"], r["stats"]["usage"]) for r in
                       map(json.loads, (d / "results.jsonl").read_text().splitlines())]
    assert strip(a) == strip(b)
    for path in ("workspace/context.md", "workspace/memory/episodes.md"):
        item = sorted((a / "items").iterdir())[0].name
        assert (a / "items" / item / path).read_text() == (b / "items" / item / path).read_text()


def test_resume_reuses_finished_items_and_rejects_overwrite(tmp_path):
    run_dir = _run(tmp_path, "r")
    with pytest.raises(FileExistsError):
        _run(tmp_path, "r")
    before = (run_dir / "results.jsonl").read_text()
    _run(tmp_path, "r", "--resume")
    assert (run_dir / "results.jsonl").read_text() == before
    with pytest.raises(ValueError, match="different settings"):
        _run(tmp_path, "r", "--resume", "--context-chars", "4000")


def test_resume_allows_operational_changes_only(tmp_path):
    run_dir = tmp_path / "r"
    argv = ["run", "--model", "fake", "--dataset", str(fixture_data.FIXTURE_PATH), "--out", str(tmp_path), "--run-name", "r"]
    assert main(argv) == 0
    assert main(argv + ["--resume", "--timeout", "60", "--rpm", "30"]) == 0  # how requests are sent: fine
    with pytest.raises(ValueError, match="different settings"):
        main(argv + ["--resume", "--effort", "low"])  # what the model produces: refused
    events = [json.loads(l)["event"] for l in next((run_dir / "items").iterdir()).joinpath("trace.jsonl").read_text().splitlines()]
    assert events.count("model_request") == events.count("model_call")  # every request logged before it is sent
