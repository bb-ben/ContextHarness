"""A quota pause, mid-item or mid-event, resumes to the same end state as an
uninterrupted run."""

import json

from conftest import ev, run_config

from context_harness.models.base import QuotaExhausted
from context_harness.models.fake import FakeModel, ScriptedModel
from context_harness.orchestrator import run
from context_harness.tracing import RawSessionLog


class QuotaAfter:
    """Wraps a model; raises QuotaExhausted once `limit` calls have succeeded."""

    def __init__(self, inner, limit):
        self.inner, self.limit, self.settings, self.calls = inner, limit, inner.settings, 0

    def complete(self, request):
        if self.calls >= self.limit:
            raise QuotaExhausted("daily quota exhausted (test)")
        self.calls += 1
        return self.inner.complete(request)


def _item_dir(run_dir):
    return next((run_dir / "items").iterdir())


def _final_state(run_dir):
    item = _item_dir(run_dir)
    result = json.loads((item / "result.json").read_text())
    files = {p.relative_to(item / "workspace").as_posix(): p.read_text()
             for p in (item / "workspace").rglob("*.md")}
    report = json.loads((item / "memory_report.json").read_text())
    return result, files, report


def _quiet(*_):
    pass


def test_pause_mid_item_then_resume_matches_uninterrupted(evolved_item, tmp_path):
    cfg = run_config("agent")
    run([evolved_item], FakeModel(), tmp_path / "straight", cfg, progress=_quiet)

    paused_dir = tmp_path / "paused"
    results, summary = run([evolved_item], QuotaAfter(FakeModel(), 3), paused_dir, cfg, progress=_quiet)
    item = _item_dir(paused_dir)
    assert results == [] and summary["paused_at"].endswith("after 3 of 4 events")
    assert not (item / "result.json").exists()
    assert json.loads((item / "checkpoint.json").read_text())["next_event"] == 3

    results, summary = run([evolved_item], FakeModel(), paused_dir, cfg, resume=True, progress=_quiet)
    assert len(results) == 1 and summary["paused_at"] is None
    straight, resumed = _final_state(tmp_path / "straight"), _final_state(paused_dir)
    assert resumed[1] == straight[1]  # identical workspace
    assert resumed[0]["predicted_letter"] == straight[0]["predicted_letter"]
    assert resumed[0]["stats"]["model_calls"] == straight[0]["stats"]["model_calls"]
    assert resumed[2]["violations_by_code"] == straight[2]["violations_by_code"] == {}
    assert len(resumed[2]["per_event"]) == 4
    raw = item / "raw_sessions.jsonl"
    assert RawSessionLog.verify(raw) and len(raw.read_text().splitlines()) == 4
    assert not (item / "checkpoint.json").exists()


def test_pause_during_answer_resumes_at_the_question(evolved_item, tmp_path):
    cfg = run_config("agent")
    run([evolved_item], QuotaAfter(FakeModel(), 5), tmp_path / "r", cfg, progress=_quiet)  # 4 events + 1 answer call
    item = _item_dir(tmp_path / "r")
    assert json.loads((item / "checkpoint.json").read_text())["next_event"] == 4
    model = FakeModel()
    results, _ = run([evolved_item], model, tmp_path / "r", cfg, resume=True, progress=_quiet)
    assert {r.phase for r in model.requests} == {"answer"}  # no event replayed twice
    assert results[0]["predicted_letter"] and results[0]["stats"]["model_calls"] == 6


def test_partial_edits_from_an_interrupted_event_are_discarded(evolved_item, tmp_path):
    """Event 1 writes a line, then the quota hits before it finishes: on resume the
    workspace is restored to the end of event 0 and event 1 is replayed cleanly."""
    state = {"first_run": True}

    def script(req):
        if state["first_run"] and ev(req) == 1 and req.metadata["step"] == 1:
            raise QuotaExhausted("daily quota exhausted (test)")
        if req.phase == "answer":
            return "", [("submit_answer", {"letter": "B"})]
        if req.metadata["step"] == 0:
            line = "PARTIAL" if state["first_run"] and ev(req) == 1 else f"- E-{ev(req)} | x"
            return "", [("append_file", {"path": "memory/episodes.md", "text": line})]
        return "done", []

    cfg = run_config("agent")
    run([evolved_item], ScriptedModel(script), tmp_path / "r", cfg, progress=_quiet)
    item = _item_dir(tmp_path / "r")
    assert "PARTIAL" in (item / "workspace" / "memory" / "episodes.md").read_text()
    state["first_run"] = False
    run([evolved_item], ScriptedModel(script), tmp_path / "r", cfg, resume=True, progress=_quiet)
    episodes = (item / "workspace" / "memory" / "episodes.md").read_text()
    assert "PARTIAL" not in episodes and episodes.count("- E-1 |") == 1
    events = [json.loads(l)["event"] for l in (item / "trace.jsonl").read_text().splitlines()]
    assert events.count("item_paused") == 1 and events.count("item_resumed") == 1


def test_baseline_state_survives_a_pause(evolved_item, tmp_path):
    cfg = run_config("summary")
    run([evolved_item], FakeModel(), tmp_path / "straight", cfg, progress=_quiet)
    run([evolved_item], QuotaAfter(FakeModel(), 2), tmp_path / "p", cfg, progress=_quiet)
    run([evolved_item], FakeModel(), tmp_path / "p", cfg, resume=True, progress=_quiet)
    a = (_item_dir(tmp_path / "straight") / "snapshots" / "final" / "summary.md").read_text()
    b = (_item_dir(tmp_path / "p") / "snapshots" / "final" / "summary.md").read_text()
    assert a == b and a.count("\n") == 3


def test_request_cap_pauses_the_run_and_resume_continues(evolved_item, tmp_path):
    from context_harness.models.base import RequestCap

    cfg = run_config("agent")
    capped = RequestCap(FakeModel(), 2)
    results, summary = run([evolved_item], capped, tmp_path / "r", cfg, progress=_quiet)
    assert results == [] and capped.count == 2 and summary["paused_at"].endswith("after 2 of 4 events")
    results, _ = run([evolved_item], RequestCap(FakeModel(), 100), tmp_path / "r", cfg, resume=True, progress=_quiet)
    assert len(results) == 1 and results[0]["predicted_letter"]
