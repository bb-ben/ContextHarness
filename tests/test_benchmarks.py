"""Benchmark adapters on tiny made-up records in each benchmark's format: replay order,
questions answered from memory as it stood at the time, and output in the right shape."""

import json

from conftest import ev

from context_harness.benchmarks import dynamicmem as dm
from context_harness.benchmarks import memconflict as mc
from context_harness.cli import main
from context_harness.models.fake import ScriptedModel


def _person():
    def session(i, date, text, questions=()):
        return {"Session_ID": i, "Date": date, "Session_Questions": list(questions),
                "Session_Dialogue": {"dialogue_turn_2": [{"role": "user", "content": f"{text} (b)"}],
                                     "dialogue_turn_1": [{"role": "user", "content": text},
                                                         {"role": "assistant", "content": "Noted."}]}}

    q = lambda qid, kind: {"question_id": qid, "question": f"{kind}?", "answer": "ref", "conflict_type": kind}
    return {"ID": "p1", "Full_Session_Chain": [
        session(1, "2022-01-01", "I live in Ohio."),
        session(2, "2022-02-01", "I moved to Texas.", [q("Q1", "dynamic_conflict"), q("Q2", "dynamic_conflict")]),
        session(3, "2022-03-01", "Someone said I studied at X; I studied at Y.", [q("Q3", "static_conflict")]),
        session(4, "2022-04-01", "Juice only before training.", [q("Q4", "conditional_conflict")]),
    ]}


def _profile_script(seen):
    def script(req):
        seen.append(req)
        if req.phase == "profile_update":
            return "", [("record", {"episode": "e", "observations": [], "profile": f"profile after event {ev(req)}"})]
        if req.phase == "memconflict_answer":
            return "", [("answer", {"answer": "an answer"})]
        if req.phase == "memconflict_judge":
            return "", [("score", {"answer_accuracy": 1 if "event 1" in req.blocks[0].text else 0.5,
                                   "update_awareness": 1, "conflict_recognition": 0, "memory_support": 1,
                                   "reason": "ok"})]
        raise AssertionError(req.phase)
    return script


def test_memconflict_answers_each_question_from_memory_as_of_its_session(tmp_path):
    person, seen = _person(), []
    rows = mc.run_person(person, "profile", ScriptedModel(_profile_script(seen)), ScriptedModel(_profile_script(seen)),
                         tmp_path, per_person=6)
    first_event = next(r for r in seen if r.phase == "profile_update").blocks[0].text
    assert "I live in Ohio." in first_event and "Noted." in first_event  # turns in order, both speakers
    assert [r["question_id"] for r in rows] == ["Q1", "Q2", "Q3", "Q4"]
    answers = [r for r in seen if r.phase == "memconflict_answer"]
    assert "profile after event 1" in answers[0].blocks[0].text  # asked right after session index 1
    assert "profile after event 3" in answers[3].blocks[0].text
    assert "I moved to Texas" not in answers[0].blocks[0].text  # memory only, never the raw dialogue
    judged = [r for r in seen if r.phase == "memconflict_judge"]
    assert "Reference answer: ref" in judged[0].blocks[0].text and "Model answer: an answer" in judged[0].blocks[0].text
    s = mc.summarize(rows)
    assert s["n"] == 4 and s["dynamic_conflict"]["answer_accuracy"] == 1.0 and s["static_conflict"]["n"] == 1
    assert s["static_conflict"]["answer_accuracy"] == 0.5 and "conflict_recognition" in s["static_conflict"]
    assert len((tmp_path / "questions.jsonl").read_text().splitlines()) == 4


def test_memconflict_sample_is_stratified_by_conflict_type():
    person = _person()
    picked = [q["conflict_type"] for qs in mc.sample_questions(person, per_person=3).values() for q in qs]
    assert sorted(picked) == ["conditional_conflict", "dynamic_conflict", "static_conflict"]
    assert mc.sample_questions(person, 3) == mc.sample_questions(person, 3)  # seeded


def _logs():
    stamps = ["2024-01-01 08:00:00", "2024-01-02 09:00:00", "2024-01-03 10:00:00",  # one ISO week
              "2024-01-09 08:00:00", "2024-01-16 08:00:00"]
    return [{"app_log_id": f"log_{i}", "timestamp": t, "app_name": "Fitbit", "api_name": "Log",
             "request": {"activity": f"run {i}"}, "response": {"ok": True}} for i, t in enumerate(stamps)]


def _packs():
    def checkpoint(cid, index):
        return {"checkpoint_id": cid, "as_of": {"log_index": index},
                "state_completion_pack": {"keys": {"habits_state:run": {"question_text": "When do they run?"},
                                                   "user_attributes_state:grill": {"question_text": "Their grill?"}}},
                "rq3_apply_service_qa": {"keys": {"habits_state:run": {"items": [
                    {"qa_id": "q1", "service_family": "user_communication", "scenario": "Monday 7am.",
                     "task_instruction": "Draft a reminder."}]}}}}
    return {"user_id": "u1", "task_contract_version": "taskabc_v2", "research_frame_version": "rq_v2",
            "checkpoints": [checkpoint("cp1", 1), checkpoint("cp2", 3)]}


def test_dynamicmem_weekly_events_split_at_checkpoints():
    events = dm.weekly_events(_logs(), [1, 3])
    assert [last for _, last in events] == [1, 2, 3, 4]  # week 1 split after the checkpoint log
    assert "[log_0]" in events[0][0].turns[0].content and "[log_1]" in events[0][0].turns[0].content


def test_dynamicmem_predictions_in_benchmark_format(tmp_path):
    user = tmp_path / "u1"
    user.mkdir()
    (user / "app_log_large.json").write_text(json.dumps(_logs()))
    (user / "task_packs.json").write_text(json.dumps(_packs()))
    seen = []

    def script(req):
        seen.append(req)
        if req.phase == "summary_update":
            return f"summary after event {ev(req)}", []
        if req.phase == "dynamicmem_state":
            return "", [("submit", {"state": {"habits_state:run": {"habits_state:run": {"timing": "07:00"}},
                                              "user_attributes_state:grill": "Weber Genesis E-325s",
                                              "not_asked": {"x": 1}}})]
        if req.phase == "dynamicmem_service":
            return "", [("submit", {"answers": [{"n": 0, "answer": "Run at 7."}]})]
        raise AssertionError(req.phase)

    got = dm.run_user(user, "summary", ScriptedModel(script), tmp_path / "out", max_checkpoints=1)
    assert got["predictions"] == 1
    assert len([r for r in seen if r.phase == "summary_update"]) == 1  # stopped after the only checkpoint scored
    state = next(r for r in seen if r.phase == "dynamicmem_state").blocks[0].text
    assert "summary after event 0" in state and "[log_0]" not in state  # memory only
    pred = json.loads((tmp_path / "out" / "tce_results.json").read_text())
    assert pred["task_contract_version"] == "taskabc_v2"
    p = pred["predictions"][0]
    assert p["checkpoint_id"] == "cp1" and p["snapshot_state"] == {  # plain-text attributes are kept too
        "habits_state:run": {"timing": "07:00"}, "user_attributes_state:grill": "Weber Genesis E-325s"}
    assert p["rq3_apply_answers"]["habits_state:run"]["items"][0] == {
        "qa_id": "q1", "service_family": "user_communication", "answer": "Run at 7.", "evidence": []}


def test_cli_runs_both_benchmarks_offline(tmp_path):
    person = _person()
    data = tmp_path / "mc.jsonl"
    data.write_text(json.dumps(person) + "\n")
    assert main(["memconflict", "--data", str(data), "--model", "fake", "--out", str(tmp_path), "--run-name", "mc"]) == 0
    summary = json.loads((tmp_path / "mc" / "summary.json").read_text())
    assert summary["results"]["profile"]["n"] == 4 and summary["results"]["summary"]["n"] == 4

    user = tmp_path / "dm" / "u1"
    user.mkdir(parents=True)
    (user / "app_log_large.json").write_text(json.dumps(_logs()))
    (user / "task_packs.json").write_text(json.dumps(_packs()))
    assert main(["dynamicmem", "--data", str(tmp_path / "dm"), "--model", "fake", "--no-score",
                 "--out", str(tmp_path), "--run-name", "dmrun"]) == 0
    rows = json.loads((tmp_path / "dmrun" / "summary.json").read_text())["results"]
    assert [(r["method"], r["predictions"]) for r in rows] == [("profile", 2), ("summary", 2)]


def test_answer_model_and_judge_share_one_cost_cap(monkeypatch):
    import pytest

    from context_harness.models import pricing
    from context_harness.models.base import ModelRequest, QuotaExhausted, TextBlock

    monkeypatch.setitem(pricing.PRICES_PER_MTOK, "scripted", (1_000_000.0, 0.0, 0.0))  # $1 per input token
    request = ModelRequest(phase="x", system="", blocks=(TextBlock("abcd" * 3),))
    answer = pricing.CostCap(ScriptedModel(lambda r: ("", [])), max_usd=5.0)
    judge = pricing.CostCap(ScriptedModel(lambda r: ("", [])), max_usd=5.0, shared_with=answer)
    answer.complete(request)
    judge.complete(request)
    assert answer.spent == judge.spent > 0
    while answer.spent < 5.0:
        judge.complete(request)
    with pytest.raises(QuotaExhausted):
        answer.complete(request)


def test_memconflict_keeps_answered_rows_when_a_run_is_stopped(tmp_path):
    import pytest

    from context_harness.models.base import PauseRun

    seen = []
    base = _profile_script(seen)

    def script(req):
        if req.phase == "profile_update" and ev(req) == 3:
            raise PauseRun("cost cap")
        return base(req)

    with pytest.raises(PauseRun):
        mc.run_person(_person(), "profile", ScriptedModel(script), ScriptedModel(base), tmp_path, per_person=6)
    kept = [json.loads(line)["question_id"] for line in (tmp_path / "questions.jsonl").read_text().splitlines()]
    assert kept == ["Q1", "Q2", "Q3"]  # answered before the stop
