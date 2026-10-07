"""context.md is a bounded working context whose edits reach the next model call."""

from conftest import ev, run_config

from context_harness.models.fake import ScriptedModel
from context_harness.orchestrator import run


def _working_context(req) -> str:
    dynamic = req.blocks[1].text
    return dynamic.split("<working_context", 1)[1].split("</working_context>", 1)[0]


def test_edits_replace_context_in_next_invocation(evolved_item, tmp_path):
    def script(req):
        session, step = ev(req), req.metadata.get("step")
        if req.phase == "memory_update" and session == 0 and step == 0:
            return "", [("write_file", {"path": "context.md", "content": "CTX_V1 user likes detail"})]
        if req.phase == "memory_update" and session == 0 and step == 1:
            return "", [("edit_file", {"path": "context.md", "old_text": "CTX_V1", "new_text": "CTX_V2"})]
        if req.phase == "answer":
            return "", [("submit_answer", {"letter": "B"})]
        return "nothing to change", []

    model = ScriptedModel(script)
    run([evolved_item], model, tmp_path / "run", run_config("agent"), progress=lambda *_: None)
    reqs = {(r.phase, ev(r), r.metadata.get("step")): r for r in model.requests}

    first = _working_context(reqs[("memory_update", 0, 0)])
    assert "no events yet" in first and "CTX_V" not in first  # the template
    after_write = _working_context(reqs[("memory_update", 0, 1)])
    assert "CTX_V1" in after_write and "no events yet" not in after_write  # replaced, not appended
    after_edit = _working_context(reqs[("memory_update", 0, 2)])
    assert "CTX_V2" in after_edit and "CTX_V1" not in after_edit
    # Carries into the next session and into the answer phase.
    assert "CTX_V2 user likes detail" in _working_context(reqs[("memory_update", 1, 0)])
    assert "CTX_V2 user likes detail" in _working_context(reqs[("answer", None, 0)])
    # The session transcript block is identical across steps (stable cache prefix).
    assert reqs[("memory_update", 0, 0)].blocks[0] == reqs[("memory_update", 0, 2)].blocks[0]
    assert reqs[("memory_update", 0, 0)].blocks[0].cache


def test_doing_nothing_is_allowed(evolved_item, tmp_path):
    model = ScriptedModel(lambda req: ("", [("submit_answer", {"letter": "A"})]) if req.phase == "answer" else ("no updates", []))
    results, _ = run([evolved_item], model, tmp_path / "run", run_config("agent"), progress=lambda *_: None)
    assert results[0]["stats"]["file_ops"] == 0
    assert results[0]["stats"]["calls_by_phase"]["memory_update"] == len(evolved_item.episode.sessions)


def test_context_budget_rejects_oversized_write(evolved_item, tmp_path):
    def script(req):
        session, step = ev(req), req.metadata.get("step")
        if req.phase == "memory_update" and session == 0 and step == 0:
            return "", [("write_file", {"path": "context.md", "content": "x" * 101})]
        if req.phase == "answer":
            return "", [("submit_answer", {"letter": "A"})]
        return "", []

    from context_harness.workspace import CONTEXT_FILE, HORIZONBENCH

    model = ScriptedModel(script)
    cfg = run_config("agent", layout=HORIZONBENCH.with_limit(CONTEXT_FILE, 100))
    results, _ = run([evolved_item], model, tmp_path / "run", cfg, progress=lambda *_: None)
    step1 = next(r for r in model.requests if ev(r) == 0 and r.metadata.get("step") == 1)
    assert "over its limit of 100" in step1.blocks[1].text  # the rejection is visible to the agent
    assert "xxxx" not in _working_context(step1)
    assert results[0]["stats"]["tool_errors"] == 1 and results[0]["stats"]["file_ops"] == 0


def _due_block(req) -> str:
    return req.blocks[1].text.split("<due>\n", 1)[1].split("\n</due>", 1)[0]


def test_contract_prompt_event_ids_finish_and_due_reminders(evolved_item, tmp_path):
    import json

    from context_harness.prompts import load_contract

    def script(req):
        if req.phase == "answer":
            return "", [("submit_answer", {"letter": "A"})]
        if ev(req) == 0:
            return "", [("append_file", {"path": "state/transient.md",
                                         "text": "- T-0.1 | active | 2025-10-02 | expires: 2025-10-05 | basis: O-0.1 | Busy week."}),
                        ("finish", {})]
        return "", [("finish", {})]  # never resolves the reminder

    model = ScriptedModel(script)
    results, _ = run([evolved_item], model, tmp_path / "run", run_config("agent"), progress=lambda *_: None)
    reqs = {(r.phase, ev(r), r.metadata.get("step")): r for r in model.requests}
    assert all(r.system == load_contract() for r in model.requests)  # agent.md is the system prompt
    assert reqs[("memory_update", 2, 0)].blocks[0].text.startswith('<event id="S-2" type="conversation" date="2025-10-12">')
    assert results[0]["stats"]["calls_by_phase"]["memory_update"] == 4  # finish ends each event in one call
    assert _due_block(reqs[("memory_update", 1, 0)]) == "(none)"  # 2025-10-02: not yet due
    assert "T-0.1 | active" in _due_block(reqs[("memory_update", 2, 0)])  # 2025-10-12: due
    report = json.loads(next((tmp_path / "run" / "items").iterdir()).joinpath("memory_report.json").read_text())
    assert report["violations_by_code"]["overdue_transient"] == 2  # flagged at S-2 and S-3
