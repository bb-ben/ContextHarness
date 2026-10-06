"""context.md is a bounded working context whose edits reach the next model call."""

from conftest import run_config

from context_harness.models.fake import ScriptedModel
from context_harness.orchestrator import run


def _working_context(req) -> str:
    dynamic = req.blocks[1].text
    return dynamic.split("<working_context", 1)[1].split("</working_context>", 1)[0]


def test_edits_replace_context_in_next_invocation(evolved_item, tmp_path):
    def script(req):
        session, step = req.metadata.get("session"), req.metadata.get("step")
        if req.phase == "memory_update" and session == 0 and step == 0:
            return "", [("write_file", {"path": "context.md", "content": "CTX_V1 user likes detail"})]
        if req.phase == "memory_update" and session == 0 and step == 1:
            return "", [("edit_file", {"path": "context.md", "old_text": "CTX_V1", "new_text": "CTX_V2"})]
        if req.phase == "answer":
            return "", [("submit_answer", {"letter": "B"})]
        return "nothing to change", []

    model = ScriptedModel(script)
    run([evolved_item], model, tmp_path / "run", run_config("agent"), progress=lambda *_: None)
    reqs = {(r.phase, r.metadata.get("session"), r.metadata.get("step")): r for r in model.requests}

    first = _working_context(reqs[("memory_update", 0, 0)])
    assert "Purpose:" in first and "CTX_V" not in first  # the template
    after_write = _working_context(reqs[("memory_update", 0, 1)])
    assert "CTX_V1" in after_write and "Purpose:" not in after_write  # replaced, not appended
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
        session, step = req.metadata.get("session"), req.metadata.get("step")
        if req.phase == "memory_update" and session == 0 and step == 0:
            return "", [("write_file", {"path": "context.md", "content": "x" * 101})]
        if req.phase == "answer":
            return "", [("submit_answer", {"letter": "A"})]
        return "", []

    from context_harness.workspace import WorkspaceBudgets

    model = ScriptedModel(script)
    cfg = run_config("agent", budgets=WorkspaceBudgets(context_max_chars=100, file_max_chars=2000))
    results, _ = run([evolved_item], model, tmp_path / "run", cfg, progress=lambda *_: None)
    step1 = next(r for r in model.requests if r.metadata.get("session") == 0 and r.metadata.get("step") == 1)
    assert "over its limit of 100" in step1.blocks[1].text  # the rejection is visible to the agent
    assert "xxxx" not in _working_context(step1)
    assert results[0]["stats"]["tool_errors"] == 1 and results[0]["stats"]["file_ops"] == 0
