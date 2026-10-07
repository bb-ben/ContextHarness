"""No future sessions, question, or labels reach the agent during replay; no
labels reach it ever."""

import re

import pytest
from conftest import ev, run_config
from fixture_data import FORBIDDEN_ALWAYS, FORBIDDEN_DURING_REPLAY

from context_harness.models.fake import FakeModel
from context_harness.orchestrator import run

MARKER = re.compile(r"SESSION_MARKER_(\d)")


@pytest.mark.parametrize("method", ["agent", "summary", "recent"])
def test_no_leakage(method, items, tmp_path):
    model = FakeModel()
    run(items, model, tmp_path / "run", run_config(method), progress=lambda *_: None)
    assert model.requests, "expected model calls"

    for req in model.requests:
        text = req.prompt_text() + repr(req.tools)
        for canary in FORBIDDEN_ALWAYS:
            assert canary not in text, f"{canary} leaked into a {req.phase} request"
        if req.phase != "answer":
            for canary in FORBIDDEN_DURING_REPLAY:
                assert canary not in text, f"{canary} leaked during replay ({req.phase})"

    # Each replay request sees the current session, and only past or current
    # sessions (earlier ones may appear via memory the method wrote itself).
    replay = [r for r in model.requests if r.phase != "answer"]
    for req in (r for r in replay if "SESSION_MARKER" in r.blocks[0].text):
        current = ev(req)
        assert f"SESSION_MARKER_{current}" in req.blocks[0].text
        seen = {int(m) for m in MARKER.findall(req.prompt_text())}
        assert max(seen) <= current, f"future session visible at session {current}: {sorted(seen)}"


def test_long_term_files_are_not_auto_included(evolved_item, tmp_path):
    """Only context.md is injected; other files appear only after read_file."""
    from context_harness.models.fake import ScriptedModel

    def script(req):
        session, step = ev(req), req.metadata.get("step")
        if req.phase == "memory_update" and session == 0 and step == 0:
            return "", [("write_file", {"path": "state/profile.md", "content": "LONG_TERM_SECRET"})]
        if req.phase == "memory_update" and session == 1 and step == 0:
            return "", [("read_file", {"path": "state/profile.md"})]
        if req.phase == "answer":
            return "", [("submit_answer", {"letter": "A"})]
        return "done", []

    model = ScriptedModel(script)
    run([evolved_item], model, tmp_path / "run", run_config("agent"), progress=lambda *_: None)
    by_key = {(r.phase, ev(r), r.metadata.get("step")): r.prompt_text() for r in model.requests}
    assert "LONG_TERM_SECRET" not in by_key[("memory_update", 1, 0)]
    assert "LONG_TERM_SECRET" in by_key[("memory_update", 1, 1)]  # after the explicit read
    assert "LONG_TERM_SECRET" not in by_key[("answer", None, 0)]
