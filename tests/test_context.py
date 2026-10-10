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
    assert report["violations_by_code"]["overdue_transient"] == 1  # one distinct violation...
    open_at = [r["event_id"] for r in report["per_event"] if r["violations"].get("overdue_transient")]
    assert open_at == ["S-2", "S-3"] and report["unresolved_at_end"] >= 1  # ...left open for two events


def test_past_due_goals_are_listed_after_their_due_day(evolved_item, tmp_path):
    def script(req):
        if req.phase == "answer":
            return "", [("submit_answer", {"letter": "A"})]
        if ev(req) == 0:
            return "", [("append_file", {"path": "state/goals.md",
                                         "text": "- G-0.1 | active | 2025-10-02 | due: 2025-10-02 | basis: O-0.1 | Give the talk."}),
                        ("finish", {})]
        return "", [("finish", {})]

    model = ScriptedModel(script)
    run([evolved_item], model, tmp_path / "run", run_config("agent"), progress=lambda *_: None)
    reqs = {(r.phase, ev(r), r.metadata.get("step")): r for r in model.requests}
    assert _due_block(reqs[("memory_update", 1, 0)]) == "(none)"  # 2025-10-02: the due day itself
    assert "G-0.1 | active" in _due_block(reqs[("memory_update", 2, 0)])  # 2025-10-12: passed


def test_hypotheses_supported_by_two_events_come_due(evolved_item, tmp_path):
    def script(req):
        if req.phase == "answer":
            return "", [("submit_answer", {"letter": "A"})]
        if ev(req) == 0:
            return "", [("append_file", {"path": "memory/episodes.md", "text":
                         "- E-0 | 2025-10-02 | source: S-0 | Chat.\n  - O-0.1 | behavior | Joked around."}),
                        ("append_file", {"path": "state/hypotheses.md",
                                         "text": "- H-0.1 | open | 2025-10-02 | topic: tone | for: O-0.1 | May like a playful tone."}),
                        ("finish", {})]
        if ev(req) == 1:
            return "", [("append_file", {"path": "memory/episodes.md", "text":
                         "- E-1 | 2025-10-02 | source: S-1 | Chat.\n  - O-1.1 | reaction | Laughed at a joke."}),
                        ("edit_file", {"path": "state/hypotheses.md", "old_text": "for: O-0.1", "new_text": "for: O-0.1, O-1.1"}),
                        ("finish", {})]
        return "", [("finish", {})]

    model = ScriptedModel(script)
    run([evolved_item], model, tmp_path / "run", run_config("agent"), progress=lambda *_: None)
    reqs = {(r.phase, ev(r), r.metadata.get("step")): r for r in model.requests}
    assert "H-0.1" not in _due_block(reqs[("memory_update", 1, 0)])  # one event of support so far
    assert "H-0.1 | open" in _due_block(reqs[("memory_update", 2, 0)])  # two events: decide it


def test_old_episodes_roll_over_into_the_archive_unchanged(evolved_item, tmp_path):
    import json
    import re

    from context_harness.entries import parse_workspace
    from context_harness.workspace import EPISODES_FILE, HORIZONBENCH

    def script(req):
        if req.phase == "answer":
            return "", [("submit_answer", {"letter": "A"})]
        k, date = re.search(r'<event id="S-(\d+)" type="[^"]*" date="([^"]+)"', req.blocks[0].text).groups()
        block = f"- E-{k} | {date} | source: S-{k} | " + "x" * 150 + f"\n  - O-{k}.1 | behavior | " + "y" * 60
        return "", [("append_file", {"path": EPISODES_FILE, "text": block}), ("finish", {})]

    layout = HORIZONBENCH.with_limit(EPISODES_FILE, 1000)
    results, _ = run([evolved_item], ScriptedModel(script), tmp_path / "run", run_config("agent", layout=layout),
                     progress=lambda *_: None)
    item = next((tmp_path / "run" / "items").iterdir())
    final = {p: (item / "snapshots" / "final" / p).read_text() for p in layout.paths}
    assert "- E-0 |" in final["memory/archive.md"] and "- E-1 |" in final["memory/archive.md"]
    assert "- E-0 |" not in final[EPISODES_FILE] and "- E-3 |" in final[EPISODES_FILE]
    parsed = parse_workspace(final, layout)
    assert parsed.entries["O-0.1"].path == "memory/archive.md" and parsed.entries["O-0.1"].parent == "E-0"
    report = json.loads((item / "memory_report.json").read_text())
    assert not {"entry_deleted", "orphan_observation", "duplicate_id", "history_rewritten"} & set(report["violations_by_code"])
    assert results[0]["stats"]["file_ops"] == 4  # the layer's rollover is not counted as the agent's edits
    events = [json.loads(l)["event"] for l in (item / "trace.jsonl").read_text().splitlines()]
    assert events.count("episode_rollover") == 1


def test_answer_budget_counts_preloaded_state(evolved_item, tmp_path):
    import re

    def script(req):
        if req.phase == "answer":
            return "", [("submit_answer", {"letter": "A"})]
        if ev(req) == 0:
            return "", [("write_file", {"path": "state/goals.md", "content": "# Goals\n" + "g" * 500 + "\n"}), ("finish", {})]
        return "", [("finish", {})]

    model = ScriptedModel(script)
    run([evolved_item], model, tmp_path / "run", run_config("agent"), progress=lambda *_: None)
    item = next((tmp_path / "run" / "items").iterdir())
    from context_harness.workspace import HORIZONBENCH

    always = sum(len((item / "snapshots" / "final" / p).read_text()) for p in ("context.md",) + HORIZONBENCH.preloaded)
    answer = next(r for r in model.requests if r.phase == "answer")
    shown = int(re.search(r"at most (\d+) more characters", answer.blocks[1].text).group(1))
    # Everything shown without a read is charged: the files themselves plus their markup and the
    # dimensions view, so the remaining budget is a little below 48,000 minus the raw file sizes.
    overhead = (48000 - always) - shown
    assert 0 < overhead < 6000 and "g" * 500 in answer.blocks[1].text
    assert "<preference_dimensions>" in answer.blocks[1].text
