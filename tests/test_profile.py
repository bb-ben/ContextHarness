"""The profile method: one call per event records numbered evidence and rewrites one
profile; every version is kept; answers read only the profile."""

import json

import fixture_data

from conftest import ev, run_config
from context_harness.cli import main
from context_harness.models.fake import ScriptedModel
from context_harness.orchestrator import run


def _script(seen):
    def script(req):
        seen.append(req)
        if req.phase == "answer":
            return "", [("submit_answer", {"letter": "A"})]
        k = ev(req)
        if k == 1:
            return "no tool this time", []  # the method retries once, then keeps the old profile
        return "", [("record", {"episode": f"Chat {k}.", "observations": [
            {"kind": "stated", "text": f"Said something in event {k}."}, {"kind": "made-up", "text": "x"}],
            "profile": f"## How they want help\n- version {k} [O-{k}.1]"})]
    return script


def test_one_call_per_event_numbered_evidence_and_versions(evolved_item, tmp_path):
    seen = []
    results, _ = run([evolved_item], ScriptedModel(_script(seen)), tmp_path / "run", run_config("profile"),
                     progress=lambda *_: None)
    item_dir = next((tmp_path / "run" / "items").iterdir())
    updates = [r for r in seen if r.phase == "profile_update"]
    n_events = results[0]["stats"]["calls_by_phase"]["profile_update"] - 1  # event 1 was retried once
    assert len(updates) == n_events + 1 and "O-0.1, O-0.2, O-0.3" in updates[0].system
    episodes = (item_dir / "snapshots" / "final" / "memory" / "episodes.md").read_text()
    assert "- E-0 |" in episodes and "  - O-0.1 | stated | Said something in event 0." in episodes
    assert "  - O-0.2 | behavior | x" in episodes  # an unknown kind falls back to behavior
    assert "- E-1 |" in episodes and "(not recorded)" in episodes  # no tool: the event still gets an E line
    versions = [json.loads(line) for line in (item_dir / "profile_versions.jsonl").read_text().splitlines()]
    assert [v["event_id"] for v in versions] == [f"S-{k}" for k in range(n_events) if k != 1]
    final = (item_dir / "snapshots" / "final" / "profile.md").read_text()
    assert final == versions[-1]["profile"]
    answer = next(r for r in seen if r.phase == "answer")
    assert final in answer.blocks[0].text and "recent_episodes" not in answer.blocks[0].text


def test_profile_runs_end_to_end_with_the_fake_and_replies_read_it(tmp_path, monkeypatch):
    assert main(["run", "--model", "fake", "--method", "profile", "--dataset", str(fixture_data.FIXTURE_PATH),
                 "--out", str(tmp_path), "--run-name", "p"]) == 0
    from context_harness.respond import snapshot_kind

    item = sorted((tmp_path / "p" / "items").iterdir())[0]
    assert snapshot_kind(item / "snapshots" / "final") == "profile"
    assert "[O-" in (item / "snapshots" / "final" / "profile.md").read_text()


def test_over_long_rewrite_is_shortened_by_the_model_not_cut(evolved_item, tmp_path):
    def script(req):
        if req.phase == "answer":
            return "", [("submit_answer", {"letter": "A"})]
        if req.phase == "profile_shorten":
            text = req.blocks[0].text
            assert "the limit is 120. Rewrite it to about 90 characters" in text
            if ev(req) == 2:  # event 2: still too long after both attempts
                return "x" * (199 if "This is 200 characters" in text else 150), []
            return "## Right now\n- short", []
        return "", [("record", {"episode": "e", "observations": [], "profile": "## Facts\n" + "f" * 191})]

    run([evolved_item], ScriptedModel(script), tmp_path / "run", run_config("profile", summary_max_chars=120),
        progress=lambda *_: None)
    item_dir = next((tmp_path / "run" / "items").iterdir())
    updates = [json.loads(line) for line in (item_dir / "trace.jsonl").read_text().splitlines()
               if '"event": "profile_update"' in line]
    assert updates[0]["shortened"] == 1 and not updates[0]["truncated"] and updates[0]["chars"] == len("## Right now\n- short")
    assert updates[2]["shortened"] == 2 and updates[2]["truncated"] and updates[2]["chars"] == 120  # cut as a last resort


def test_facts_ledger_keeps_facts_records_changes_and_conflicts():
    from context_harness.methods.profile import FactsLedger

    ledger = FactsLedger()
    assert ledger.apply("birthdate", "1988-05-14", False, "E-2", "2022-01-24") == "added"
    assert ledger.apply("Residence", "Darwin, Australia", False, "E-3", "2022-02-01") == "added"
    assert ledger.apply("birthdate", "1988-05-14.", False, "E-9", "2022-04-01") == "repeated"
    assert ledger.apply("residence", "Melbourne, Australia", True, "E-5", "2022-02-25") == "changed"
    assert ledger.apply("birthdate", "1990-05-14", False, "E-27", "2022-08-03") == "conflict"
    assert ledger.apply("birthdate", "1990-05-14", False, "E-30", "2022-09-01") == "repeated"
    copied = "Melbourne, Australia [E-5, 2022-02-25]; earlier: Darwin, Australia (until 2022-02-25)"
    assert ledger.apply("residence", copied, False, "E-31", "2022-09-02") == "repeated"  # a copied ledger line
    text = ledger.render()
    assert "- birthdate: 1988-05-14 [E-2, 2022-01-24]; CONFLICT, also reported: 1990-05-14 [E-27, 2022-08-03]" in text
    assert "- Residence: Melbourne, Australia [E-5, 2022-02-25]; earlier: Darwin, Australia (until 2022-02-25)" in text
    restored = FactsLedger(list(ledger.entries.values()))
    assert restored.render() == text


def test_facts_survive_profile_rewrites_and_reach_the_answer(evolved_item, tmp_path):
    seen = []

    def script(req):
        seen.append(req)
        if req.phase == "answer":
            return "", [("submit_answer", {"letter": "A"})]
        k = ev(req)
        facts = [{"key": "allergy", "value": "peanuts", "changed": False}] if k == 0 else []
        return "", [("record", {"episode": "e", "observations": [], "facts": facts,
                                "profile": f"## How they want help\n- version {k}"})]

    run([evolved_item], ScriptedModel(script), tmp_path / "run", run_config("profile"), progress=lambda *_: None)
    updates = [r for r in seen if r.phase == "profile_update"]
    assert "<facts>\n(none yet)\n</facts>" in updates[0].blocks[0].text
    assert "- allergy: peanuts [E-0," in updates[-1].blocks[0].text  # shown in every later update
    answer = next(r for r in seen if r.phase == "answer")
    assert "- allergy: peanuts [E-0," in answer.prompt_text() and "version" in answer.prompt_text()
    item_dir = next((tmp_path / "run" / "items").iterdir())
    snap = item_dir / "snapshots" / "final"
    assert "- allergy: peanuts" in (snap / "facts.md").read_text()
    from context_harness.methods.profile import read_profile_memory
    assert read_profile_memory(snap).startswith("## Lasting facts") and "- allergy: peanuts" in read_profile_memory(snap)
    versions = [json.loads(line) for line in (item_dir / "profile_versions.jsonl").read_text().splitlines()]
    assert "allergy" in versions[-1]["facts"]
