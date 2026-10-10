"""Item vetting reads the person's own words around a labeled change, never runs during
a method, and calls an item suspect when the change doesn't show or the question's
end contradicts the label."""

import json

import fixture_data

from context_harness.cli import main
from context_harness.dataset import load_items
from context_harness.models.fake import ScriptedModel
from context_harness.vet import label_free_flags, render_vet_input, verdict, vet_item


def _evolved():
    return next(it for it in load_items(source=str(fixture_data.FIXTURE_PATH))
                if (it.gold.preference_evolution or {}).get("changed_attributes"))


def test_verdicts():
    a = lambda before, after: {"attribute": "x", "before": before, "after": after, "evidence": ""}  # noqa: E731
    assert verdict([a("original", "current")], "no") == ("usable", [])
    assert verdict([a("not_expressed", "current")], "unclear")[0] == "usable"
    v, reasons = verdict([a("original", "original")], "no")
    assert v == "suspect" and any("still shows the original" in r for r in reasons)
    assert verdict([a("original", "current")], "yes")[0] == "memory_only"  # the question's end contradicts the label
    assert verdict([a("current", "current")], "no")[0] == "suspect"  # no change: it was there all along


def test_input_splits_user_turns_at_the_change_and_shows_the_label():
    item = _evolved()
    text = render_vet_input(item)
    flags = label_free_flags(item)
    before = text.split("<before_change>")[1].split("</before_change>")[0]
    after = text.split("<after_change>")[1].split("</after_change>")[0]
    assert flags["sessions_after_change"] == after.count("[S-") and flags["sessions_before_change"] == before.count("[S-")
    assert all(d < flags["change_date"] for d in __import__("re").findall(r"\[S-\d+ (\S+)\]", before))
    assert "<labeled_correct_response" in text and "assistant:" not in before + after  # only the person's words


def test_vet_item_and_cli(tmp_path, monkeypatch):
    item = _evolved()
    changed = list(item.gold.preference_evolution["changed_attributes"])

    def script(req):
        assert req.phase == "vet"
        return "", [("submit_vetting", {
            "attributes": [{"attribute": k, "before": "original", "after": "current", "evidence": "S-3"} for k in changed],
            "final_turn_conflict": "no", "final_turn_explanation": "fits"})]

    r = vet_item(ScriptedModel(script), item)
    assert r["verdict"] == "usable" and [a["attribute"] for a in r["attributes"]] == changed
    monkeypatch.setattr("context_harness.cli.make_model", lambda settings: ScriptedModel(script))
    out = tmp_path / "vet.json"
    assert main(["vet-items", "--dataset", str(fixture_data.FIXTURE_PATH), "--model", "fake", "--out", str(out)]) == 0
    reports = json.loads(out.read_text())
    assert {r["verdict"] for r in reports} <= {"usable", "no_change"} and any(r["verdict"] == "usable" for r in reports)
