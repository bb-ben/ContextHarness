"""The FoodBody scenarios load as items, stay in sync with their generator, and their
labels are coherent: something changes, something stays, and the answer isn't the
shortest option."""

import json
import runpy
from pathlib import Path

from context_harness.dataset import load_items
from context_harness.vet import label_free_flags

SCENARIOS = Path(__file__).parent.parent / "scenarios"


def test_jsonl_matches_its_generator():
    built = runpy.run_path(str(SCENARIOS / "foodbody.py"))["SCENARIOS"]
    on_disk = [json.loads(line) for line in (SCENARIOS / "foodbody.jsonl").read_text().splitlines()]
    assert on_disk == built, "run `python scenarios/foodbody.py` to rebuild foodbody.jsonl"


def test_scenarios_are_coherent_items():
    items = load_items(source=str(SCENARIOS / "foodbody.jsonl"))
    assert len(items) == 5
    for item in items:
        pe = item.gold.preference_evolution
        assert pe["changed_attributes"] and set(pe["current_attributes"]) - set(pe["changed_attributes"])
        flags = label_free_flags(item)
        assert not flags["correct_is_shortest"] and flags["sessions_before_change"] >= 3
        assert flags["sessions_after_change"] >= 3
        assert item.gold.distractor_letter and item.gold.distractor_letter != item.gold.correct_letter
