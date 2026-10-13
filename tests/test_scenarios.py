"""The FoodBody scenarios load as items, stay in sync with their generators, and their
labels and checklists are coherent: something changes, something stays, the answer is
neither the shortest nor (for the long set) the longest option, and every scenario has a
reference checklist."""

import json
import runpy
from pathlib import Path

import pytest

from context_harness.dataset import load_items
from context_harness.vet import label_free_flags

SCENARIOS = Path(__file__).parent.parent / "scenarios"
SETS = [("foodbody", 5, 3), ("foodbody_long", 3, 30)]


@pytest.mark.parametrize("name,count,min_events", SETS)
def test_files_match_their_generator(name, count, min_events):
    built = runpy.run_path(str(SCENARIOS / f"{name}.py"))
    on_disk = [json.loads(line) for line in (SCENARIOS / f"{name}.jsonl").read_text().splitlines()]
    assert on_disk == built["SCENARIOS"], f"run `python scenarios/{name}.py` to rebuild"
    assert json.loads((SCENARIOS / f"{name}_checklists.json").read_text()) == built["CHECKLISTS"]


@pytest.mark.parametrize("name,count,min_events", SETS)
def test_scenarios_are_coherent_items_with_checklists(name, count, min_events):
    items = load_items(source=str(SCENARIOS / f"{name}.jsonl"))
    checklists = json.loads((SCENARIOS / f"{name}_checklists.json").read_text())
    assert len(items) == count and set(checklists) == {i.gold.item_id for i in items}
    for item in items:
        pe = item.gold.preference_evolution
        assert pe["changed_attributes"] and set(pe["current_attributes"]) - set(pe["changed_attributes"])
        flags = label_free_flags(item)
        assert not flags["correct_is_shortest"] and flags["sessions_before_change"] >= 3
        assert flags["sessions_after_change"] >= 3 and len(item.episode.sessions) >= min_events
        assert item.gold.distractor_letter and item.gold.distractor_letter != item.gold.correct_letter
        c = checklists[item.gold.item_id]
        assert c["must_contain"] and c["must_not_assert"] and c["reply_should"]
        if name == "foodbody_long":
            lengths = {o.letter: len(o.text) for o in item.episode.question.options}
            assert lengths[item.gold.correct_letter] < max(lengths.values())
