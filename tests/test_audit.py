"""The dataset audit measures what docs/DATASET_NOTES.md claims it does."""

import fixture_data

from context_harness.audit import audit_row, format_audit, summarize


def test_audit_row_detects_post_probe_sessions_and_front_loaded_life_events():
    r = audit_row(fixture_data.evolved_row(), 0)
    # Stored order: life-event chat, post-question chat, then three dated sessions.
    assert r["n_history"] == 5 and r["n_replayed"] == 4
    assert r["n_after_probe"] == 1 and r["n_scenarioless_after_probe"] == 1
    assert r["scenarioless_prefix"] == 2 and r["text_monotonic"] is False
    assert r["probe_ends_with_user"] and r["n_evolution_events"] == 1 and r["n_evolution_after_probe"] == 0


def test_audit_length_measures_match_option_texts():
    row = fixture_data.evolved_row()
    r = audit_row(row, 0)
    import json

    lengths = {o["letter"]: len(o["option"]) for o in json.loads(row["options"])}
    assert r["option_chars"] == lengths
    shortest = min(sorted(lengths), key=lengths.get)
    assert r["shortest_correct"] == (shortest == "B")
    assert r["correct_char_rank"] == 1 + sum(v < lengths["B"] for v in lengths.values())


def test_summary_and_report_include_parse_errors():
    records = [audit_row(fixture_data.evolved_row(), 0), audit_row(fixture_data.static_row(), 1),
               {"index": 2, "item_id": "x", "error": "ValueError: bad date"}]
    s = summarize(records)
    assert s["n_items"] == 3 and s["n_parse_errors"] == 1
    assert s["overall"]["n"] == 2 and s["overall"]["items_with_post_probe_sessions"] == 1
    assert "parse errors: 1" in format_audit(s)
