"""Parsing and chronological ordering."""

from datetime import datetime

import fixture_data
import pytest

from context_harness.dataset import load_items, parse_item
from context_harness.hb_compat import QUESTION_STEM, parse_conversations


def test_parser_matches_upstream_format():
    segs = parse_conversations(fixture_data.evolved_row()["conversation"])
    assert len(segs) == 6
    assert segs[0]["scenario"] == ""  # evolution-event chats have no scenario
    # A multi-line assistant turn is reassembled from continuation lines.
    assert segs[2]["turns"][1]["content"].startswith("Of course. Here is a thorough plan:\n1. Morning hike")


def test_chronological_policy_sorts_stably_and_drops_future(evolved_item):
    sessions = evolved_item.episode.sessions
    markers = [next(m for m in ("SESSION_MARKER_0", "SESSION_MARKER_1", "SESSION_MARKER_2", "SESSION_MARKER_3")
                    if m in s.turns[0].content) for s in sessions]
    # Text order was [evo(12), future(28), s(2), s(2), s(15)]; replay is by date,
    # ties keep text order, and the post-probe session is gone.
    assert markers == ["SESSION_MARKER_0", "SESSION_MARKER_1", "SESSION_MARKER_2", "SESSION_MARKER_3"]
    dates = [datetime.fromisoformat(s.date) for s in sessions]
    assert dates == sorted(dates)
    assert max(dates) <= datetime.fromisoformat(evolved_item.episode.probe.date)
    assert [s.index for s in sessions] == [0, 1, 2, 3]
    assert [s.source_index for s in sessions] == [2, 3, 0, 4]

    report = evolved_item.ordering
    assert report.text_order_monotonic is False
    assert report.dropped_after_probe == ("2025-10-28T09:00:00.000001",)
    assert report.n_history_sessions_in_text == 5 and report.n_replayed == 4


def test_text_policy_preserves_benchmark_order():
    item = parse_item(fixture_data.evolved_row(), index=0, order="text")
    assert [s.source_index for s in item.episode.sessions] == [0, 1, 2, 3, 4]
    assert any("CANARY_FUTURE_SESSION" in s.turns[0].content for s in item.episode.sessions)
    assert item.ordering.dropped_after_probe == ()


def test_probe_and_question_are_split_from_history(evolved_item):
    ep = evolved_item.episode
    assert all("CANARY_PROBE_TURN" not in t.content for s in ep.sessions for t in s.turns)
    assert ep.probe.turns[-1].role == "user"
    assert ep.probe.turns[-1].content.startswith("CANARY_PROBE_TURN")
    assert ep.question.stem == QUESTION_STEM
    assert ep.question.letters == ("A", "B", "C", "D", "E")
    # Option values are evaluator-only.
    assert all("canary_value" not in o.text for o in ep.question.options)
    assert evolved_item.gold.option_values["B"] == "canary_value_short"


def test_item_key_is_opaque(evolved_item):
    key = evolved_item.episode.item_key
    assert "canary" not in key.lower() and "CANARY_ITEM_ID" not in key and len(key) == 12


def test_hidden_scenarios_and_max_sessions():
    item = parse_item(fixture_data.evolved_row(), index=0, show_scenarios=False, max_sessions=2)
    assert all(s.scenario == "" for s in item.episode.sessions) and item.episode.probe.scenario == ""
    assert [s.index for s in item.episode.sessions] == [0, 1]
    assert "SESSION_MARKER_2" in item.episode.sessions[0].turns[0].content  # most recent two kept
    assert item.ordering.truncated_to_last == 2


def test_committed_fixture_matches_builder(tmp_path):
    assert fixture_data.FIXTURE_PATH.read_text() == fixture_data.write_fixture(tmp_path / "f.jsonl").read_text()
    items = load_items(source=fixture_data.FIXTURE_PATH)
    assert [it.gold.has_evolved for it in items] == [True, False]


def test_real_sample_replay_is_chronological_with_no_future_sessions():
    """Uses the cached HF sample if present (downloaded by any earlier run)."""
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            "stellalisy/HorizonBench", "sample/test-00000-of-00001.parquet",
            repo_type="dataset", local_files_only=True,
        )
    except Exception:
        pytest.skip("HorizonBench sample not cached; run `context-harness inspect-data` once")
    items = load_items(source=path)
    assert len(items) == 10
    for it in items:
        dates = [datetime.fromisoformat(s.date) for s in it.episode.sessions]
        assert dates == sorted(dates)
        assert all(d <= datetime.fromisoformat(it.episode.probe.date) for d in dates)
        assert it.episode.probe.turns[-1].role == "user"
    dropped = {it.gold.user_id: len(it.ordering.dropped_after_probe) for it in items}
    assert dropped["sonnet-4.5/user_112"] == 13 and dropped["sonnet-4.5/user_68"] == 19
    assert sum(dropped.values()) == 32


def test_guarded_parser_keeps_date_and_scenario_lines_inside_turns():
    from context_harness.hb_compat import parse_sessions

    text = fixture_data.build_conversation([
        {"date": "2025-10-01T09:00:00.000001", "scenario": "Real scenario.",
         "turns": [("user", "Catalog this:\nDate: 1943-04-15\nDate: YYYY-MM-DD\nScenario: a plot twist"),
                   ("assistant", "Done.")]},
        {"date": "2025-10-02T09:00:00.000001", "turns": [("user", "Next?")]},
    ])
    upstream, guarded = parse_conversations(text), parse_sessions(text)
    assert len(upstream) == 4  # upstream invents sessions at both "Date:" lines
    assert upstream[2]["scenario"] == "a plot twist"  # and lets turn text overwrite a scenario
    assert len(guarded) == 2
    assert guarded[0]["scenario"] == "Real scenario."
    assert guarded[0]["turns"][0]["content"] == "Catalog this:\nDate: 1943-04-15\nDate: YYYY-MM-DD\nScenario: a plot twist"


def test_guarded_parser_matches_upstream_on_clean_text():
    from context_harness.hb_compat import parse_sessions

    for row in fixture_data.rows():
        assert parse_sessions(row["conversation"]) == parse_conversations(row["conversation"])
