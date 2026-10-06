"""Answer extraction, scoring, and aggregation."""

import pytest
from conftest import run_config

from context_harness.evaluation import aggregate, score
from context_harness.hb_compat import extract_letter
from context_harness.models.fake import ScriptedModel
from context_harness.orchestrator import run
from context_harness.types import AnswerResult


@pytest.mark.parametrize(
    "text,expected",
    [("B", "B"), ("  C. because", "C"), ("The answer is D", "D"), ("none", ""), ("E\n", "E")],
)
def test_extract_letter_matches_upstream(text, expected):
    assert extract_letter(text) == expected


def test_score(evolved_item):
    gold = evolved_item.gold  # correct B, distractor A
    assert score(AnswerResult("B", "tool"), gold) | {} == {
        "predicted_letter": "B",
        "answer_via": "tool",
        "correct": True,
        "invalid": False,
        "chose_distractor": False,
        "predicted_value": "canary_value_short",
    }
    wrong = score(AnswerResult("A", "tool"), gold)
    assert not wrong["correct"] and wrong["chose_distractor"] is True
    invalid = score(AnswerResult(None, "none"), gold)
    assert invalid["invalid"] and not invalid["correct"]


def test_static_items_have_no_distractor(static_item):
    assert score(AnswerResult("A", "tool"), static_item.gold)["chose_distractor"] is None


def _result(correct, evolved, distractor=None, chose=None, generator="g1", error=None):
    return {
        "correct": correct,
        "invalid": False,
        "chose_distractor": chose,
        "error": error,
        "latency_s": 1.0,
        "gold": {"has_evolved": evolved, "distractor_letter": distractor, "generator": generator},
        "stats": {
            "model_calls": 2, "file_ops": 1, "file_reads": 0, "tool_errors": 0, "refusals": 0,
            "usage": {"input_tokens": 10, "output_tokens": 2, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0},
        },
        "evolution_diagnostics": None,
    }


def test_aggregate():
    results = [
        _result(True, True, "A", False),
        _result(False, True, "A", True),
        _result(False, True, "A", False, generator="g2"),
        _result(True, False),
        _result(False, False),
        _result(False, False, error="boom"),
    ]
    agg = aggregate(results)
    assert agg["n_items"] == 6 and agg["n_errors"] == 1
    assert agg["overall"]["accuracy"] == pytest.approx(2 / 5)
    assert agg["evolved"]["accuracy"] == pytest.approx(1 / 3)
    assert agg["static"]["accuracy"] == pytest.approx(1 / 2)
    assert agg["evolved_minus_static"] == pytest.approx(1 / 3 - 1 / 2)
    assert agg["distractor"]["chose_distractor_rate"] == pytest.approx(1 / 3)
    assert agg["distractor"]["distractor_share_of_errors"] == pytest.approx(1 / 2)
    assert set(agg["by_generator"]) == {"g1", "g2"}
    assert agg["cost"]["tokens_total"]["input_tokens"] == 60


def test_text_answer_falls_back_to_letter_extraction(evolved_item, tmp_path):
    model = ScriptedModel(lambda req: ("B", []) if req.phase == "answer" else ("", []))
    results, _ = run([evolved_item], model, tmp_path / "run", run_config("agent"), progress=lambda *_: None)
    assert results[0]["predicted_letter"] == "B" and results[0]["answer_via"] == "text_fallback"
    assert results[0]["correct"]


def test_no_answer_is_invalid_after_step_limit(evolved_item, tmp_path):
    model = ScriptedModel(lambda req: ("let me think", []))
    results, _ = run([evolved_item], model, tmp_path / "run", run_config("agent", max_answer_steps=3),
                     progress=lambda *_: None)
    r = results[0]
    assert r["invalid"] and not r["correct"] and r["answer_details"]["stop"] == "max_steps"
    assert r["stats"]["calls_by_phase"]["answer"] == 3


def test_invalid_letter_is_rejected_by_tool_schema(evolved_item, tmp_path):
    def script(req):
        if req.phase != "answer":
            return "", []
        return ("", [("submit_answer", {"letter": "Z"})]) if req.metadata["step"] == 0 else ("", [("submit_answer", {"letter": "C"})])

    results, _ = run([evolved_item], ScriptedModel(script), tmp_path / "run", run_config("agent"),
                     progress=lambda *_: None)
    assert results[0]["predicted_letter"] == "C" and results[0]["stats"]["tool_errors"] == 1


def test_shortest_option_reference_is_label_free_and_scored(evolved_item):
    from context_harness.evaluation import reference_heuristics
    from context_harness.types import Option, Question

    q = Question("stem", (Option("A", "long long long"), Option("B", "mid mid"), Option("C", "xy"),
                          Option("D", "ab"), Option("E", "longest of them all")))
    ref = reference_heuristics(q, evolved_item.gold)  # gold is B
    assert ref == {"shortest_option": "C", "shortest_option_correct": False}  # ties break by letter
