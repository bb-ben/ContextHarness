"""Evaluator side: scoring and aggregation. Only this module and the
orchestrator's post-answer section ever read GoldLabel."""

from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean
from typing import Any

from .types import AnswerResult, GoldLabel, Question

CHANCE = 0.2


def score(answer: AnswerResult, gold: GoldLabel) -> dict[str, Any]:
    pred = answer.letter
    return {
        "predicted_letter": pred or "",
        "answer_via": answer.via,
        "correct": pred == gold.correct_letter,
        "invalid": pred is None,
        "chose_distractor": (pred == gold.distractor_letter) if gold.distractor_letter else None,
        "predicted_value": gold.option_values.get(pred or "", ""),
    }


def reference_heuristics(question: Question, gold: GoldLabel) -> dict[str, Any]:
    """Label-free answer heuristics, scored for reference. On the sample the correct
    option is often the shortest one (see docs/DATASET_NOTES.md), so model results
    should be read against this line."""
    shortest = min(question.options, key=lambda o: (len(o.text), o.letter)).letter
    return {"shortest_option": shortest, "shortest_option_correct": shortest == gold.correct_letter}


def evolution_diagnostics(gold: GoldLabel, session_log: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """For evolved items: did the method write memory while observing the session(s)
    on each evolution date? Evolution-event conversations share their date with the
    ground-truth evolution record (see docs/DATASET_NOTES.md)."""
    if not gold.preference_evolution:
        return None
    out = []
    for step in gold.preference_evolution.get("evolution_history", []):
        day = str(step.get("date", ""))[:10]
        same_day = [s for s in session_log if s["date"][:10] == day]
        out.append(
            {
                "date": day,
                "event_name": step.get("event_name", ""),
                "replayed_sessions": [s["index"] for s in same_day],
                "memory_writes": sum(s["file_ops"] for s in same_day),
            }
        )
    return out


def bootstrap_ci(values: list[int], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(n_boot))
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1])


def _acc(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "accuracy": None, "ci95": None}
    vals = [int(r["correct"]) for r in rows]
    lo, hi = bootstrap_ci(vals)
    return {"n": len(rows), "accuracy": mean(vals), "ci95": [lo, hi]}


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in results if not r.get("error")]
    evolved = [r for r in scored if r["gold"]["has_evolved"]]
    static = [r for r in scored if not r["gold"]["has_evolved"]]
    with_distractor = [r for r in evolved if r["gold"]["distractor_letter"]]
    wrong_with_distractor = [r for r in with_distractor if not r["correct"] and not r["invalid"]]

    overall, evo, sta = _acc(scored), _acc(evolved), _acc(static)
    by_generator = defaultdict(list)
    for r in scored:
        by_generator[r["gold"]["generator"]].append(r)

    evo_diag = [r["evolution_diagnostics"] for r in evolved if r.get("evolution_diagnostics")]
    usage_keys = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    tokens = {k: sum(r["stats"]["usage"][k] for r in results) for k in usage_keys}

    return {
        "n_items": len(results),
        "n_errors": len(results) - len(scored),
        "n_invalid": sum(r["invalid"] for r in scored),
        "overall": overall,
        "evolved": evo,
        "static": sta,
        "evolved_minus_static": (
            evo["accuracy"] - sta["accuracy"] if evo["n"] and sta["n"] else None
        ),
        "distractor": {
            "n_evolved_with_distractor": len(with_distractor),
            "chose_distractor_rate": (
                mean(int(r["chose_distractor"]) for r in with_distractor) if with_distractor else None
            ),
            "n_wrong": len(wrong_with_distractor),
            "distractor_share_of_errors": (
                mean(int(r["chose_distractor"]) for r in wrong_with_distractor)
                if wrong_with_distractor
                else None
            ),
            "chance_share_of_errors": 0.25,
        },
        "by_generator": {g: _acc(rows) for g, rows in sorted(by_generator.items())},
        "memory": {
            "evolved_items_with_write_on_evolution_date": (
                mean(int(any(d["memory_writes"] > 0 for d in diag)) for diag in evo_diag) if evo_diag else None
            ),
            "mean_file_ops": mean(r["stats"]["file_ops"] for r in results) if results else None,
            "mean_file_reads": mean(r["stats"]["file_reads"] for r in results) if results else None,
        },
        "cost": {
            "tokens_total": tokens,
            "model_calls_total": sum(r["stats"]["model_calls"] for r in results),
            "mean_latency_s": mean(r["latency_s"] for r in results) if results else None,
            "tool_errors_total": sum(r["stats"]["tool_errors"] for r in results),
            "refusals_total": sum(r["stats"]["refusals"] for r in results),
        },
        "chance": CHANCE,
        "shortest_option_accuracy": (
            mean(int(r["reference_heuristics"]["shortest_option_correct"]) for r in scored)
            if scored and all("reference_heuristics" in r for r in scored)
            else None
        ),
    }


def _pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{100 * x:5.1f}%"


def format_summary(agg: dict[str, Any], title: str) -> str:
    def acc_line(label: str, a: dict[str, Any]) -> str:
        ci = f"[{100 * a['ci95'][0]:.1f}, {100 * a['ci95'][1]:.1f}]" if a["ci95"] else ""
        return f"  {label:<16} {_pct(a['accuracy'])}  {ci:<14} (n={a['n']})"

    d, m, c = agg["distractor"], agg["memory"], agg["cost"]
    lines = [
        "=" * 64,
        f"  {title}",
        "=" * 64,
        acc_line("Overall", agg["overall"]),
        acc_line("Evolved", agg["evolved"]),
        acc_line("Static", agg["static"]),
        f"  Chance           {_pct(agg['chance'])}",
        f"  Shortest-option  {_pct(agg.get('shortest_option_accuracy'))}  (label-free reference heuristic)",
        f"  Errors: {agg['n_errors']}   Invalid answers: {agg['n_invalid']}",
        f"  Old-preference distractor chosen: {_pct(d['chose_distractor_rate'])} of "
        f"{d['n_evolved_with_distractor']} evolved items; {_pct(d['distractor_share_of_errors'])} of "
        f"{d['n_wrong']} errors (chance 25.0%)",
        f"  Wrote memory on an evolution-event date: {_pct(m['evolved_items_with_write_on_evolution_date'])}"
        " of evolved items",
        f"  Mean file ops/item: {m['mean_file_ops']}   mean file reads/item: {m['mean_file_reads']}",
        f"  Model calls: {c['model_calls_total']}   tokens: {c['tokens_total']}",
        f"  Mean latency/item: {c['mean_latency_s']:.2f}s" if c["mean_latency_s"] is not None else "",
    ]
    if len(agg["by_generator"]) > 1:
        lines.append("  By generator:")
        lines += [acc_line(g, a) for g, a in agg["by_generator"].items()]
    lines.append("=" * 64)
    return "\n".join(line for line in lines if line)
