"""Per-design tables for MemConflict and DynamicMem runs, with the spread across people
and users, and cost from the traces (so runs stopped early still count).

Usage: python scenarios/summarize_benchmarks.py runs/<run> [runs/<run> ...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from context_harness.benchmarks.dynamicmem import HEADLINE  # noqa: E402
from context_harness.benchmarks.memconflict import TYPES, summarize  # noqa: E402
from context_harness.models.base import Usage  # noqa: E402
from context_harness.models.pricing import PRICES_PER_MTOK, cost_usd  # noqa: E402


def trace_cost(path: Path) -> float:
    """Spend recorded in a trace; responses name a dated snapshot (gpt-5.4-mini-2026-03-17)."""
    total = 0.0
    if not path.exists():  # an answer-only replay has no trace; its rows carry their cost
        return total
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if "usage" in row and row.get("model"):
            u = row["usage"]
            model = max((m for m in PRICES_PER_MTOK if row["model"].startswith(m)), key=len, default=row["model"])
            total += cost_usd(Usage(**{k: u.get(k, 0) for k in Usage.__dataclass_fields__}), model) or 0.0
    return total


def judge_cost_estimate(path: Path, model: str = "gpt-5.4-mini") -> float:
    """DynamicMem's evaluator doesn't report its usage; estimate it from the judge inputs
    and outputs it saves (4 characters per token, no hidden reasoning)."""
    if not path.exists():
        return 0.0
    calls = [c for v in json.loads(path.read_text()).values() if isinstance(v, list) for c in v if isinstance(c, dict)]
    text = lambda x: x if isinstance(x, str) else json.dumps(x)  # noqa: E731
    usage = Usage(input_tokens=sum(len(text(c.get("prompt", ""))) for c in calls) // 4,
                  output_tokens=sum(len(text(c.get("raw_output", ""))) for c in calls) // 4)
    return cost_usd(usage, model) or 0.0


def fmt(xs: list[float]) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    return f"{mean(xs):.2f}" + (f" ({min(xs):.2f}-{max(xs):.2f})" if len(xs) > 1 and min(xs) != max(xs) else "")


def main(runs: list[Path]) -> None:
    mc_rows: dict[str, dict[str, list[dict]]] = defaultdict(dict)  # method -> person -> rows
    dm_scores: dict[str, dict[str, dict]] = defaultdict(dict)  # method -> user -> scores
    cost: dict[str, float] = defaultdict(float)
    for run in runs:
        for q in run.glob("*/person_*/questions.jsonl"):
            rows = [json.loads(line) for line in q.read_text().splitlines()]
            method = q.parent.parent.name
            mc_rows[method][f"{run.name}/{q.parent.name}"] = rows
            cost[f"memconflict {method}"] += trace_cost(q.parent / "trace.jsonl") + sum(
                r["answer_cost_usd"] + r["judge_cost_usd"] for r in rows)
        for e in run.glob("*/*/tce_eval.json"):
            summary = json.loads(e.read_text())["summary"]
            method = e.parent.parent.name
            dm_scores[method][e.parent.name] = {k: summary.get(v) for k, v in HEADLINE.items()}
            cost[f"dynamicmem {method}"] += trace_cost(e.parent / "trace.jsonl")
            cost["dynamicmem evaluator judge (est.)"] += judge_cost_estimate(e.parent / "judge_io.json")
    if mc_rows:
        print("MemConflict (answer accuracy 0-1; mean over people, range in brackets)")
        print(f"{'design':<9} {'people':>6} {'n':>4}  {'overall':<17} " + " ".join(f"{t.split('_')[0]:<17}" for t in TYPES)
              + f" {'update aware':<17} {'conflict noticed':<17} memory has evidence")
        for method, people in sorted(mc_rows.items()):
            per = [summarize(rs) for rs in people.values()]
            everyone = summarize([r for rs in people.values() for r in rs])
            print(f"{method:<9} {len(per):>6} {everyone['n']:>4}  {fmt([p['answer_accuracy'] for p in per]):<17} "
                  + " ".join(f"{fmt([p[t]['answer_accuracy'] for p in per]):<17}" for t in TYPES)
                  + f" {fmt([p['dynamic_conflict']['update_awareness'] for p in per]):<17}"
                  + f" {fmt([p['static_conflict']['conflict_recognition'] for p in per]):<17}"
                  + f" {fmt([mean(r['memory_support'] for r in rs) for rs in people.values()])}")
        print("  n per type: " + ", ".join(f"{m}: " + "/".join(str(summarize([r for rs in p.values() for r in rs])[t]['n'])
                                                                for t in TYPES) for m, p in sorted(mc_rows.items())))
    if dm_scores:
        print("\nDynamicMem (0-1; DynamicMem's own evaluator; mean over users, range in brackets)")
        print(f"{'design':<9} {'users':>5}  " + " ".join(f"{k:<22}" for k in HEADLINE))
        for method, users in sorted(dm_scores.items()):
            print(f"{method:<9} {len(users):>5}  " + " ".join(f"{fmt([u[k] for u in users.values()]):<22}" for k in HEADLINE))
    if cost:
        print("\nCost from traces, our judge and an estimate for DynamicMem's evaluator: "
              + ", ".join(f"{k} ${v:.2f}" for k, v in sorted(cost.items())) + f"; total ${sum(cost.values()):.2f}")


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:]])
