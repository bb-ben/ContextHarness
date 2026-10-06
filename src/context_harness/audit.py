"""Evaluator-side dataset audit: ordering anomalies and option-length bias.

Reads labels, so it must never run inside an agent episode. See
docs/DATASET_NOTES.md for what each measure is checking.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from .dataset import iter_rows, parse_item
from .evaluation import bootstrap_ci
from .hb_compat import parse_sessions


def _dt(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).strip())


def _rank(letter: str, sizes: dict[str, int]) -> int:
    """1 = strictly shortest; ties share the better rank."""
    return 1 + sum(v < sizes[letter] for v in sizes.values())


def audit_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    item = parse_item(row, index, order="chronological")
    segments = parse_sessions(row["conversation"])
    history, probe = segments[:-1], segments[-1]
    probe_dt = _dt(probe["date"])
    after = {i for i, s in enumerate(history) if _dt(s["date"]) > probe_dt}
    scenarioless = {i for i, s in enumerate(history) if not s["scenario"]}
    prefix = next((i for i, s in enumerate(history) if s["scenario"]), len(history))

    gold = item.gold
    options = item.episode.question.options
    chars = {o.letter: len(o.text) for o in options}
    words = {o.letter: len(o.text.split()) for o in options}
    correct = gold.correct_letter
    # Same tie-breaks as evaluation.reference_heuristics.
    shortest = min(options, key=lambda o: (len(o.text), o.letter)).letter
    shortest_words = min(options, key=lambda o: (len(o.text.split()), o.letter)).letter
    longest = max(options, key=lambda o: (len(o.text), -ord(o.letter))).letter
    others = [v for k, v in chars.items() if k != correct]

    evolution = gold.preference_evolution or {}
    evo_dates = [_dt(h["date"]) for h in evolution.get("evolution_history", []) if h.get("date")]

    return {
        "index": index,
        "item_id": gold.item_id,
        "generator": gold.generator,
        "user_id": gold.user_id,
        "has_evolved": gold.has_evolved,
        "has_distractor": gold.distractor_letter is not None,
        "n_history": len(history),
        "n_replayed": item.ordering.n_replayed,
        "n_after_probe": len(after),
        "text_monotonic": item.ordering.text_order_monotonic,
        "n_moved": item.ordering.n_moved,
        "n_scenarioless": len(scenarioless),
        "n_scenarioless_after_probe": len(after & scenarioless),
        "scenarioless_prefix": prefix,
        "probe_turns": len(probe["turns"]),
        "probe_ends_with_user": bool(probe["turns"]) and probe["turns"][-1]["role"] == "user",
        "n_evolution_events": len(evo_dates),
        "n_evolution_after_probe": sum(d > probe_dt for d in evo_dates),
        "option_chars": chars,
        "correct_letter": correct,
        "correct_char_rank": _rank(correct, chars),
        "correct_word_rank": _rank(correct, words),
        "shortest_correct": shortest == correct,
        "shortest_by_words_correct": shortest_words == correct,
        "longest_correct": longest == correct,
        "correct_vs_median_other": chars[correct] / median(others),
        "distractor_char_rank": _rank(gold.distractor_letter, chars) if gold.distractor_letter else None,
    }


def _group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    shortest = [int(r["shortest_correct"]) for r in rows]
    ranks = Counter(r["correct_char_rank"] for r in rows)
    affected = [r for r in rows if r["n_after_probe"]]
    return {
        "n": n,
        "shortest_option_accuracy": mean(shortest),
        "shortest_option_ci95": list(bootstrap_ci(shortest)),
        "shortest_by_words_accuracy": mean(int(r["shortest_by_words_correct"]) for r in rows),
        "longest_option_accuracy": mean(int(r["longest_correct"]) for r in rows),
        "correct_length_rank_share": {k: ranks.get(k, 0) / n for k in range(1, 6)},
        "median_correct_vs_median_other_length": median(r["correct_vs_median_other"] for r in rows),
        "items_with_post_probe_sessions": len(affected),
        "share_items_with_post_probe_sessions": len(affected) / n,
        "users_with_post_probe_sessions": len({r["user_id"] for r in affected}),
        "users_total": len({r["user_id"] for r in rows}),
        "post_probe_sessions_total": sum(r["n_after_probe"] for r in rows),
        "mean_post_probe_sessions_when_present": mean(r["n_after_probe"] for r in affected) if affected else 0,
        "share_text_order_not_monotonic": mean(int(not r["text_monotonic"]) for r in rows),
        "share_scenarioless_prefix_ge_3": mean(int(r["scenarioless_prefix"] >= 3) for r in rows),
        "items_with_evolution_event_after_probe": sum(1 for r in rows if r["n_evolution_after_probe"]),
        "share_probe_ends_with_user": mean(int(r["probe_ends_with_user"]) for r in rows),
        "mean_history_sessions": mean(r["n_history"] for r in rows),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if "error" not in r]
    errors = [r for r in records if "error" in r]
    by_gen: dict[str, list] = defaultdict(list)
    for r in ok:
        by_gen[r["generator"]].append(r)
    with_distractor = [r for r in ok if r["has_distractor"]]
    return {
        "n_items": len(records),
        "n_parse_errors": len(errors),
        "parse_errors": errors[:20],
        "overall": _group(ok),
        "by_generator": {g: _group(rows) for g, rows in sorted(by_gen.items())},
        "evolved": _group([r for r in ok if r["has_evolved"]]),
        "static": _group([r for r in ok if not r["has_evolved"]]),
        "distractor": {
            "n": len(with_distractor),
            "distractor_shorter_than_correct_share": (
                mean(int(r["distractor_char_rank"] < r["correct_char_rank"]) for r in with_distractor)
                if with_distractor else None
            ),
            "distractor_is_shortest_share": (
                mean(int(r["distractor_char_rank"] == 1) for r in with_distractor) if with_distractor else None
            ),
        },
    }


def run_audit(rows: Iterable[dict[str, Any]], out_dir: Path, max_items: int | None = None,
              progress=print) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with (out_dir / "audit_items.jsonl").open("w") as f:
        for i, row in enumerate(rows):
            if max_items is not None and i >= max_items:
                break
            try:
                record = audit_row(row, i)
            except Exception as e:  # keep going; parse failures are themselves findings
                record = {"index": i, "item_id": row.get("id", ""), "error": f"{type(e).__name__}: {e}"}
            records.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (i + 1) % 250 == 0:
                progress(f"  audited {i + 1} items")
    summary = summarize(records)
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def format_audit(summary: dict[str, Any]) -> str:
    def pct(x):
        return "  n/a" if x is None else f"{100 * x:5.1f}%"

    def row(label, g):
        if not g.get("n"):
            return f"  {label:<16} (none)"
        ranks = " ".join(pct(g["correct_length_rank_share"][k]) for k in range(1, 6))
        return (
            f"  {label:<16} n={g['n']:<5} shortest {pct(g['shortest_option_accuracy'])}  "
            f"ranks 1..5 [{ranks}]  post-probe items {pct(g['share_items_with_post_probe_sessions'])} "
            f"({g['post_probe_sessions_total']} sessions, {g['users_with_post_probe_sessions']}/{g['users_total']} users)"
        )

    o = summary["overall"]
    lines = [
        "=" * 72,
        "  Dataset audit (evaluator-side; uses labels)",
        "=" * 72,
        f"  Items: {summary['n_items']}   parse errors: {summary['n_parse_errors']}",
        f"  Shortest-option rule: {pct(o.get('shortest_option_accuracy'))} "
        f"(95% CI {pct(o['shortest_option_ci95'][0])}-{pct(o['shortest_option_ci95'][1])}); "
        f"by words {pct(o.get('shortest_by_words_accuracy'))}; longest-option rule {pct(o.get('longest_option_accuracy'))}; "
        "chance 20.0%",
        f"  Correct option length / median other option: {o.get('median_correct_vs_median_other_length', 0):.2f} (median)",
        f"  Text order not chronological: {pct(o.get('share_text_order_not_monotonic'))} of items; "
        f"evolution events dated after the question: {o.get('items_with_evolution_event_after_probe')} items",
        "",
        row("overall", o),
    ]
    lines += [row(g, s) for g, s in summary["by_generator"].items()]
    lines += [row("evolved", summary["evolved"]), row("static", summary["static"])]
    d = summary["distractor"]
    lines.append(
        f"  Old-preference distractor shorter than correct: {pct(d['distractor_shorter_than_correct_share'])} "
        f"of {d['n']}; distractor is shortest: {pct(d['distractor_is_shortest_share'])}"
    )
    lines.append("=" * 72)
    return "\n".join(lines)
