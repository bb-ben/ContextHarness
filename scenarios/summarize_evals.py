"""Combine `evaluate` summaries into per-design tables: means with the spread across runs.

Usage: python scenarios/summarize_evals.py runs/<eval run>/summary.json [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict
from statistics import mean

CRITERIA = ("faithfulness", "currency", "recall", "change_tracking", "provenance", "consistency")


def rows(paths: list[str]) -> list[dict]:
    """Rows from summary.json files, or from an evaluation folder's per-run eval.json files
    (an evaluation stopped early has no summary)."""
    out = []
    for path in map(Path, paths):
        if path.is_dir():
            out += [json.loads(f.read_text()) for f in sorted(path.glob("*/eval.json"))]
        else:
            out += json.loads(path.read_text())["rows"]
    return out


def fmt(xs: list[float]) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    return f"{mean(xs):.2f}" + (f" ({min(xs):.2f}-{max(xs):.2f})" if len(xs) > 1 and min(xs) != max(xs) else "")


def table(rs: list[dict]) -> None:
    by_method = defaultdict(list)
    for r in rs:
        by_method[r["method"]].append(r)
    print(f"{'design':<9} {'n':>3}  {'must-contain':<17} {'stale kept':<17} {'criteria /12':<17} "
          f"{'reply follows':<17} {'reply violates':<17} asks-known  agreement")
    for method, group in sorted(by_method.items()):
        mem = [g["memory"] for g in group]
        rep = [g["reply"] for g in group if "reply" in g]
        crit = [sum(m["criteria"][k] or 0 for k in CRITERIA) for m in mem]
        print(f"{method:<9} {len(group):>3}  {fmt([m['contain_rate'] for m in mem]):<17} "
              f"{fmt([m['stale_rate'] for m in mem]):<17} {fmt(crit):<17} "
              f"{fmt([x['follow_rate'] for x in rep]):<17} {fmt([x['violate_rate'] for x in rep]):<17} "
              f"{sum(bool(x['asks_for_known']) for x in rep)}/{len(rep):<9} "
              f"{fmt([m['agreement'] for m in mem] + [x['agreement'] for x in rep])}")
    print("\nper scenario (must-contain / stale kept / reply follows):")
    by_case = defaultdict(lambda: defaultdict(list))
    for r in rs:
        by_case[r["case"]][r["method"]].append(r)
    for case, methods in sorted(by_case.items()):
        cells = []
        for method, group in sorted(methods.items()):
            cells.append(f"{method}: {fmt([g['memory']['contain_rate'] for g in group])} / "
                         f"{fmt([g['memory']['stale_rate'] for g in group])} / "
                         f"{fmt([g['reply']['follow_rate'] for g in group if 'reply' in g])}")
        print(f"  {case:<9} " + "   |   ".join(cells))
    print("\ncriteria means by design:")
    for method, group in sorted(by_method.items()):
        print(f"  {method:<9} " + "  ".join(f"{k[:9]} {fmt([g['memory']['criteria'][k] for g in group])}"
                                            for k in CRITERIA))


if __name__ == "__main__":
    table(rows(sys.argv[1:]))
