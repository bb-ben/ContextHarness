"""Rebuild DynamicMem predictions from a run's trace and score them again with
DynamicMem's evaluator (for runs made before plain-text attribute answers were kept).

Usage: python scenarios/rescore_dynamicmem.py runs/<run>/<method>/<user> [...]
The previous score is kept as tce_eval_before_rescore.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from context_harness.benchmarks.dynamicmem import evaluate_with_dynamicmem, rebuild_snapshots  # noqa: E402

DATA, REPO = Path("data/external/dynamicmem_data"), Path("data/external/DynamicMem")

for out in map(Path, sys.argv[1:]):
    user_dir = DATA / out.name
    if (out / "tce_eval.json").exists() and not (out / "tce_eval_before_rescore.json").exists():
        (out / "tce_eval.json").rename(out / "tce_eval_before_rescore.json")
    changed = rebuild_snapshots(out, user_dir)
    print(out, f"{changed} predictions changed;", evaluate_with_dynamicmem(REPO, user_dir, out, "gpt-5.4-mini"))
