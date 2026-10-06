"""Command line: `context-harness run | summarize | inspect-data`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .dataset import iter_rows, load_items, parse_item
from .evaluation import aggregate, format_summary
from .methods import METHODS, MethodConfig
from .models import ModelSettings, make_model
from .orchestrator import RunConfig, run
from .workspace import WorkspaceBudgets

DEFAULT_MODEL = "claude-opus-5-5"


def _add_dataset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="sample", help="HF config: sample (10 items) or benchmark")
    p.add_argument("--dataset", help="Local .parquet/.jsonl/.json file instead of the HF dataset")
    p.add_argument("--max-items", type=int, help="Use only the first N items")
    p.add_argument("--items", help="Comma-separated item indices, e.g. 0,3,7")
    p.add_argument("--order", choices=["chronological", "text"], default="chronological",
                   help="chronological: sort by date, drop sessions dated after the probe (default); "
                        "text: the benchmark's exact text order and content")
    p.add_argument("--no-scenarios", action="store_true", help="Hide generator 'Scenario:' notes")
    p.add_argument("--max-sessions", type=int,
                   help="Replay only the N most recent sessions (smoke tests / cost control)")


def _load(args: argparse.Namespace):
    indices = [int(x) for x in args.items.split(",")] if args.items else None
    return load_items(
        config=args.config,
        source=args.dataset,
        max_items=args.max_items,
        item_indices=indices,
        order=args.order,
        show_scenarios=not args.no_scenarios,
        max_sessions=args.max_sessions,
    )


def cmd_run(args: argparse.Namespace) -> int:
    settings = ModelSettings(
        model=args.model,
        max_tokens=args.max_tokens,
        effort=None if args.effort == "none" else args.effort,
        fallbacks=not args.no_fallbacks,
    )
    method_config = MethodConfig(
        budgets=WorkspaceBudgets(context_max_chars=args.context_chars, file_max_chars=args.file_chars),
        max_update_steps=args.max_update_steps,
        max_answer_steps=args.max_answer_steps,
        answer_budget_chars=args.answer_budget_chars,
        summary_max_chars=args.summary_chars,
    )
    items = _load(args)
    dataset_info = {
        "config": args.config,
        "source": args.dataset,
        "order": args.order,
        "show_scenarios": not args.no_scenarios,
        "max_sessions": args.max_sessions,
        "item_indices": [it.index for it in items],
    }
    cfg = RunConfig(args.method, settings, method_config, dataset_info, args.log_prompts)
    run_name = args.run_name or f"{time.strftime('%Y%m%d-%H%M%S')}_{args.method}_{args.model.replace('/', '_')}"
    run_dir = Path(args.out) / run_name
    print(f"Run directory: {run_dir}")
    _, summary = run(items, make_model(settings), run_dir, cfg, resume=args.resume)
    print(format_summary(summary, f"{args.method} / {args.model}  ({run_dir.name})"))
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    for run_dir in map(Path, args.run_dirs):
        results = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines() if line]
        cfg = json.loads((run_dir / "run_config.json").read_text())
        summary = aggregate(results)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(format_summary(summary, f"{cfg['method']} / {cfg['model_settings']['model']}  ({run_dir.name})"))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Print per-item structure and ordering anomalies (no labels beyond metadata)."""
    for i, row in enumerate(iter_rows(args.config, args.dataset)):
        if args.max_items is not None and i >= args.max_items:
            break
        item = parse_item(row, i, order=args.order)
        o = item.ordering
        print(
            f"[{i}] {item.gold.user_id:<26} evolved={item.gold.has_evolved!s:<5} "
            f"history={o.n_history_sessions_in_text:>3} replayed={o.n_replayed:>3} "
            f"text_monotonic={o.text_order_monotonic!s:<5} moved={o.n_moved:>3} "
            f"dropped_after_probe={len(o.dropped_after_probe):>2} probe_turns={len(item.episode.probe.turns)}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="context-harness", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="Replay items through a method and score the final question")
    p.add_argument("--method", choices=sorted(METHODS), default="agent")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"'fake' or an Anthropic model id (default {DEFAULT_MODEL})")
    _add_dataset_args(p)
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max", "none"],
                   help="output_config.effort; 'none' omits it (required for Haiku 4.5)")
    p.add_argument("--no-fallbacks", action="store_true", help="Disable server-side refusal fallback")
    p.add_argument("--context-chars", type=int, default=8000, help="context.md limit")
    p.add_argument("--file-chars", type=int, default=20000, help="Limit for every other workspace file")
    p.add_argument("--answer-budget-chars", type=int, default=48000,
                   help="Memory text visible at answer time, for every method")
    p.add_argument("--summary-chars", type=int, default=8000, help="Rolling-summary limit")
    p.add_argument("--max-update-steps", type=int, default=8)
    p.add_argument("--max-answer-steps", type=int, default=6)
    p.add_argument("--out", default="runs")
    p.add_argument("--run-name")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log-prompts", action="store_true", help="Store full request text in traces")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("summarize", help="Recompute and print aggregates for run directories")
    p.add_argument("run_dirs", nargs="+")
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("inspect-data", help="Print per-item session structure and ordering anomalies")
    _add_dataset_args(p)
    p.set_defaults(func=cmd_inspect)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
