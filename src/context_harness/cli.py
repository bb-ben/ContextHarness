"""Command line: `context-harness run | summarize | inspect-memory | inspect-data | audit-data`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .dataset import iter_rows, load_items, parse_item
from .evaluation import aggregate, format_summary
from .methods import METHODS, MethodConfig
from .models import ModelSettings, RequestCap, make_model
from .models.pricing import CostCap
from .orchestrator import RunConfig, run
from .workspace import CONTEXT_FILE, LAYOUTS

DEFAULT_MODEL = "gemini-3.7-flash"


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
        requests_per_minute=args.rpm,
        timeout_s=args.timeout,
        fallbacks=not args.no_fallbacks,
    )
    method_config = MethodConfig(
        layout=LAYOUTS[args.layout].with_limit(CONTEXT_FILE, args.context_chars),
        preload_state=not args.no_preload,
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
    model = make_model(settings)
    if args.max_requests is not None:
        model = RequestCap(model, args.max_requests)
        print(f"Request cap: at most {args.max_requests} model requests in this invocation")
    if args.max_cost_usd is not None:
        model = CostCap(model, args.max_cost_usd)
        print(f"Cost cap: pause once estimated spend reaches ${args.max_cost_usd:.2f} in this invocation")
    _, summary = run(items, model, run_dir, cfg, resume=args.resume)
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


def cmd_inspect_memory(args: argparse.Namespace) -> int:
    """Per-event file access and memory-check violations for one item directory."""
    item_dir = Path(args.item_dir)
    path = item_dir / "memory_report.json"
    if not path.exists():
        print(f"No memory_report.json in {item_dir} (the item may be unfinished, or the method keeps no workspace).")
        return 1
    report = json.loads(path.read_text())
    files = sorted({p for r in report["per_event"] for p in r["access"]})
    short = {p: p.split("/")[-1].removesuffix(".md")[:11] for p in files}
    print(f"{report['events']} events · layout {report['layout']} · "
          f"{report['events_with_violations']} events with violations")
    print("Access per event (r = reads, w = writes, ! = errors):")
    print("  event   " + " ".join(f"{short[p]:>11}" for p in files) + "  violations")
    for r in report["per_event"]:
        cells = []
        for p in files:
            c = r["access"].get(p, {})
            cell = "".join(f"{c[k]}{k[0] if k != 'errors' else '!'}" for k in ("reads", "writes", "errors") if c.get(k))
            cells.append(f"{cell or '.':>11}")
        v = ", ".join(f"{k}×{n}" for k, n in r["violations"].items())
        print(f"  {r['event_id']:<7} " + " ".join(cells) + f"  {v}")
    print("Violations by code:", report["violations_by_code"] or "none")
    print("Final metrics:", json.dumps(report["final_metrics"]))
    for v in report["violations"][: args.limit]:
        print(f"  [{v['event_id']}] {v['code']}: {v['path']}:{v['line']} {v.get('id') or ''} {v['detail']}")
    return 0


def _short(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def cmd_memory_timeline(args: argparse.Namespace) -> int:
    """What each event added, changed, archived and removed in each kind of memory."""
    path = Path(args.item_dir) / "memory_report.json"
    if not path.exists():
        print(f"No memory_report.json in {args.item_dir}.")
        return 1
    timeline = json.loads(path.read_text()).get("timeline", [])
    kinds = args.kind or ["working_context", "episodic", "state", "decisions"]
    n = 10_000 if args.full else 110
    totals: dict[str, dict[str, int]] = {k: {"added": 0, "changed": 0, "moved": 0, "removed": 0} for k in kinds}
    for t in timeline:
        lines = []
        if "working_context" in kinds:
            wc = t["working_context"]
            for key in ("added", "removed"):
                totals["working_context"][key] += len(wc[key])
            if wc["added"] or wc["removed"]:
                cites = " ".join([f"+{i}" for i in wc["cited_added"]] + [f"-{i}" for i in wc["cited_removed"]])
                lines.append(f"  working context: {wc['bullets']} bullets" + (f" (cites {cites})" if cites else ""))
                lines += [f"    + {_short(b, n)}" for b in wc["added"]] + [f"    - {_short(b, n)}" for b in wc["removed"]]
        for kind in ("episodic", "state", "decisions"):
            if kind not in kinds or kind not in t:
                continue
            d = t[kind]
            for key in totals[kind]:
                totals[kind][key] += len(d[key])
            items = [f"    + {a['id']} [{a['status']}] {_short(a['statement'], n)}" for a in d["added"]]
            items += [f"    ~ {c['id']} " + (f"{c['status'][0]} -> {c['status'][1]}" if c["status"][0] != c["status"][1] else "")
                      + "".join(f" {k}: {v[1]}" for k, v in c["fields"].items() if k not in ("until",)) for c in d["changed"]]
            items += [f"    > {m['id']} moved {m['from']} -> {m['to']}" for m in d["moved"]]
            items += [f"    x {r['id']} REMOVED: {_short(r['statement'], n)}" for r in d["removed"]]
            if items:
                lines.append(f"  {kind}:")
                lines += items
        if lines:
            print(f"{t['event_id']} · {t['date']}")
            print("\n".join(lines))
    print("\nTotals over the run:")
    for kind, c in totals.items():
        print(f"  {kind:<16} " + "  ".join(f"{k} {v}" for k, v in c.items() if kind != "working_context" or k in ("added", "removed")))
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


def cmd_audit(args: argparse.Namespace) -> int:
    """Evaluator-side audit of ordering anomalies and option-length bias (reads labels)."""
    from .audit import format_audit, run_audit

    summary = run_audit(iter_rows(args.config, args.dataset), Path(args.out), args.max_items)
    print(format_audit(summary))
    print(f"Per-item records and summary written to {args.out}/")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="context-harness", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="Replay items through a method and score the final question")
    p.add_argument("--method", choices=sorted(METHODS), default="agent")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"'fake', a gemini-* id, or an Anthropic model id (default {DEFAULT_MODEL})")
    _add_dataset_args(p)
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--effort", default="medium", choices=["minimal", "low", "medium", "high", "xhigh", "max", "none"],
                   help="Reasoning effort: Gemini thinking level, OpenAI reasoning_effort, or Anthropic effort; "
                        "'none' omits it")
    p.add_argument("--rpm", type=float, help="Pace requests to at most this many per minute (free-tier quotas)")
    p.add_argument("--timeout", type=float, default=300.0,
                   help="Seconds to wait for one model response before retrying (default 300)")
    p.add_argument("--max-cost-usd", type=float,
                   help="Pause once estimated spend reaches this many dollars in this invocation (priced models only)")
    p.add_argument("--max-requests", type=int,
                   help="Stop after this many model requests in this invocation (the run pauses; continue with --resume)")
    p.add_argument("--layout", choices=sorted(LAYOUTS), default="general", help="Workspace file layout")
    p.add_argument("--no-fallbacks", action="store_true", help="Disable server-side refusal fallback")
    p.add_argument("--context-chars", type=int, default=6000, help="context.md limit")
    p.add_argument("--no-preload", action="store_true",
                   help="Contract v1 behavior: show only context.md automatically, not the state files")
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

    p = sub.add_parser("inspect-memory", help="Show per-event file access and memory violations for one item")
    p.add_argument("item_dir", help="runs/<run>/items/<item>")
    p.add_argument("--limit", type=int, default=30, help="Detailed violations to print")
    p.set_defaults(func=cmd_inspect_memory)

    p = sub.add_parser("memory-timeline", help="Per event: what changed in working context, episodic memory and state")
    p.add_argument("item_dir", help="runs/<run>/items/<item>")
    p.add_argument("--kind", action="append", choices=["working_context", "episodic", "state", "decisions"],
                   help="Show only this kind of memory (repeatable)")
    p.add_argument("--full", action="store_true", help="Do not truncate statements")
    p.set_defaults(func=cmd_memory_timeline)

    p = sub.add_parser("inspect-data", help="Print per-item session structure and ordering anomalies")
    _add_dataset_args(p)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("audit-data", help="Audit ordering anomalies and option-length bias (evaluator-side)")
    p.add_argument("--config", default="sample", help="HF config: sample or benchmark")
    p.add_argument("--dataset", help="Local .parquet/.jsonl/.json file instead of the HF dataset")
    p.add_argument("--max-items", type=int)
    p.add_argument("--out", default="data/audit", help="Directory for audit_items.jsonl and audit_summary.json")
    p.set_defaults(func=cmd_audit)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
