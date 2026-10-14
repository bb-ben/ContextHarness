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
from .models.pricing import PRICES_PER_MTOK, CostCap
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
        answer_strategy=args.answer_strategy,
        review_step=args.review_step,
        check_feedback=not args.no_check_feedback,
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


def cmd_replay_answer(args: argparse.Namespace) -> int:
    """Rerun only the answer step against a saved memory, per strategy, N times each."""
    from collections import defaultdict

    from .models.base import PauseRun
    from .prompts import ANSWER_STRATEGIES
    from .replay import find_snapshot, load_source_item, replay_answer

    source = Path(args.source)
    snapshot, item = find_snapshot(source), load_source_item(source)
    settings = ModelSettings(model=args.model, max_tokens=args.max_tokens,
                             effort=None if args.effort == "none" else args.effort, timeout_s=args.timeout)
    model = make_model(settings)
    if args.max_requests is not None:
        model = RequestCap(model, args.max_requests)
    if args.max_cost_usd is not None:
        model = CostCap(model, args.max_cost_usd)
    run_dir = Path(args.out) / (args.run_name or f"{time.strftime('%Y%m%d-%H%M%S')}_replay_{source.name}")
    strategies = args.strategy or ["default"]
    print(f"Replaying the answer step for {item.gold.item_id[:60]}…  from {snapshot}")
    print(f"Strategies {strategies} × {args.repeats} repeats on {args.model}; results in {run_dir}")
    by_strategy: dict[str, list[dict]] = defaultdict(list)
    try:
        for strategy in strategies:
            cfg = RunConfig("agent", settings, MethodConfig(layout=LAYOUTS[args.layout], answer_strategy=strategy),
                            {"replay_of": str(source)}, args.log_prompts)
            for k in range(1, args.repeats + 1):
                r = replay_answer(item, snapshot, model, cfg, run_dir / f"{strategy}_{k}")
                by_strategy[strategy].append(r)
                print(f"  {strategy:<16} #{k}: {r['predicted_letter'] or '-'} "
                      f"({'correct' if r['correct'] else 'wrong'}), {r['stats']['model_calls']} call(s), "
                      f"{r['stats']['file_reads']} read(s)")
    except PauseRun as e:
        print(f"Stopped: {e}")
    gold = item.gold
    print(f"\nCorrect: {gold.correct_letter} ({gold.option_values.get(gold.correct_letter, '')})"
          + (f"; old-preference trap: {gold.distractor_letter}" if gold.distractor_letter else ""))
    summary = {}
    for strategy, rs in by_strategy.items():
        cost = sum(r["estimated_cost_usd"] or 0 for r in rs)
        summary[strategy] = {
            "answers": [r["predicted_letter"] for r in rs],
            "correct": sum(r["correct"] for r in rs),
            "chose_trap": sum(bool(r["chose_distractor"]) for r in rs),
            "mean_reads": sum(r["stats"]["file_reads"] for r in rs) / len(rs),
            "cost_usd": round(cost, 4),
        }
        s_ = summary[strategy]
        print(f"  {strategy:<16} answers {' '.join(a or '-' for a in s_['answers'])}  correct {s_['correct']}/{len(rs)}  "
              f"trap {s_['chose_trap']}/{len(rs)}  reads/answer {s_['mean_reads']:.1f}  cost ${cost:.3f}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps({"source": str(source), "model": args.model,
                                                      "effort": args.effort, "strategies": summary}, indent=2))
    return 0


def cmd_judge_memory(args: argparse.Namespace) -> int:
    """Grade finished runs' final memories against the ground-truth preference attributes."""
    from .judge import combine_repeats, judge_memory, supported_scores, write_judgement
    from .models.base import PauseRun
    from .replay import find_snapshot, load_source_item

    settings = ModelSettings(model=args.model, max_tokens=args.max_tokens,
                             effort=None if args.effort == "none" else args.effort, timeout_s=args.timeout)
    model = make_model(settings)
    if args.max_cost_usd is not None:
        model = CostCap(model, args.max_cost_usd)
    total = 0.0
    for source in map(Path, args.sources):
        item, snapshot = load_source_item(source), find_snapshot(source)
        runs = []
        try:
            for _ in range(args.repeats):
                runs.append(judge_memory(model, item.gold, snapshot))
        except PauseRun as e:
            print(f"Stopped: {e}")
        if not runs:
            break
        total += sum(r["estimated_cost_usd"] or 0 for r in runs)
        # runs/<run>/items/<item> is named by run and item, so two runs of one item don't collide
        name = f"{source.parent.parent.name}__{source.name}" if source.parent.name == "items" else source.name
        out = Path(args.out) / f"{name}.json" if args.out else source / "memory_judgement.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        j = combine_repeats(runs)
        if args.vet:
            vetted = {r["item_id"]: r for r in json.loads(Path(args.vet).read_text())}.get(item.gold.item_id)
            if vetted is not None:
                j["on_supported_changes"] = supported_scores(j, vetted.get("supported_attributes", []))
        write_judgement(out, j)
        print(f"{source}  ({j['memory_chars']} chars of memory) -> {out}")
        spread = (f" (single runs {j['current_fidelity_range'][0]}–{j['current_fidelity_range'][1]}, "
                  f"agreement {j['agreement']:.0%})" if len(runs) > 1 else "")
        print(f"  current fidelity {j['current_fidelity']}{spread}; changed {j['changed_fidelity']}; "
              f"stale {j['stale']}/{j['n_changed']}; changes recorded {j['changes_recorded']}/{j['n_changed']}")
        for a in j["attributes"]:
            expect = f"{a['original']} -> {a['current']}" if a["changed"] else a["current"]
            votes = f" {'/'.join(v[:3] for v in a['votes'])}" if "votes" in a else ""
            print(f"    {a['attribute']:<26} {expect:<24} {a['current_match']:<11}{votes:<14} {_short(a['memory_value'], 60)}")
        if s := j.get("on_supported_changes"):
            print(f"  on changes the conversations support (dropped {s['dropped'] or 'none'}): current fidelity "
                  f"{s['current_fidelity']}, changed {s['changed_fidelity']}, changes recorded "
                  f"{s['changes_recorded']}/{s['n_changed']}")
    print(f"Estimated judge cost: ${total:.3f}")
    return 0


def cmd_vet_items(args: argparse.Namespace) -> int:
    """Check that each item's labeled change shows up in the person's own words."""
    from .models.base import PauseRun
    from .vet import vet_item

    items = _load(args)
    settings = ModelSettings(model=args.model, max_tokens=args.max_tokens,
                             effort=None if args.effort == "none" else args.effort, timeout_s=args.timeout)
    model = make_model(settings)
    if args.max_cost_usd is not None:
        model = CostCap(model, args.max_cost_usd)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    try:
        for k, item in enumerate(items):
            try:
                r = vet_item(model, item)
            except RuntimeError as e:  # one item the model couldn't finish shouldn't lose the others
                print(f"[{k + 1}/{len(items)}] {item.gold.item_id[:50]}: ERROR {e}")
                continue
            reports.append(r)
            print(f"[{k + 1}/{len(items)}] {item.gold.item_id[:50]}: {r['verdict'].upper()}"
                  f"  ({r['sessions_after_change']} sessions after the change; "
                  f"correct option {'is' if r['correct_is_shortest'] else 'is not'} the shortest)")
            for a in r["attributes"]:
                mark = "supported" if a.get("change_supported") else "NOT supported"
                print(f"    {a['attribute']:<28} before old/new {a.get('before_original', 0)}/{a.get('before_current', 0)}"
                      f"  after old/new {a.get('after_original', 0)}/{a.get('after_current', 0)}  {mark}")
            if r.get("final_turn_conflict"):
                print(f"    question's final request conflicts with the label: {r['final_turn_conflict']}"
                      f" — {_short(r['final_turn_explanation'], 110)}")
    except PauseRun as e:
        print(f"Stopped: {e}")
    out.write_text(json.dumps(reports, indent=2))
    counts = {v: sum(r["verdict"] == v for r in reports) for v in ("usable", "memory_only", "suspect", "no_change")}
    print(f"{counts['usable']} usable, {counts['memory_only']} for memory grading only, {counts['suspect']} suspect, "
          f"{counts['no_change']} without a labeled change; report in {out}; "
          f"estimated cost ${sum(r['estimated_cost_usd'] or 0 for r in reports):.3f}")
    return 0


def cmd_respond_eval(args: argparse.Namespace) -> int:
    """Write a reply to each item's final conversation under several memory conditions; judge each."""
    from .entries import parse_workspace
    from .evaluation import utilization
    from .models.base import PauseRun
    from .models.pricing import cost_usd
    from .methods.profile import read_profile_memory
    from .replay import find_snapshot, load_source_item
    from .respond import (full_history_text, judge_response, respond_with_agent, respond_with_text, snapshot_kind,
                          write)

    def adapter(effort: str):
        m = make_model(ModelSettings(model=args.model, max_tokens=args.max_tokens,
                                     effort=None if effort == "none" else effort, timeout_s=args.timeout))
        return CostCap(m, args.max_cost_usd) if args.max_cost_usd is not None else m

    responder, judge = adapter(args.effort), adapter(args.judge_effort)
    conditions = args.condition or ["memory", "none"]
    run_dir = Path(args.out) / (args.run_name or f"{time.strftime('%Y%m%d-%H%M%S')}_respond")
    spent, rows = 0.0, []
    try:
        for source in map(Path, args.sources):
            item, snapshot = load_source_item(source), find_snapshot(source)
            kind = snapshot_kind(snapshot)
            label = f"{source.parent.parent.name}__{source.name}" if source.parent.name == "items" else source.name
            cfg = RunConfig("agent", responder.settings, MethodConfig(layout=LAYOUTS[args.layout]), {}, False)
            for condition in conditions:
                for k in range(1, args.repeats + 1):
                    if args.max_cost_usd is not None and spent >= args.max_cost_usd:
                        raise PauseRun(f"cost cap ${args.max_cost_usd:.2f} reached")
                    out = run_dir / label / f"{condition}_{k}"
                    if condition == "memory" and kind == "agent":
                        reply = respond_with_agent(item, snapshot, responder, cfg, out)
                        used = utilization(reply["applied"], parse_workspace(
                            {p: (snapshot / p).read_text() for p in cfg.method_config.layout.paths
                             if (snapshot / p).exists()}, cfg.method_config.layout))
                    elif condition == "memory" and kind == "summary":
                        reply, used = respond_with_text(item, (snapshot / "summary.md").read_text(), "a rolling summary",
                                                        responder, out), None
                    elif condition == "memory" and kind == "profile":
                        reply, used = respond_with_text(item, read_profile_memory(snapshot),
                                                        "a profile of who they are now, with evidence", responder,
                                                        out), None
                    elif condition == "full-history":
                        reply, used = respond_with_text(item, full_history_text(item), "every earlier conversation",
                                                        responder, out), None
                    else:
                        reply, used = respond_with_text(item, None, "", responder, out), None
                    reply_cost = cost_usd(reply.pop("usage"), args.model) or 0.0
                    grade = judge_response(judge, item.gold, item, reply["text"] or "", args.judge_repeats)
                    spent += reply_cost + grade["judge_cost_usd"]
                    row = {"source": label, "memory_kind": kind if condition == "memory" else condition,
                           "condition": condition, "repeat": k, **reply, "utilization": used, **grade,
                           "reply_cost_usd": round(reply_cost, 5)}
                    write(out / "reply.json", row)
                    rows.append(row)
                    used_note = (f"; rests on {len(used['live'])} live, {len(used['closed'])} replaced entries"
                                 if used else "")
                    print(f"{label[:44]:<44} {condition:<13} #{k}: changed attributes fit now "
                          f"{grade['changed_fit_current']}, fit before {grade['changed_fit_original']}{used_note}")
    except PauseRun as e:
        print(f"Stopped: {e}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write(run_dir / "summary.json", {"model": args.model, "rows": [
        {k: r[k] for k in ("source", "condition", "memory_kind", "repeat", "changed_fit_current",
                           "changed_fit_original", "all_fit_current", "utilization")} for r in rows],
        "estimated_cost_usd": round(spent, 4)})
    print(f"\nEstimated cost ${spent:.3f}; replies and grades in {run_dir}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Grade finished runs' memories and replies against reference checklists."""
    from collections import defaultdict

    from .judge import memory_text
    from .memeval import CRITERIA, evaluate_memory, evaluate_reply, load_checklists
    from .models.base import PauseRun
    from .models.pricing import cost_usd
    from .replay import find_snapshot, load_source_item
    from .respond import reply_from_memory, write

    def adapter(effort: str):
        m = make_model(ModelSettings(model=args.model, max_tokens=args.max_tokens,
                                     effort=None if effort == "none" else effort, timeout_s=args.timeout))
        return CostCap(m, args.max_cost_usd) if args.max_cost_usd is not None else m

    checklists = load_checklists(args.checklists)
    judge, responder = adapter(args.judge_effort), adapter(args.effort)
    run_dir = Path(args.out) / (args.run_name or f"{time.strftime('%Y%m%d-%H%M%S')}_evaluate")
    rows, spent = [], 0.0
    try:
        for source in map(Path, args.sources):
            item, snapshot = load_source_item(source), find_snapshot(source)
            checklist = checklists.get(item.gold.item_id)
            if checklist is None:
                print(f"skip {source}: no checklist for {item.gold.item_id}")
                continue
            if args.max_cost_usd is not None and spent >= args.max_cost_usd:
                raise PauseRun(f"cost cap ${args.max_cost_usd:.2f} reached")
            method = json.loads((source / "result.json").read_text())["method"]
            label = f"{source.parent.parent.name}__{source.name}" if source.parent.name == "items" else source.name
            out = run_dir / label
            out.mkdir(parents=True, exist_ok=True)
            mem = evaluate_memory(judge, item, memory_text(snapshot), checklist, args.repeats)
            row = {"source": label, "method": method, "case": item.gold.user_id.split("/")[-1], "memory": mem}
            spent += mem["judge_cost_usd"]
            if not args.no_reply:
                cfg = RunConfig("agent", responder.settings, MethodConfig(layout=LAYOUTS[args.layout]), {}, False)
                reply = reply_from_memory(item, snapshot, responder, cfg, out / "reply")
                reply_cost = cost_usd(reply.pop("usage"), args.model) or 0.0
                graded = evaluate_reply(judge, item, reply.get("text") or "", checklist, args.repeats)
                row["reply"] = {**graded, "text": reply.get("text"), "reply_cost_usd": round(reply_cost, 5)}
                spent += reply_cost + graded["judge_cost_usd"]
            write(out / "eval.json", row)
            rows.append(row)
            r = row.get("reply") or {}
            print(f"{label[:52]:<52} must-contain {mem['contain_rate']}  stale {mem['stale_rate']}  "
                  f"criteria {sum(v for v in mem['criteria'].values() if v is not None)}/{2 * len(CRITERIA)}"
                  + (f"  reply follows {r['follow_rate']} violates {r['violate_rate']}" if r else ""))
    except PauseRun as e:
        print(f"Stopped: {e}")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["case"])].append(row)
    print(f"\n{'method':<9} {'case':<10} {'n':>2}  must-contain  stale  " + "  ".join(k[:6] for k in CRITERIA)
          + "  reply-follows  reply-violates")
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")  # noqa: E731
    for (method, case), rs in sorted(groups.items()):
        crit = "  ".join(f"{mean([r['memory']['criteria'][k] for r in rs if r['memory']['criteria'][k] is not None]):6.2f}"
                         for k in CRITERIA)
        reps = [r["reply"] for r in rs if "reply" in r]
        print(f"{method:<9} {case:<10} {len(rs):>2}  {mean([r['memory']['contain_rate'] for r in rs]):12.2f}  "
              f"{mean([r['memory']['stale_rate'] for r in rs if r['memory']['stale_rate'] is not None]):5.2f}  {crit}  "
              f"{mean([x['follow_rate'] for x in reps]):13.2f}  {mean([x['violate_rate'] for x in reps]):14.2f}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write(run_dir / "summary.json", {"model": args.model, "judge_effort": args.judge_effort, "repeats": args.repeats,
                                     "rows": rows, "estimated_cost_usd": round(spent, 4)})
    print(f"\nEstimated cost ${spent:.3f}; evaluations in {run_dir}")
    return 0


def _bench_model(args: argparse.Namespace, effort: str, shared_with=None, model: str | None = None):
    m = make_model(ModelSettings(model=model or args.model, max_tokens=args.max_tokens,
                                 effort=None if effort == "none" else effort, timeout_s=args.timeout))
    if m.settings.model not in PRICES_PER_MTOK:  # the fake model: nothing to count
        return m
    return CostCap(m, args.max_cost_usd if args.max_cost_usd is not None else float("inf"), shared_with)


def _spent(model) -> float:
    return getattr(model, "spent", 0.0)  # shared by every cap made with shared_with=model


def cmd_memconflict(args: argparse.Namespace) -> int:
    """Replay MemConflict chains through memory methods; answer and grade sampled questions."""
    from .benchmarks import memconflict as mc
    from .models.base import PauseRun
    from .respond import write

    people = mc.load_people(Path(args.data))
    picked = [people[int(i)] for i in args.people.split(",")] if args.people else people[: args.max_people]
    model = _bench_model(args, args.effort)
    judge = _bench_model(args, args.judge_effort, shared_with=model if isinstance(model, CostCap) else None,
                         model=args.judge_model or ("fake" if args.model == "fake" else "gpt-5.4-mini"))
    run_dir = Path(args.out) / (args.run_name or f"{time.strftime('%Y%m%d-%H%M%S')}_memconflict")
    rows = []
    try:
        for method in args.methods.split(","):
            for n, person in enumerate(picked):
                if args.max_sessions:
                    person = {**person, "Full_Session_Chain": person["Full_Session_Chain"][: args.max_sessions]}
                got = mc.run_person(person, method, model, judge, run_dir / method / f"person_{n}", args.questions)
                rows += [{**r, "method": method} for r in got]
                print(f"{method:<8} person {n}: {len(got)} questions, accuracy "
                      f"{mc.summarize(got)['answer_accuracy']}  (spent ${_spent(model):.2f})")
    except PauseRun as e:
        print(f"Stopped: {e}")
    summary = {m: mc.summarize([r for r in rows if r["method"] == m]) for m in args.methods.split(",")}
    run_dir.mkdir(parents=True, exist_ok=True)
    write(run_dir / "summary.json", {"model": args.model, "effort": args.effort, "judge_model": judge.settings.model,
                                     "judge_effort": args.judge_effort,
                                     "questions_per_person": args.questions, "people": len(picked),
                                     "results": summary, "estimated_cost_usd": round(_spent(model), 4)})
    print(json.dumps(summary, indent=1))
    print(f"\nEstimated cost ${_spent(model):.3f}; results in {run_dir}")
    return 0


def cmd_dynamicmem(args: argparse.Namespace) -> int:
    """Replay DynamicMem app logs through memory methods; write predictions; score with its evaluator."""
    from .benchmarks import dynamicmem as dm
    from .models.base import PauseRun
    from .respond import write

    data = Path(args.data)
    users = args.users.split(",") if args.users else sorted(p.parent.name for p in data.glob("*/task_packs.json"))[: args.max_users]
    model = _bench_model(args, args.effort)
    run_dir = Path(args.out) / (args.run_name or f"{time.strftime('%Y%m%d-%H%M%S')}_dynamicmem")
    results = []
    try:
        for method in args.methods.split(","):
            for user in users:
                out = run_dir / method / user
                row = {"method": method, "user": user,
                       **dm.run_user(data / user, method, model, out, max_checkpoints=args.max_checkpoints)}
                if args.score:
                    row["scores"] = dm.evaluate_with_dynamicmem(Path(args.repo), data / user, out,
                                                                None if args.judge_model == "none" else args.judge_model)
                results.append(row)
                print(f"{method:<8} {user}: {row}  (spent on replay/answers ${_spent(model):.2f})")
    except PauseRun as e:
        print(f"Stopped: {e}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write(run_dir / "summary.json", {"model": args.model, "effort": args.effort, "judge_model": args.judge_model,
                                     "results": results, "estimated_cost_usd": round(_spent(model), 4),
                                     "note": "judge cost (DynamicMem's evaluator) is not included"})
    print(f"\nEstimated replay/answer cost ${_spent(model):.3f}; results in {run_dir}")
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
    p.add_argument("--review-step", action="store_true",
                   help="After each update, one more short call to re-read memory and fix contradictions")
    p.add_argument("--no-check-feedback", action="store_true",
                   help="Don't show the agent the memory checks' open problems")
    p.add_argument("--answer-strategy", default="dimension-first", choices=["default", "dimension-first"],
                   help="How the agent uses memory when answering (default: dimension-first)")
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

    from .prompts import ANSWER_STRATEGIES

    p = sub.add_parser("replay-answer", help="Rerun only the answer step against a saved memory")
    p.add_argument("source", help="A finished item: runs/<run>/items/<item> or examples/<run>")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--effort", default="medium", choices=["minimal", "low", "medium", "high", "xhigh", "max", "none"])
    p.add_argument("--strategy", action="append", choices=sorted(ANSWER_STRATEGIES),
                   help="Answer strategy (repeatable; default: default)")
    p.add_argument("--repeats", type=int, default=3, help="Answers per strategy")
    p.add_argument("--layout", choices=sorted(LAYOUTS), default="general")
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--max-requests", type=int)
    p.add_argument("--max-cost-usd", type=float)
    p.add_argument("--out", default="runs")
    p.add_argument("--run-name")
    p.add_argument("--log-prompts", action="store_true")
    p.set_defaults(func=cmd_replay_answer)

    p = sub.add_parser("judge-memory", help="Grade final memories against ground-truth preference attributes")
    p.add_argument("sources", nargs="+", help="Finished items: runs/<run>/items/<item> or examples/<run>")
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--effort", default="high", choices=["minimal", "low", "medium", "high", "xhigh", "max", "none"])
    p.add_argument("--repeats", type=int, default=3, help="Judgings per memory; the majority grade is reported")
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--max-cost-usd", type=float)
    p.add_argument("--out", help="Directory for judgements (default: memory_judgement.json in each item)")
    p.add_argument("--vet", help="A vet-items report: also score only the changes the conversations support")
    p.set_defaults(func=cmd_judge_memory)

    p = sub.add_parser("respond-eval", help="Write replies from saved memory under several conditions; judge them")
    p.add_argument("sources", nargs="+", help="Finished items: runs/<run>/items/<item> or examples/<run>")
    p.add_argument("--condition", action="append", choices=["memory", "full-history", "none"],
                   help="Repeatable; default: memory and none. 'memory' is the source's own memory "
                        "(the agent's workspace or a summary)")
    p.add_argument("--repeats", type=int, default=1, help="Replies per condition")
    p.add_argument("--judge-repeats", type=int, default=3, help="Judgings per reply; the majority fit is kept")
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--effort", default="medium", choices=["minimal", "low", "medium", "high", "xhigh", "max", "none"])
    p.add_argument("--judge-effort", default="high", choices=["minimal", "low", "medium", "high", "xhigh", "max", "none"])
    p.add_argument("--layout", choices=sorted(LAYOUTS), default="general")
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--max-cost-usd", type=float)
    p.add_argument("--out", default="runs")
    p.add_argument("--run-name")
    p.set_defaults(func=cmd_respond_eval)

    p = sub.add_parser("evaluate", help="Grade finished runs' memory and replies against reference checklists")
    p.add_argument("sources", nargs="+", help="Finished items: runs/<run>/items/<item>")
    p.add_argument("--checklists", nargs="+", required=True, help="JSON files mapping item id to its checklist")
    p.add_argument("--repeats", type=int, default=3, help="Judgings per memory and per reply; majority kept")
    p.add_argument("--no-reply", action="store_true", help="Grade memory only")
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--effort", default="medium", choices=["minimal", "low", "medium", "high", "xhigh", "max", "none"],
                   help="Effort for writing the reply")
    p.add_argument("--judge-effort", default="high", choices=["minimal", "low", "medium", "high", "xhigh", "max", "none"])
    p.add_argument("--layout", choices=sorted(LAYOUTS), default="general")
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--max-cost-usd", type=float)
    p.add_argument("--out", default="runs")
    p.add_argument("--run-name")
    p.set_defaults(func=cmd_evaluate)

    def bench_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--methods", default="profile,summary", help="Comma-separated: profile, summary")
        p.add_argument("--model", default="gpt-5.4-mini")
        p.add_argument("--effort", default="medium", choices=["minimal", "low", "medium", "high", "xhigh", "max", "none"])
        p.add_argument("--max-tokens", type=int, default=16000)
        p.add_argument("--timeout", type=float, default=300.0)
        p.add_argument("--max-cost-usd", type=float, help="One budget for memory, answers and judging (not DynamicMem's own evaluator)")
        p.add_argument("--out", default="runs")
        p.add_argument("--run-name")

    p = sub.add_parser("memconflict", help="MemConflict: right memory when memories disagree (local data clone)")
    p.add_argument("--data", default="data/external/MemConflict/Data/Step4_4.jsonl")
    p.add_argument("--people", help="Comma-separated person indices (0-29)")
    p.add_argument("--max-people", type=int, default=1)
    p.add_argument("--questions", type=int, default=12, help="Questions per person, split evenly by conflict type")
    p.add_argument("--max-sessions", type=int, help="Replay only the first N sessions (smoke tests)")
    p.add_argument("--judge-model", help="Judge model (default gpt-5.4-mini, kept fixed so designs and models "
                                          "are graded alike)")
    p.add_argument("--judge-effort", default="high", choices=["minimal", "low", "medium", "high", "xhigh", "max", "none"])
    bench_args(p)
    p.set_defaults(func=cmd_memconflict)

    p = sub.add_parser("dynamicmem", help="DynamicMem: user state over months of app logs (local data download)")
    p.add_argument("--data", default="data/external/dynamicmem_data")
    p.add_argument("--users", help="Comma-separated user folders, e.g. 001_user_001")
    p.add_argument("--max-users", type=int, default=1)
    p.add_argument("--max-checkpoints", type=int, help="Stop after N of the 5 checkpoints")
    p.add_argument("--score", action=argparse.BooleanOptionalAction, default=True,
                   help="Score with DynamicMem's own evaluator (needs --repo)")
    p.add_argument("--repo", default="data/external/DynamicMem", help="Local clone of the DynamicMem repository")
    p.add_argument("--judge-model", default="gpt-5.4-mini", help="Judge for DynamicMem's evaluator (paper: gpt-5.4); none = its rule-based scoring only")
    bench_args(p)
    p.set_defaults(func=cmd_dynamicmem)

    p = sub.add_parser("vet-items", help="Check that each labeled preference change shows up in the sessions")
    _add_dataset_args(p)
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--effort", default="medium", choices=["minimal", "low", "medium", "high", "xhigh", "max", "none"])
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--max-cost-usd", type=float)
    p.add_argument("--out", default="data/vet_items.json")
    p.set_defaults(func=cmd_vet_items)

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
