"""Sequential runner: replay events, ask the question, then score.

Leakage boundary: the method receives events one at a time, then the probe and
question. GoldLabel is read only after `method.answer` has returned.

Checkpoints: after every completed event the item's progress (workspace, method
state, stats, memory-check history) is saved, so a run that hits a daily quota
pauses cleanly and `--resume` continues from the last completed event.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .checks import EventInfo, MemoryMonitor
from .dataset import item_slug
from .entries import parse_workspace
from .evaluation import aggregate, evidence_chain, evolution_diagnostics, reference_heuristics, score, utilization
from .hb_compat import HB_COMMIT
from .methods import MethodConfig, MethodContext, make_method
from .models.base import ModelAdapter, ModelSettings, PauseRun
from .models.pricing import cost_usd
from .prompts import load_contract
from .tracing import RawSessionLog, TracedModel, TraceStats, TraceWriter
from .types import AnswerResult, BenchmarkItem

CHECKPOINT = "checkpoint.json"
# Settings that change how requests are sent but not what the model produces; a paused
# run may resume with different values for these.
OPERATIONAL_SETTINGS = {"max_retries", "timeout_s", "requests_per_minute"}


def _comparable(config: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in config.items() if k != "dataset"}
    out["model_settings"] = {k: v for k, v in out["model_settings"].items() if k not in OPERATIONAL_SETTINGS}
    return out


@dataclass(frozen=True)
class RunConfig:
    method: str
    settings: ModelSettings
    method_config: MethodConfig = field(default_factory=MethodConfig)
    dataset: dict[str, Any] = field(default_factory=dict)
    log_prompts: bool = False

    def to_dict(self) -> dict[str, Any]:
        mc = asdict(self.method_config)
        mc["layout"] = {"name": self.method_config.layout.name,
                        "files": {f.path: f.limit for f in self.method_config.layout.files}}
        return {
            "harness_version": __version__,
            "horizonbench_commit": HB_COMMIT,
            "method": self.method,
            "model_settings": asdict(self.settings),
            "method_config": mc,
            "contract_sha256": _contract_hash(self.method_config) if self.method == "agent" else None,
            "dataset": self.dataset,
            "log_prompts": self.log_prompts,
        }


def _contract_hash(mc: MethodConfig) -> str:
    import hashlib

    return hashlib.sha256(load_contract(mc.contract_path).encode()).hexdigest()



def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    os.replace(tmp, path)


def run_item(item: BenchmarkItem, model: ModelAdapter, item_dir: Path, cfg: RunConfig,
             resume: bool = False) -> dict[str, Any] | None:
    """Run one item. Returns its result, or None if it paused on quota."""
    checkpoint_path = item_dir / CHECKPOINT
    resuming = resume and checkpoint_path.exists()
    if not resuming:
        item_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = json.loads(checkpoint_path.read_text()) if resuming else None

    trace = TraceWriter(item_dir / "trace.jsonl", log_prompts=cfg.log_prompts)
    if resuming:
        trace.stats = TraceStats.from_dict(checkpoint["stats"], checkpoint["access"])
        raw_log = RawSessionLog.resume(item_dir / "raw_sessions.jsonl", checkpoint["next_event"])
    else:
        raw_log = RawSessionLog(item_dir / "raw_sessions.jsonl")
    stats = trace.stats
    episode = item.episode
    method = make_method(cfg.method, MethodContext(TracedModel(model, trace), trace, item_dir, cfg.method_config, resume=resuming))
    monitor = MemoryMonitor(cfg.method_config.layout) if method.has_workspace else None

    if resuming:
        if method.has_workspace:
            method.ws.restore_snapshot(checkpoint["workspace"])
            monitor.restore(checkpoint["monitor"], checkpoint["workspace"])
        method.restore_state(checkpoint["method_state"], episode.sessions)
        session_log, start, elapsed_before = checkpoint["session_log"], checkpoint["next_event"], checkpoint["elapsed_s"]
        trace.emit("item_resumed", next_event=start, events_total=len(episode.sessions))
    else:
        session_log, start, elapsed_before = [], 0, 0.0
        trace.emit("item_start", item_key=episode.item_key, method=cfg.method, model=cfg.settings.model,
                   n_sessions=len(episode.sessions), ordering=asdict(item.ordering))

    started = time.perf_counter()

    def save_checkpoint(next_event: int) -> None:
        _write_json(checkpoint_path, {
            "next_event": next_event,
            "stats": stats.to_dict(),
            "access": stats.access,
            "session_log": session_log,
            "method_state": method.export_state(),
            "workspace": method.ws.snapshot() if method.has_workspace else None,
            "monitor": monitor.state() if monitor else None,
            "elapsed_s": elapsed_before + time.perf_counter() - started,
        })

    answer, error, paused = AnswerResult(None, "none"), None, False
    try:
        # ---------------- agent side: observations only, in replay order ----------------
        for session in episode.sessions[start:]:
            digest = raw_log.append(session)
            before = (stats.file_ops, stats.model_calls, stats.usage.total_input_tokens, stats.usage.output_tokens, len(stats.access))
            trace.emit("event_start", event_id=session.event_id, source_index=session.source_index, date=session.date,
                       n_turns=len(session.turns), chars=sum(len(t.content) for t in session.turns), sha256=digest)
            t0 = time.perf_counter()
            method.observe(session)
            entry = {
                "index": session.index,
                "event_id": session.event_id,
                "date": session.date,
                "file_ops": stats.file_ops - before[0],
                "model_calls": stats.model_calls - before[1],
                "input_tokens": stats.usage.total_input_tokens - before[2],
                "output_tokens": stats.usage.output_tokens - before[3],
                "latency_s": round(time.perf_counter() - t0, 3),
            }
            if monitor:
                record = monitor.after_event(EventInfo(session.index, session.event_id, session.date[:10]),
                                             method.ws.snapshot(), stats.access[before[4]:])
                entry["violations"] = sum(record["violations"].values())  # open after this event
                entry["new_violations"] = sum(record["new_violations"].values())  # first seen this event
                trace.emit("memory_check", **{k: record[k] for k in ("event_id", "access", "violations",
                                                                     "new_violations", "metrics")})
            session_log.append(entry)
            trace.emit("event_end", **{k: v for k, v in entry.items() if k != "index"})
            save_checkpoint(session.index + 1)
        trace.emit("question_start", probe_date=episode.probe.date, probe_turns=len(episode.probe.turns))
        answer = method.answer(episode.probe, episode.question, f"S-{len(episode.sessions)}")
        trace.emit("answer", letter=answer.letter, via=answer.via, details=answer.details)
    except PauseRun as e:  # quota, request cap, or provider outage: stop at the last completed event
        paused = True
        trace.emit("item_paused", reason=str(e), resume_from=len(session_log))
    except Exception as e:  # record and continue with the next item
        error = f"{type(e).__name__}: {e}"
        trace.emit("item_error", error=error, traceback=traceback.format_exc())
    finally:
        raw_log.close()
    if paused:
        trace.close()
        return None
    latency = elapsed_before + time.perf_counter() - started

    # ---------------- evaluator side: labels are read only below ----------------
    snapshot = method.memory_snapshot()
    for path, content in snapshot.items():
        target = item_dir / "snapshots" / "final" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    memory_summary = chain = used = None
    if monitor:
        report = monitor.report()
        (item_dir / "memory_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
        memory_summary = {k: report[k] for k in ("events", "violations_by_code", "events_with_violations",
                                                  "events_with_new_violations", "unresolved_at_end",
                                                  "access_totals", "final_metrics")}
        parsed_final = parse_workspace(snapshot, cfg.method_config.layout)
        chain = evidence_chain(item.gold, session_log, parsed_final)
        used = utilization(answer.details.get("applied", []), parsed_final)
    scored = score(answer, item.gold)
    result = {
        "item_index": item.index,
        "item_id": item.gold.item_id,
        "item_key": episode.item_key,
        "method": cfg.method,
        "model": cfg.settings.model,
        **scored,
        "error": error,
        "gold": {
            "correct_letter": item.gold.correct_letter,
            "distractor_letter": item.gold.distractor_letter,
            "has_evolved": item.gold.has_evolved,
            "generator": item.gold.generator,
            "user_id": item.gold.user_id,
            "preference_domain": item.gold.preference_domain,
        },
        "ordering": asdict(item.ordering),
        "answer_details": answer.details,
        "utilization": used,
        "reference_heuristics": reference_heuristics(episode.question, item.gold),
        "latency_s": round(latency, 3),
        "stats": stats.to_dict(),
        "estimated_cost_usd": cost_usd(stats.usage, cfg.settings.model),
        "memory_chars": {p: len(c) for p, c in snapshot.items()},
        "memory": memory_summary,
        "evolution_diagnostics": evolution_diagnostics(item.gold, session_log) if method.has_workspace else None,
        "evidence_chain": chain,
        "sessions": session_log,
    }
    trace.emit("evaluation", **scored, correct_letter=item.gold.correct_letter, error=error)
    trace.close()
    (item_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    if error is None:  # keep the checkpoint after a failure so the progress is not lost
        checkpoint_path.unlink(missing_ok=True)
    return result


def run(items: list[BenchmarkItem], model: ModelAdapter, run_dir: Path, cfg: RunConfig, resume: bool = False,
        progress=print) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items_dir = run_dir / "items"
    config_path = run_dir / "run_config.json"
    if items_dir.exists() and any(items_dir.iterdir()) and not resume:
        raise FileExistsError(f"{run_dir} already has results; pass --resume or choose another --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = cfg.to_dict()
    if resume and config_path.exists():
        previous = json.loads(config_path.read_text())
        if _comparable(previous) != _comparable(config):
            raise ValueError(f"--resume with different settings than {config_path}")
    config_path.write_text(json.dumps(config, indent=2))

    results, paused_at = [], None
    for n, item in enumerate(items, 1):
        item_dir = items_dir / item_slug(item)
        if resume and (item_dir / "result.json").exists():
            results.append(json.loads((item_dir / "result.json").read_text()))
            progress(f"[{n}/{len(items)}] {item_dir.name}: cached")
            continue
        can_resume = resume and (item_dir / CHECKPOINT).exists()
        if item_dir.exists() and not can_resume:  # partial item without a checkpoint
            _remove_tree(item_dir)
        verb = "resuming" if can_resume else "replaying"
        progress(f"[{n}/{len(items)}] {item_dir.name}: {verb} {len(item.episode.sessions)} events")
        result = run_item(item, model, item_dir, cfg, resume=can_resume)
        if result is None:
            done = json.loads((item_dir / CHECKPOINT).read_text())["next_event"] if (item_dir / CHECKPOINT).exists() else 0
            paused_at = f"{item_dir.name} after {done} of {len(item.episode.sessions)} events"
            progress(f"[{n}/{len(items)}] {item_dir.name}: PAUSED (quota, request cap, or provider outage) after {done} events; rerun with --resume")
            break
        results.append(result)
        status = "ERROR " + result["error"] if result["error"] else ("correct" if result["correct"] else "wrong")
        progress(
            f"[{n}/{len(items)}] {item_dir.name}: predicted {result['predicted_letter'] or '-'} "
            f"(gold {result['gold']['correct_letter']}) {status}; {result['stats']['model_calls']} calls, "
            f"{result['stats']['file_ops']} file ops, {result['latency_s']:.1f}s"
        )

    with (run_dir / "results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = aggregate(results)
    summary["paused_at"] = paused_at
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return results, summary


def _remove_tree(path: Path) -> None:
    import shutil
    import stat

    def make_writable(func, p, _exc):
        Path(p).chmod(stat.S_IWUSR | stat.S_IRUSR)
        func(p)

    shutil.rmtree(path, onexc=make_writable)
