"""Sequential runner: replay sessions, ask the question, then score.

Leakage boundary: the method receives sessions one at a time, then the probe and
question. GoldLabel is read only after `method.answer` has returned.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .dataset import item_slug
from .evaluation import aggregate, evolution_diagnostics, reference_heuristics, score
from .hb_compat import HB_COMMIT
from .methods import MethodConfig, MethodContext, make_method
from .models.base import ModelAdapter, ModelSettings
from .tracing import RawSessionLog, TracedModel, TraceWriter
from .types import AnswerResult, BenchmarkItem


@dataclass(frozen=True)
class RunConfig:
    method: str
    settings: ModelSettings
    method_config: MethodConfig = field(default_factory=MethodConfig)
    dataset: dict[str, Any] = field(default_factory=dict)
    log_prompts: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "harness_version": __version__,
            "horizonbench_commit": HB_COMMIT,
            "method": self.method,
            "model_settings": asdict(self.settings),
            "method_config": asdict(self.method_config),
            "dataset": self.dataset,
            "log_prompts": self.log_prompts,
        }


def run_item(item: BenchmarkItem, model: ModelAdapter, item_dir: Path, cfg: RunConfig) -> dict[str, Any]:
    item_dir.mkdir(parents=True, exist_ok=False)
    trace = TraceWriter(item_dir / "trace.jsonl", log_prompts=cfg.log_prompts)
    raw_log = RawSessionLog(item_dir / "raw_sessions.jsonl")
    stats = trace.stats
    episode = item.episode
    method = make_method(cfg.method, MethodContext(TracedModel(model, trace), trace, item_dir, cfg.method_config))
    trace.emit(
        "item_start",
        item_key=episode.item_key,
        method=cfg.method,
        model=cfg.settings.model,
        n_sessions=len(episode.sessions),
        ordering=asdict(item.ordering),
    )

    start = time.perf_counter()
    session_log: list[dict[str, Any]] = []
    answer, error = AnswerResult(None, "none"), None
    try:
        # ---------------- agent side: observations only, in replay order ----------------
        for session in episode.sessions:
            digest = raw_log.append(session)
            before = (stats.file_ops, stats.model_calls, stats.usage.total_input_tokens, stats.usage.output_tokens)
            trace.emit(
                "session_start",
                session=session.index,
                source_index=session.source_index,
                date=session.date,
                n_turns=len(session.turns),
                chars=sum(len(t.content) for t in session.turns),
                sha256=digest,
            )
            t0 = time.perf_counter()
            method.observe_session(session)
            entry = {
                "index": session.index,
                "date": session.date,
                "file_ops": stats.file_ops - before[0],
                "model_calls": stats.model_calls - before[1],
                "input_tokens": stats.usage.total_input_tokens - before[2],
                "output_tokens": stats.usage.output_tokens - before[3],
                "latency_s": round(time.perf_counter() - t0, 3),
            }
            session_log.append(entry)
            trace.emit("session_end", session=session.index, **{k: v for k, v in entry.items() if k != "index"})
        trace.emit("question_start", probe_date=episode.probe.date, probe_turns=len(episode.probe.turns))
        answer = method.answer(episode.probe, episode.question)
        trace.emit("answer", letter=answer.letter, via=answer.via, details=answer.details)
    except Exception as e:  # record and continue with the next item
        error = f"{type(e).__name__}: {e}"
        trace.emit("item_error", error=error, traceback=traceback.format_exc())
    finally:
        raw_log.close()
    latency = time.perf_counter() - start

    # ---------------- evaluator side: labels are read only below ----------------
    memory = method.memory_snapshot()
    for path, content in memory.items():
        target = item_dir / "snapshots" / "final" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    scored = score(answer, item.gold)
    reference = reference_heuristics(episode.question, item.gold)
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
        "reference_heuristics": reference,
        "latency_s": round(latency, 3),
        "stats": stats.to_dict(),
        "memory_chars": {p: len(c) for p, c in memory.items()},
        "evolution_diagnostics": evolution_diagnostics(item.gold, session_log) if method.has_workspace else None,
        "sessions": session_log,
    }
    trace.emit("evaluation", **scored, correct_letter=item.gold.correct_letter, error=error)
    trace.close()
    (item_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def run(
    items: list[BenchmarkItem],
    model: ModelAdapter,
    run_dir: Path,
    cfg: RunConfig,
    resume: bool = False,
    progress=print,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items_dir = run_dir / "items"
    config_path = run_dir / "run_config.json"
    if items_dir.exists() and any(items_dir.iterdir()) and not resume:
        raise FileExistsError(f"{run_dir} already has results; pass --resume or choose another --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = cfg.to_dict()
    if resume and config_path.exists():
        previous = json.loads(config_path.read_text())
        if {k: v for k, v in previous.items() if k != "dataset"} != {k: v for k, v in config.items() if k != "dataset"}:
            raise ValueError(f"--resume with different settings than {config_path}")
    config_path.write_text(json.dumps(config, indent=2))

    results = []
    for n, item in enumerate(items, 1):
        item_dir = items_dir / item_slug(item)
        result_path = item_dir / "result.json"
        if resume and result_path.exists():
            results.append(json.loads(result_path.read_text()))
            progress(f"[{n}/{len(items)}] {item_dir.name}: cached")
            continue
        if item_dir.exists():  # partial item from an interrupted run
            _remove_tree(item_dir)
        progress(f"[{n}/{len(items)}] {item_dir.name}: replaying {len(item.episode.sessions)} sessions")
        result = run_item(item, model, item_dir, cfg)
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
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return results, summary


def _remove_tree(path: Path) -> None:
    import shutil
    import stat

    def make_writable(func, p, _exc):
        Path(p).chmod(stat.S_IWUSR | stat.S_IRUSR)
        func(p)

    shutil.rmtree(path, onexc=make_writable)
