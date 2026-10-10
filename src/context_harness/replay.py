"""Answer-only replay: rerun just the answering step against a saved memory.

Replaying all events to test a change in how memory is *used* costs dozens of
calls; restoring a run's final memory and answering again costs one or two. Each
replay gets a fresh workspace copied from the snapshot, so the source run is never
modified, and labels are read only after the answer, as in a full run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .dataset import load_items
from .evaluation import score
from .methods.agent import WorkspaceAgent
from .methods.base import MethodContext
from .models.base import ModelAdapter
from .models.pricing import cost_usd
from .orchestrator import RunConfig
from .tracing import TracedModel, TraceWriter
from .types import BenchmarkItem
from .workspace import Workspace


def find_snapshot(source: Path) -> Path:
    for name in ("snapshots/final", "final_memory"):
        if (source / name).is_dir():
            return source / name
    raise FileNotFoundError(f"No saved final memory under {source} (expected snapshots/final or final_memory)")


def load_source_item(source: Path) -> BenchmarkItem:
    """The item a saved run answered, loaded with that run's own dataset settings."""
    result = json.loads((source / "result.json").read_text())
    config_path = next(p for p in (source / "run_config.json", source.parent.parent / "run_config.json") if p.exists())
    ds = json.loads(config_path.read_text())["dataset"]
    items = load_items(
        config=ds.get("config") or "sample",
        source=ds.get("source"),
        item_indices=[result["item_index"]],
        order=ds.get("order", "chronological"),
        show_scenarios=ds.get("show_scenarios", True),
        max_sessions=ds.get("max_sessions"),
    )
    if not items or items[0].gold.item_id != result["item_id"]:
        raise ValueError(f"Could not reload the item answered in {source}")
    return items[0]


def replay_answer(item: BenchmarkItem, snapshot: Path, model: ModelAdapter, cfg: RunConfig,
                  out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=False)
    layout = cfg.method_config.layout
    ws = Workspace.create(out_dir / "workspace", layout)
    ws.restore_snapshot({p: (snapshot / p).read_text() for p in layout.paths if (snapshot / p).exists()})
    trace = TraceWriter(out_dir / "trace.jsonl", log_prompts=cfg.log_prompts)
    trace.emit("replay_start", snapshot=str(snapshot), strategy=cfg.method_config.answer_strategy,
               model=cfg.settings.model)
    agent = WorkspaceAgent(MethodContext(TracedModel(model, trace), trace, out_dir, cfg.method_config, resume=True))
    episode = item.episode
    start = time.perf_counter()
    answer = agent.answer(episode.probe, episode.question, f"S-{len(episode.sessions)}")
    latency = time.perf_counter() - start
    # ---------------- evaluator side: labels are read only below ----------------
    scored = score(answer, item.gold)
    result = {
        "item_id": item.gold.item_id,
        "strategy": cfg.method_config.answer_strategy,
        "model": cfg.settings.model,
        **scored,
        "gold": {"correct_letter": item.gold.correct_letter, "distractor_letter": item.gold.distractor_letter},
        "answer_details": answer.details,
        "stats": trace.stats.to_dict(),
        "estimated_cost_usd": cost_usd(trace.stats.usage, cfg.settings.model),
        "latency_s": round(latency, 2),
    }
    trace.emit("evaluation", **scored, correct_letter=item.gold.correct_letter)
    trace.close()
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result
