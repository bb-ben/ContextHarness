"""Load HorizonBench rows and split each into an agent-visible Episode and an
evaluator-only GoldLabel. See docs/DATASET_NOTES.md for the schema findings that
motivate the ordering policies."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Literal

from .hb_compat import QUESTION_STEM, parse_sessions
from .types import (
    BenchmarkItem,
    Episode,
    GoldLabel,
    Option,
    OrderingReport,
    Probe,
    Question,
    Session,
    Turn,
)

DATASET_ID = "stellalisy/HorizonBench"
OrderPolicy = Literal["chronological", "text"]


# --------------------------------------------------------------------------- rows


def iter_rows(config: str = "sample", source: str | Path | None = None) -> Iterator[dict[str, Any]]:
    """Yield raw rows from a local file (.parquet/.jsonl/.json) or the HF dataset."""
    if source is not None:
        yield from _iter_local(Path(source))
        return
    for path in _hf_parquet_files(config):
        yield from _iter_parquet(path)


def _iter_local(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".parquet":
        yield from _iter_parquet(path)
    elif path.suffix == ".jsonl":
        with path.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == ".json":
        yield from json.loads(path.read_text())
    else:
        raise ValueError(f"Unsupported dataset file: {path}")


def _iter_parquet(path: Path) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
        yield from batch.to_pylist()


def _hf_parquet_files(config: str) -> Iterator[Path]:
    """Download (or reuse cached) parquet shards for a config, one at a time."""
    from huggingface_hub import HfApi, hf_hub_download

    try:
        names = sorted(
            f
            for f in HfApi().list_repo_files(DATASET_ID, repo_type="dataset")
            if f.startswith(f"{config}/") and f.endswith(".parquet")
        )
        offline = False
    except Exception:
        # Offline: fall back to the single known sample shard if it is cached.
        if config != "sample":
            raise
        names, offline = ["sample/test-00000-of-00001.parquet"], True
    if not names:
        raise ValueError(f"No parquet files for config {config!r} in {DATASET_ID}")
    for name in names:
        yield Path(
            hf_hub_download(DATASET_ID, name, repo_type="dataset", local_files_only=offline)
        )


# ---------------------------------------------------------------------- parsing


def parse_item(
    row: dict[str, Any],
    index: int,
    order: OrderPolicy = "chronological",
    show_scenarios: bool = True,
    max_sessions: int | None = None,
) -> BenchmarkItem:
    segments = parse_sessions(row["conversation"])
    if len(segments) < 2:
        raise ValueError(f"Item {row['id']!r}: expected history sessions plus a probe segment")
    *history, probe_seg = segments
    probe_dt = _parse_dt(probe_seg["date"])
    dated = [(i, seg, _parse_dt(seg["date"])) for i, seg in enumerate(history)]
    monotonic = all(a[2] <= b[2] for a, b in zip(dated, dated[1:]))

    if order == "chronological":
        kept = sorted((d for d in dated if d[2] <= probe_dt), key=lambda d: d[2])  # stable
        dropped = tuple(seg["date"] for _, seg, dt in dated if dt > probe_dt)
    elif order == "text":
        kept, dropped = list(dated), ()
    else:
        raise ValueError(f"Unknown order policy: {order!r}")

    truncated_to = None
    if max_sessions is not None and len(kept) > max_sessions:
        kept, truncated_to = kept[len(kept) - max_sessions :], max_sessions

    sessions = tuple(
        Session(
            index=k,
            source_index=i,
            date=seg["date"],
            scenario=seg["scenario"] if show_scenarios else "",
            turns=_turns(seg),
        )
        for k, (i, seg, _) in enumerate(kept)
    )
    text_ranked = sorted(s.source_index for s in sessions)
    n_moved = sum(1 for s, src in zip(sessions, text_ranked) if s.source_index != src)

    options = row["options"]
    if isinstance(options, str):
        options = json.loads(options)
    question = Question(
        stem=QUESTION_STEM,
        options=tuple(Option(letter=o["letter"], text=o["option"]) for o in options),
    )
    probe = Probe(
        date=probe_seg["date"],
        scenario=probe_seg["scenario"] if show_scenarios else "",
        turns=_turns(probe_seg),
    )
    episode = Episode(
        item_key=hashlib.sha256(row["id"].encode()).hexdigest()[:12],
        sessions=sessions,
        probe=probe,
        question=question,
    )

    evolution = row.get("preference_evolution")
    if isinstance(evolution, str):
        evolution = json.loads(evolution)
    gold = GoldLabel(
        item_id=row["id"],
        generator=row.get("generator", ""),
        user_id=row.get("user_id", ""),
        correct_letter=row["correct_letter"],
        distractor_letter=row.get("distractor_letter") or None,
        has_evolved=bool(row.get("has_evolved", False)),
        preference_domain=row.get("preference_domain", ""),
        option_values={o["letter"]: str(o.get("value", "")) for o in options},
        preference_evolution=evolution or None,
    )
    report = OrderingReport(
        policy=order,
        n_history_sessions_in_text=len(history),
        n_replayed=len(sessions),
        text_order_monotonic=monotonic,
        n_moved=n_moved,
        dropped_after_probe=dropped,
        truncated_to_last=truncated_to,
    )
    return BenchmarkItem(
        index=index,
        episode=episode,
        gold=gold,
        ordering=report,
        conversation_sha256=hashlib.sha256(row["conversation"].encode()).hexdigest(),
    )


def load_items(
    config: str = "sample",
    source: str | Path | None = None,
    max_items: int | None = None,
    item_indices: list[int] | None = None,
    **parse_kwargs: Any,
) -> list[BenchmarkItem]:
    wanted = set(item_indices) if item_indices else None
    items = []
    for i, row in enumerate(iter_rows(config, source)):
        if wanted is not None and i not in wanted:
            continue
        items.append(parse_item(row, index=i, **parse_kwargs))
        if max_items is not None and len(items) >= max_items:
            break
        if wanted is not None and len(items) == len(wanted):
            break
    return items


def item_slug(item: BenchmarkItem) -> str:
    """Human-browsable directory name that does not reveal the preference domain."""
    user = item.gold.user_id.replace("/", "_") or "user"
    return f"{item.index:03d}_{user}_{item.episode.item_key[:8]}"


def _turns(seg: dict[str, Any]) -> tuple[Turn, ...]:
    # The parser appends trailing blank lines to the last turn; strip them.
    return tuple(Turn(role=t["role"], content=t["content"].rstrip()) for t in seg["turns"])


def _parse_dt(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError as e:
        raise ValueError(f"Unparseable session date {value!r}") from e
