"""Per-item JSONL trace, model-call accounting, and the immutable raw session log."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .models.base import ModelAdapter, ModelRequest, ModelResponse, Usage
from .types import Session


@dataclass
class TraceStats:
    model_calls: int = 0
    usage: Usage = field(default_factory=Usage)
    usage_by_phase: dict[str, Usage] = field(default_factory=lambda: defaultdict(Usage))
    calls_by_phase: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    served_by: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    tool_calls: int = 0
    tool_errors: int = 0
    file_ops: int = 0  # successful workspace mutations
    file_reads: int = 0
    refusals: int = 0
    access: list[dict[str, Any]] = field(default_factory=list)  # per tool call: tool, kind, paths, ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_calls": self.model_calls,
            "usage": asdict(self.usage),
            "usage_by_phase": {k: asdict(v) for k, v in self.usage_by_phase.items()},
            "calls_by_phase": dict(self.calls_by_phase),
            "served_by": dict(self.served_by),
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "file_ops": self.file_ops,
            "file_reads": self.file_reads,
            "refusals": self.refusals,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], access: list[dict[str, Any]] | None = None) -> "TraceStats":
        stats = cls(
            model_calls=d["model_calls"], usage=Usage(**d["usage"]), tool_calls=d["tool_calls"],
            tool_errors=d["tool_errors"], file_ops=d["file_ops"], file_reads=d["file_reads"],
            refusals=d["refusals"], access=list(access or []),
        )
        for phase, u in d["usage_by_phase"].items():
            stats.usage_by_phase[phase] = Usage(**u)
        stats.calls_by_phase.update(d["calls_by_phase"])
        stats.served_by.update(d["served_by"])
        return stats


class TraceWriter:
    def __init__(self, path: Path, log_prompts: bool = False):
        self.path = path
        self.log_prompts = log_prompts
        self.stats = TraceStats()
        self._seq = 0
        self._t0 = time.perf_counter()
        self._file = path.open("a")

    def emit(self, event: str, /, **fields: Any) -> None:
        if "event" in fields:
            raise ValueError("'event' is the record type; pass the event's ID as event_id")
        record = {"seq": self._seq, "t": round(time.perf_counter() - self._t0, 4), "event": event}
        record.update(fields)
        self._file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._file.flush()
        self._seq += 1

    def close(self) -> None:
        self._file.close()


class TracedModel:
    """Wraps any ModelAdapter: logs every call and accumulates token usage."""

    def __init__(self, inner: ModelAdapter, trace: TraceWriter):
        self.inner = inner
        self.settings = inner.settings
        self.trace = trace

    def complete(self, request: ModelRequest) -> ModelResponse:
        prompt = request.prompt_text()
        base = {
            "phase": request.phase,
            **request.metadata,
            "request_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "request_chars": len(prompt),
        }
        if self.trace.log_prompts:
            base["request"] = {
                "system": request.system,
                "blocks": [b.text for b in request.blocks],
                "tools": [t.name for t in request.tools],
            }
        try:
            response = self.inner.complete(request)
        except Exception as e:
            self.trace.emit("model_error", **base, error=f"{type(e).__name__}: {e}")
            raise
        stats = self.trace.stats
        stats.model_calls += 1
        stats.usage += response.usage
        stats.usage_by_phase[request.phase] += response.usage
        stats.calls_by_phase[request.phase] += 1
        stats.served_by[response.model] += 1
        if response.stop_reason == "refusal":
            stats.refusals += 1
        self.trace.emit(
            "model_call",
            **base,
            model=response.model,
            stop_reason=response.stop_reason,
            usage=asdict(response.usage),
            latency_s=round(response.latency_s, 3),
            request_id=response.request_id,
            response_text=response.text,
            tool_calls=[{"id": c.id, "name": c.name, "arguments": c.arguments} for c in response.tool_calls],
            **({"extra": response.extra} if response.extra else {}),
        )
        return response


class RawSessionLog:
    """Append-only, hash-chained log of sessions exactly as observed.

    Lives outside the workspace; agent tools cannot address it. Made read-only
    when the item finishes.
    """

    def __init__(self, path: Path):
        self.path = path
        self._prev = "0" * 64
        self._file = path.open("x")  # never overwrite an existing log

    @classmethod
    def resume(cls, path: Path, keep: int) -> "RawSessionLog":
        """Reopen a paused item's log, keeping its first `keep` records.

        Records past `keep` belong to an event that was interrupted and will be
        replayed, so they are dropped. The kept prefix must verify.
        """
        lines = path.read_text().splitlines()[:keep]
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        path.write_text("".join(line + "\n" for line in lines))
        if not cls.verify(path):
            raise ValueError(f"{path} failed hash-chain verification; cannot resume")
        log = cls.__new__(cls)
        log.path = path
        log._prev = json.loads(lines[-1])["sha256"] if lines else "0" * 64
        log._file = path.open("a")
        return log

    def append(self, session: Session) -> str:
        body = {
            "index": session.index,
            "source_index": session.source_index,
            "date": session.date,
            "scenario": session.scenario,
            "turns": [{"role": t.role, "content": t.content} for t in session.turns],
        }
        payload = json.dumps(body, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256((self._prev + payload).encode()).hexdigest()
        self._file.write(json.dumps({**body, "prev_sha256": self._prev, "sha256": digest}, ensure_ascii=False) + "\n")
        self._file.flush()
        self._prev = digest
        return digest

    def close(self) -> None:
        self._file.close()
        os.chmod(self.path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    @staticmethod
    def verify(path: Path) -> bool:
        prev = "0" * 64
        for line in path.read_text().splitlines():
            record = json.loads(line)
            digest, claimed_prev = record.pop("sha256"), record.pop("prev_sha256")
            payload = json.dumps(record, ensure_ascii=False, sort_keys=True)
            if claimed_prev != prev or hashlib.sha256((prev + payload).encode()).hexdigest() != digest:
                return False
            prev = digest
        return True
