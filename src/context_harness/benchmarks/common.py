"""Shared pieces for benchmark adapters: a memory method outside the HorizonBench loop,
its memory as plain text, and single model calls that return a tool's arguments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..methods import MethodConfig, MethodContext, make_method
from ..models.base import ModelAdapter, ModelRequest, TextBlock, ToolSpec
from ..models.pricing import cost_usd
from ..tracing import TracedModel, TraceWriter


def open_method(name: str, model: ModelAdapter, item_dir: Path, config: MethodConfig | None = None):
    """A fresh memory method writing its trace and files under item_dir."""
    item_dir.mkdir(parents=True, exist_ok=True)
    trace = TraceWriter(item_dir / "trace.jsonl")
    ctx = MethodContext(TracedModel(model, trace), trace, item_dir, config or MethodConfig())
    return make_method(name, ctx), trace


def memory_text(method) -> str:
    """What the method would read when answering: the profile or the summary, not raw logs."""
    snap = method.memory_snapshot()
    for key in ("profile.md", "summary.md"):
        if key in snap:
            return snap[key]
    raise ValueError(f"benchmark adapters support the profile and summary methods, not {method.name!r}")


def call_tool(model: ModelAdapter, phase: str, system: str, text: str, spec: ToolSpec,
              retries: int = 2) -> tuple[dict[str, Any] | None, float, str]:
    """One request offering one tool; returns its arguments (or None), the cost, and any text."""
    request = ModelRequest(phase=phase, system=system, blocks=(TextBlock(text),), tools=(spec,))
    cost, text_out = 0.0, ""
    for _ in range(retries):
        response = model.complete(request)
        cost += cost_usd(response.usage, model.settings.model) or 0.0
        text_out = response.text
        call = next((c for c in response.tool_calls if c.name == spec.name), None)
        if call is not None:
            return call.arguments, cost, text_out
    return None, cost, text_out
