"""Generated-response evaluation: does memory make the reply fit the person now?

A multiple-choice answer only says which of five fixed replies a model prefers. What
the layer is for is the reply the agent actually writes. Here the agent writes a reply
to the item's final conversation, and an evaluator-side judge grades it against the
ground-truth attributes: for each attribute that changed, does the reply fit the new
value, the old one, or neither?

The same reply is written under several memory conditions, so the effect of memory
shows directly:
- `memory`: the agent's own saved workspace, used through the contract (agent.md as
  the system prompt, `<working_context>`, `<state>`, `<preference_dimensions>`, lookups);
- `summary`: a rolling-summary baseline's saved summary, as plain text;
- `full-history`: every session, as plain text;
- `none`: no memory at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import prompts
from .judge import ground_truth
from .methods.agent import WorkspaceAgent
from .methods.base import MethodContext
from .methods.profile import read_profile_memory
from .models.base import ModelAdapter, ModelRequest, TextBlock, ToolSpec
from .models.pricing import cost_usd
from .orchestrator import RunConfig
from .tracing import TracedModel, TraceWriter
from .types import BenchmarkItem, GoldLabel
from .workspace import Workspace

FITS = ("current", "original", "neutral", "other")
CONDITIONS = ("memory", "summary", "full-history", "none")

PLAIN_SYSTEM = """\
You are an assistant with a long-term relationship with one person. Reply to their
last message in <current_conversation>. {memory_note} Write the reply you would
actually send, shaped by who they are now, and submit it with the respond tool."""

RESPONSE_JUDGE_SYSTEM = """\
You grade a reply that an assistant wrote to a person, against ground truth about how
that person wants to be helped now. Some attributes changed after a life event; for
those you also know the value they used to have.

For each attribute, report `fit`:
- `current`: the reply clearly follows the current value;
- `original`: the reply clearly follows the old value (changed attributes only);
- `neutral`: the reply gives no real signal either way on this attribute;
- `other`: the reply goes against the current value in some other way.
Judge the reply's form and approach, not whether its content is correct. Read each
attribute by its meaning: the ground truth uses a coarse taxonomy. Give one short
reason per attribute, then submit everything with one submit_grades call."""


def _respond_spec() -> ToolSpec:
    return ToolSpec("respond", "Submit the reply you would send the person.",
                    {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]})


def snapshot_kind(snapshot: Path) -> str:
    if (snapshot / "context.md").exists():
        return "agent"
    if (snapshot / "summary.md").exists():
        return "summary"
    if (snapshot / "profile.md").exists():
        return "profile"
    return "unknown"


def respond_with_agent(item: BenchmarkItem, snapshot: Path, model: ModelAdapter, cfg: RunConfig,
                       out_dir: Path) -> dict[str, Any]:
    """The agent's own reply, using its saved memory exactly as it would at answer time."""
    out_dir.mkdir(parents=True, exist_ok=False)
    layout = cfg.method_config.layout
    ws = Workspace.create(out_dir / "workspace", layout)
    ws.restore_snapshot({p: (snapshot / p).read_text() for p in layout.paths if (snapshot / p).exists()})
    trace = TraceWriter(out_dir / "trace.jsonl", log_prompts=cfg.log_prompts)
    agent = WorkspaceAgent(MethodContext(TracedModel(model, trace), trace, out_dir, cfg.method_config, resume=True))
    reply = agent.respond(item.episode.probe, f"S-{len(item.episode.sessions)}")
    trace.close()
    return {**reply, "usage": trace.stats.usage}


def respond_with_text(item: BenchmarkItem, memory: str | None, label: str, model: ModelAdapter,
                      out_dir: Path) -> dict[str, Any]:
    """A reply from plain-text memory (a summary or the full history) or from none."""
    out_dir.mkdir(parents=True, exist_ok=False)
    note = (f"<memory> holds what you know about them from earlier ({label})." if memory
            else "You know nothing about them beyond this conversation.")
    blocks = ([TextBlock(f"<memory source=\"{label}\">\n{memory}\n</memory>", cache=True)] if memory else []) + [
        TextBlock(prompts.render_probe(item.episode.probe, f"S-{len(item.episode.sessions)}"))]
    trace = TraceWriter(out_dir / "trace.jsonl")
    traced = TracedModel(model, trace)
    request = ModelRequest(phase="respond", system=PLAIN_SYSTEM.format(memory_note=note), blocks=tuple(blocks),
                           tools=(_respond_spec(),))
    text = None
    for _ in range(2):
        response = traced.complete(request)
        call = next((c for c in response.tool_calls if c.name == "respond"), None)
        text = call.arguments.get("text") if call else (response.text or None)
        if text:
            break
    trace.close()
    return {"text": text, "applied": [], "usage": trace.stats.usage}


def reply_from_memory(item: BenchmarkItem, snapshot: Path, model: ModelAdapter, cfg: RunConfig,
                      out_dir: Path) -> dict[str, Any]:
    """The reply each memory design would give, using that design's own memory: the agent
    through its workspace and contract, a summary or profile as plain text."""
    kind = snapshot_kind(snapshot)
    if kind == "agent":
        return respond_with_agent(item, snapshot, model, cfg, out_dir)
    if kind == "summary":
        return respond_with_text(item, (snapshot / "summary.md").read_text(), "a rolling summary", model, out_dir)
    if kind == "profile":
        return respond_with_text(item, read_profile_memory(snapshot),
                                 "a profile of who they are now, with evidence", model, out_dir)
    raise ValueError(f"No memory to reply from in {snapshot}")


def full_history_text(item: BenchmarkItem) -> str:
    return "\n\n".join(prompts.render_session(s) for s in item.episode.sessions)


def _grade_spec(attributes: list[str]) -> ToolSpec:
    grade = {"type": "object", "properties": {
        "attribute": {"type": "string", "enum": attributes},
        "fit": {"type": "string", "enum": list(FITS)},
        "reason": {"type": "string"}}, "required": ["attribute", "fit", "reason"]}
    return ToolSpec("submit_grades", "Submit one grade per attribute.",
                    {"type": "object", "properties": {"grades": {"type": "array", "items": grade}},
                     "required": ["grades"]})


def render_response_judge_input(truth: dict[str, Any], item: BenchmarkItem, reply: str) -> str:
    lines = [f"<ground_truth domain=\"{truth['preference_domain']}\">"]
    for a in truth["attributes"]:
        lines.append(f"- {a['attribute']}: now {a['current']}" + (f" (used to be {a['original']})" if a["changed"] else ""))
    lines.append("</ground_truth>")
    convo = "\n".join(f"{t.role}: {t.content.strip()[:1500]}" for t in item.episode.probe.turns[-3:])
    return "\n".join(lines) + f"\n\n<conversation_end>\n{convo}\n</conversation_end>\n\n<reply>\n{reply}\n</reply>"


def judge_response(model: ModelAdapter, gold: GoldLabel, item: BenchmarkItem, reply: str,
                   repeats: int = 3) -> dict[str, Any]:
    """Majority fit per attribute over `repeats` judgings; ties go to the less favorable fit."""
    truth = ground_truth(gold)
    attrs = [a["attribute"] for a in truth["attributes"]]
    request = ModelRequest(phase="judge_response", system=RESPONSE_JUDGE_SYSTEM,
                           blocks=(TextBlock(render_response_judge_input(truth, item, reply)),),
                           tools=(_grade_spec(attrs),))
    votes: dict[str, list[str]] = {a: [] for a in attrs}
    reasons: dict[str, str] = {}
    cost = 0.0
    for _ in range(repeats):
        response = model.complete(request)
        cost += cost_usd(response.usage, model.settings.model) or 0.0
        call = next((c for c in response.tool_calls if c.name == "submit_grades"), None)
        for g in (call.arguments.get("grades", []) if call else []):
            if g.get("attribute") in votes and g.get("fit") in FITS:
                votes[g["attribute"]].append(g["fit"])
                reasons.setdefault(g["attribute"], g.get("reason", ""))
    order = {"other": 0, "original": 1, "neutral": 2, "current": 3}  # tie-break toward the worse fit
    rows = []
    for a in truth["attributes"]:
        v = votes[a["attribute"]] or ["neutral"]
        fit = max(set(v), key=lambda f: (v.count(f), -order[f]))
        rows.append({**a, "fit": fit, "votes": v, "reason": reasons.get(a["attribute"], "")})
    changed = [r for r in rows if r["changed"]]
    share = lambda rs, f: round(sum(r["fit"] == f for r in rs) / len(rs), 3) if rs else None  # noqa: E731
    return {
        "attributes": rows,
        "changed_fit_current": share(changed, "current"),
        "changed_fit_original": share(changed, "original"),
        "all_fit_current": share(rows, "current"),
        "judge_cost_usd": round(cost, 5),
    }


def write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
