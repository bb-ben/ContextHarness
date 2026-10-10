"""Memory judge: grade what a finished run's memory believes against ground truth.

Evaluator side only. The judge sees the benchmark's ground-truth preference
attributes, which no method ever sees, and it runs after the item is finished.

A multiple-choice answer is one noisy bit that hinges on the wording of five
options. HorizonBench also records the person's preference as a few attributes,
with their values before and after each life event. The judge reads the memory
(current state, superseded entries and hypotheses; not raw episodes) and grades,
per attribute, whether memory holds the current value, still asserts the old one,
and recorded the change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models.base import ModelAdapter, ModelRequest, TextBlock, ToolSpec
from .models.pricing import cost_usd
from .types import GoldLabel

MATCHES = ("matches", "partial", "absent", "contradicts")
MATCH_SCORE = {"matches": 1.0, "partial": 0.5, "absent": 0.0, "contradicts": 0.0}
_BELIEF_TYPES = ("F-", "P-", "G-", "T-", "H-", "D-", "R-")

JUDGE_SYSTEM = """\
You grade a memory that an assistant wrote about one person against ground truth
about that person's preferences. The memory is the assistant's only record of them.

Judge only what the memory says. Do not infer a belief the memory does not state,
and do not give credit for evidence that is recorded but never turned into a belief.
Entries marked `superseded`, `retracted`, `expired` or `refuted` are past beliefs;
`current`, `active`, `confirmed` and `open` entries (and the working context) are
what the memory believes now. An `open` hypothesis is a tentative belief: it counts
as `partial` at most.

The ground truth uses a coarse taxonomy. Read each attribute by its meaning: for
example, `data_driven_emphasis: True` means they want decisions grounded in data
and evidence.

Credit an entry only if it is about the attribute's own subject. A preference about
how answers are formatted (checklists, bullets, step-by-step) says nothing about how
fast they decide or how they handle ambiguity, and a preference about tone says
nothing about how much they rely on evidence. When no entry is about the subject,
the grade is `absent`, however close the wording looks.

For every attribute, report:
- memory_value: what the memory currently believes about it, in a short phrase,
  or "nothing".
- current_match: `matches` (a current belief on this subject states the expected
  current value), `partial` (a current belief on this subject leans toward it, or
  only an open hypothesis states it), `absent` (no belief on this subject) or
  `contradicts` (a current belief on this subject is incompatible with the current
  value).
- For attributes that changed: original_still_current (a current belief states the
  original value) and change_recorded (the memory shows its belief on this subject
  moving toward the current value: a superseded entry whose content on this subject
  differs from its successor's, or a hypothesis tying the life event to this
  subject; a supersession that only rewords the same belief does not count).
  For attributes that did not change, set both to false.
- evidence: the entry IDs (or short quotes) your grade rests on.

Submit everything with one submit_grades call."""


def memory_text(snapshot: Path) -> str:
    """The beliefs in a saved memory: state and working context, without raw episodes.

    Works for the agent's workspace and for the rolling-summary baseline.
    """
    parts = []
    for path in sorted(snapshot.rglob("*.md")):
        rel = path.relative_to(snapshot).as_posix()
        text = path.read_text()
        if rel == "memory/episodes.md":
            continue
        if rel == "memory/archive.md":  # closed beliefs live here too; episodes do not belong in the grade
            text = "\n".join(line for line in text.splitlines() if line.lstrip("- ").startswith(_BELIEF_TYPES))
        if text.strip():
            parts.append(f"<file path=\"{rel}\">\n{text.strip()}\n</file>")
    return "\n\n".join(parts)


def ground_truth(gold: GoldLabel) -> dict[str, Any]:
    pe = gold.preference_evolution or {}
    current = pe.get("current_attributes") or {}
    changed = pe.get("changed_attributes") or {}
    events = [{"date": str(h.get("date", ""))[:10], "event": h.get("event_name", "")}
              for h in pe.get("evolution_history", [])]
    return {
        "preference_domain": gold.preference_domain,
        "attributes": [
            {"attribute": k, "current": str(v),
             **({"original": str(changed[k]["original"]), "changed": True} if k in changed else {"changed": False})}
            for k, v in current.items()
        ],
        "life_events": events,
    }


def _submit_spec(attributes: list[str]) -> ToolSpec:
    grade = {
        "type": "object",
        "properties": {
            "attribute": {"type": "string", "enum": attributes},
            "memory_value": {"type": "string"},
            "current_match": {"type": "string", "enum": list(MATCHES)},
            "original_still_current": {"type": "boolean"},
            "change_recorded": {"type": "boolean"},
            "evidence": {"type": "string"},
        },
        "required": ["attribute", "memory_value", "current_match", "original_still_current",
                     "change_recorded", "evidence"],
    }
    return ToolSpec("submit_grades", "Submit one grade per attribute.",
                    {"type": "object", "properties": {"grades": {"type": "array", "items": grade}},
                     "required": ["grades"]})


def render_judge_input(truth: dict[str, Any], memory: str) -> str:
    lines = [f"<ground_truth domain=\"{truth['preference_domain']}\">"]
    for a in truth["attributes"]:
        if a["changed"]:
            lines.append(f"- {a['attribute']}: now {a['current']} (changed from {a['original']})")
        else:
            lines.append(f"- {a['attribute']}: {a['current']} (unchanged)")
    for e in truth["life_events"]:
        lines.append(f"Change triggered by: {e['event']} ({e['date']})")
    lines.append("</ground_truth>")
    return "\n".join(lines) + f"\n\n<memory>\n{memory or '(empty)'}\n</memory>\n\nGrade every attribute."


def score_grades(truth: dict[str, Any], grades: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-attribute grades. A missing grade counts as `absent`."""
    by_attr = {g["attribute"]: g for g in grades}
    rows = []
    for a in truth["attributes"]:
        g = by_attr.get(a["attribute"], {"current_match": "absent", "memory_value": "(not graded)",
                                         "original_still_current": False, "change_recorded": False, "evidence": ""})
        rows.append({**a, **{k: g.get(k) for k in ("memory_value", "current_match", "original_still_current",
                                                   "change_recorded", "evidence")}})
    changed = [r for r in rows if r["changed"]]

    def mean(rs):
        return round(sum(MATCH_SCORE.get(r["current_match"], 0.0) for r in rs) / len(rs), 3) if rs else None

    return {
        "attributes": rows,
        "current_fidelity": mean(rows),            # how well memory holds the person as they are now
        "changed_fidelity": mean(changed),         # the same, on the attributes that changed
        "stale": sum(bool(r["original_still_current"]) for r in changed),
        "changes_recorded": sum(bool(r["change_recorded"]) for r in changed),
        "n_attributes": len(rows),
        "n_changed": len(changed),
        "ungraded": [a["attribute"] for a in truth["attributes"] if a["attribute"] not in by_attr],
    }


def combine_repeats(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Majority grade per attribute across repeated judgings; ties go to the lower score."""
    if len(runs) == 1:
        return runs[0]
    rows, agree = [], []
    for i, first in enumerate(runs[0]["attributes"]):
        grades = [r["attributes"][i] for r in runs]
        counts = {m: sum(g["current_match"] == m for g in grades) for m in MATCHES}
        mode = max(MATCHES, key=lambda m: (counts[m], -MATCH_SCORE[m]))
        agree.append(counts[mode] / len(runs))
        majority = lambda key: sum(bool(g[key]) for g in grades) * 2 > len(runs)  # noqa: E731
        rows.append({**next(g for g in grades if g["current_match"] == mode), "current_match": mode,
                     "original_still_current": majority("original_still_current"),
                     "change_recorded": majority("change_recorded"),
                     "votes": [g["current_match"] for g in grades]})
    truth = {"attributes": [{k: r[k] for k in ("attribute", "current", "changed") + (("original",) if r["changed"] else ())}
                            for r in rows]}
    combined = score_grades(truth, rows)
    combined["attributes"] = rows
    fid = [r["current_fidelity"] for r in runs]
    return {**{k: runs[0][k] for k in ("item_id", "judge_model", "memory_chars")}, **combined,
            "repeats": len(runs), "agreement": round(sum(agree) / len(agree), 3),
            "current_fidelity_range": [min(fid), max(fid)],
            "estimated_cost_usd": round(sum(r["estimated_cost_usd"] or 0 for r in runs), 5),
            "runs": runs}


def judge_memory(model: ModelAdapter, gold: GoldLabel, snapshot: Path) -> dict[str, Any]:
    truth = ground_truth(gold)
    if not truth["attributes"]:
        raise ValueError(f"{gold.item_id} has no ground-truth attributes to grade against")
    memory = memory_text(snapshot)
    spec = _submit_spec([a["attribute"] for a in truth["attributes"]])
    request = ModelRequest(phase="judge", system=JUDGE_SYSTEM,
                           blocks=(TextBlock(render_judge_input(truth, memory)),), tools=(spec,))
    usage, grades, response = [], None, None
    for _ in range(2):  # one retry if the judge answers without the tool
        response = model.complete(request)
        usage.append(response.usage)
        call = next((c for c in response.tool_calls if c.name == "submit_grades"), None)
        if call is not None and isinstance(call.arguments.get("grades"), list):
            grades = call.arguments["grades"]
            break
    if grades is None:
        raise RuntimeError(f"The judge did not submit grades (stop reason {response.stop_reason})")
    costs = [cost_usd(u, model.settings.model) for u in usage]
    return {
        "item_id": gold.item_id,
        "judge_model": model.settings.model,
        "memory_chars": len(memory),
        **score_grades(truth, grades),
        "estimated_cost_usd": None if None in costs else round(sum(costs), 5),
    }


def write_judgement(out: Path, judgement: dict[str, Any]) -> None:
    out.write_text(json.dumps(judgement, indent=2))
