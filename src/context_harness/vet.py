"""Item vetting: does a labeled preference change actually show up in the sessions?

Evaluator side only. HorizonBench items are generated, and a label can disagree with
the conversations it labels: in sample item 5 the person is labeled as coming to want
longer answers, yet keeps asking for "shorter replies" and ends with "strip it to pure
bullets". An item like that measures the label, not the memory.

For each changed attribute, a model reads the person's own turns before and after the
change and says which value they show in each period. It also checks whether the
question's final request conflicts with the labeled answer. Two label-free flags are
computed directly: whether the correct option is the shortest (the benchmark's known
length bias) and how many sessions follow the change.
"""

from __future__ import annotations

from typing import Any

from .models.base import ModelAdapter, ModelRequest, TextBlock, ToolSpec
from .models.pricing import cost_usd
from .types import BenchmarkItem

COUNT_FIELDS = ("before_original", "before_current", "after_original", "after_current")
TURN_CHARS = 500  # per user turn; almost all turns are shorter

VET_SYSTEM = """\
You check a benchmark item for consistency. The item labels a person's preference
attributes, with values before and after a life event. You see what the person
themselves wrote before and after that event, and you judge whether their own words
show each value. Ignore the content of their tasks unless it reveals how they want to
be helped.

For each changed attribute, count sessions on BOTH sides. Quotes that support a value
are easy to find in a long history; what matters is the balance.
- before_original / before_current: how many sessions before the change contain the
  person's own words pointing to the original value / to the current value.
- after_original / after_current: the same, for sessions after the change.
A session that points both ways counts on both sides; one that says nothing counts on
neither.
- evidence: session IDs and short quotes for each side, especially after the change.

Then compare the question's final conversation with the labeled correct response:
final_turn_conflict is `yes` if the person's last request there asks for something the
labeled correct response does not do (or the trap response does), `no` if it fits,
`unclear` otherwise. Explain in one sentence.

Submit everything with one submit_vetting call."""


def _user_turns(sessions, start: int) -> str:
    lines = []
    for i, s in enumerate(sessions, start):
        turns = [t.content.strip().replace("\n", " ")[:TURN_CHARS] for t in s.turns if t.role == "user"]
        if turns:
            lines.append(f"[S-{i} {s.date[:10]}] " + " / ".join(turns))
    return "\n".join(lines) or "(none)"


def label_free_flags(item: BenchmarkItem) -> dict[str, Any]:
    gold, q = item.gold, item.episode.question
    lengths = {o.letter: len(o.text) for o in q.options}
    pe = gold.preference_evolution or {}
    hist = pe.get("evolution_history") or []
    change = str(hist[0].get("date", ""))[:10] if hist else None
    after = sum(1 for s in item.episode.sessions if change and s.date[:10] >= change)
    return {
        "correct_is_shortest": lengths[gold.correct_letter] == min(lengths.values()),
        "sessions_before_change": len(item.episode.sessions) - after,
        "sessions_after_change": after,
        "change_date": change,
    }


def render_vet_input(item: BenchmarkItem) -> str:
    gold, ep = item.gold, item.episode
    pe = gold.preference_evolution or {}
    change = label_free_flags(item)["change_date"]
    split = next((i for i, s in enumerate(ep.sessions) if s.date[:10] >= change), len(ep.sessions))
    events = "; ".join(f"{h.get('event_name', '')} ({str(h.get('date', ''))[:10]})" for h in pe.get("evolution_history", []))
    attrs = "\n".join(f"- {k}: {v['original']} -> {v['current']}" for k, v in pe["changed_attributes"].items())
    options = {o.letter: o.text for o in ep.question.options}
    trap = gold.distractor_letter
    final_turns = "\n".join(f"{t.role}: {t.content.strip()[:1500]}" for t in ep.probe.turns[-3:])
    return (
        f"<labels domain=\"{gold.preference_domain}\" life_event=\"{events}\">\n{attrs}\n</labels>\n\n"
        f"<before_change>\n{_user_turns(ep.sessions[:split], 0)}\n</before_change>\n\n"
        f"<after_change>\n{_user_turns(ep.sessions[split:], split)}\n</after_change>\n\n"
        f"<question_conversation_end>\n{final_turns}\n</question_conversation_end>\n\n"
        f"<labeled_correct_response value=\"{gold.option_values.get(gold.correct_letter, '')}\">\n"
        f"{options[gold.correct_letter]}\n</labeled_correct_response>\n"
        + (f"<trap_response value=\"{gold.option_values.get(trap, '')}\">\n{options[trap]}\n</trap_response>\n"
           if trap else "")
    )


def _submit_spec(attributes: list[str]) -> ToolSpec:
    attr = {
        "type": "object",
        "properties": {
            "attribute": {"type": "string", "enum": attributes},
            **{f: {"type": "integer", "minimum": 0} for f in COUNT_FIELDS},
            "evidence": {"type": "string"},
        },
        "required": ["attribute", *COUNT_FIELDS, "evidence"],
    }
    return ToolSpec("submit_vetting", "Submit the vetting for this item.", {
        "type": "object",
        "properties": {
            "attributes": {"type": "array", "items": attr},
            "final_turn_conflict": {"type": "string", "enum": ["yes", "no", "unclear"]},
            "final_turn_explanation": {"type": "string"},
        },
        "required": ["attributes", "final_turn_conflict", "final_turn_explanation"],
    })


def change_supported(a: dict[str, Any]) -> bool:
    """The person's own words favor the new value after the change, and didn't already
    favor it before. (Item 6: 25 sessions after the change still asked for numbered
    steps, against 12 for summaries; a one-sided reading had called it a change.)"""
    n = {f: int(a.get(f) or 0) for f in COUNT_FIELDS}
    return n["after_current"] > n["after_original"] and n["before_original"] >= n["before_current"]


def verdict(attributes: list[dict[str, Any]], final_turn_conflict: str) -> tuple[str, list[str]]:
    """Two uses, two bars. Grading memory against the labels needs a change that shows in
    the person's own words; trusting the multiple-choice answer also needs a question
    whose final request doesn't contradict the labeled answer.

    `usable`: both hold. `memory_only`: the change shows, but the answer is unreliable.
    `suspect`: no labeled change shows."""
    shown = [a for a in attributes if change_supported(a)]
    reasons = []
    if not shown:
        reasons.append("no changed attribute shows its new value after the change")
    for a in attributes:
        if not change_supported(a):
            reasons.append(f"{a['attribute']}: after the change, {a.get('after_original', 0)} sessions point to the "
                           f"original value and {a.get('after_current', 0)} to the new one "
                           f"(before: {a.get('before_original', 0)} vs {a.get('before_current', 0)})")
    if final_turn_conflict == "yes":
        reasons.append("the question's final request conflicts with the labeled answer")
    if not shown:
        return "suspect", reasons
    return ("memory_only" if final_turn_conflict == "yes" else "usable"), reasons


def vet_item(model: ModelAdapter, item: BenchmarkItem) -> dict[str, Any]:
    gold = item.gold
    flags = label_free_flags(item)
    changed = list(((gold.preference_evolution or {}).get("changed_attributes") or {}).keys())
    if not changed:
        return {"item_id": gold.item_id, **flags, "verdict": "no_change", "reasons": ["no labeled change"],
                "attributes": [], "estimated_cost_usd": 0.0}
    request = ModelRequest(phase="vet", system=VET_SYSTEM, blocks=(TextBlock(render_vet_input(item)),),
                           tools=(_submit_spec(changed),))
    usage, args, response = [], None, None
    for _ in range(2):  # one retry if the model answers without the tool
        response = model.complete(request)
        usage.append(response.usage)
        call = next((c for c in response.tool_calls if c.name == "submit_vetting"), None)
        if call is not None and isinstance(call.arguments.get("attributes"), list):
            args = call.arguments
            break
    if args is None:
        raise RuntimeError(f"The vetting model did not submit (stop reason {response.stop_reason})")
    graded = {a.get("attribute"): a for a in args["attributes"]}
    attributes = [graded.get(k, {"attribute": k, **{f: 0 for f in COUNT_FIELDS}, "evidence": "(not graded)"})
                  for k in changed]
    for a in attributes:
        a["change_supported"] = change_supported(a)
    v, reasons = verdict(attributes, args.get("final_turn_conflict", "unclear"))
    costs = [cost_usd(u, model.settings.model) for u in usage]
    return {
        "item_id": gold.item_id,
        **flags,
        "verdict": v,
        "reasons": reasons,
        "attributes": attributes,
        "supported_attributes": [a["attribute"] for a in attributes if a["change_supported"]],
        "final_turn_conflict": args.get("final_turn_conflict"),
        "final_turn_explanation": args.get("final_turn_explanation", ""),
        "vet_model": model.settings.model,
        "estimated_cost_usd": None if None in costs else round(sum(costs), 5),
    }
