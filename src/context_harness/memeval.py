"""Evaluation framework: grade memory and replies against a reference checklist.

Evaluator side only, and independent of any benchmark's labels. For each test case a
person writes, before the run, a short checklist:

- must_contain: facts and current preferences a correct memory holds;
- must_not_assert: outdated beliefs a correct memory no longer presents as current;
- reply_should: what a good reply to the final message does.

An LLM judge marks each item, and scores six general criteria of memory management
against the person's own messages (never the memory method's notes). Each judgment is
repeated and the majority kept, with the agreement between repeats reported.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from .models.base import ModelAdapter, ModelRequest, TextBlock, ToolSpec
from .models.pricing import cost_usd
from .types import BenchmarkItem

CRITERIA = {
    "faithfulness": "Every belief in memory is supported by what the person said or did; nothing invented.",
    "currency": "Memory describes the person now; beliefs that changed are replaced, not kept as current.",
    "recall": "The important facts and preferences are there, including ones mentioned only in passing.",
    "change_tracking": "When something changed, memory records what changed and roughly when.",
    "provenance": "Claims point to their evidence (cited observations, dates or quotes), and it supports them.",
    "consistency": "No two current beliefs contradict each other.",
}
CONTAIN_MARKS = ("present", "partial", "missing", "contradicted")
STALE_MARKS = ("asserted", "past_only", "absent")  # asserted = still presented as current (bad)
REPLY_MARKS = ("follows", "partial", "ignores", "violates")
TURN_CHARS = 500

MEMORY_JUDGE_SYSTEM = """\
You evaluate how well an assistant's memory of one person is managed. You see the
person's own messages from every earlier conversation (the evidence), the memory the
assistant kept, and a reference checklist written in advance. Judge only what the
memory says; don't infer what it doesn't state.

For each must_contain item, mark it:
- present: memory states it as current;
- partial: memory states part of it, or only tentatively;
- missing: memory doesn't state it;
- contradicted: memory states something incompatible as current.

For each must_not_assert item (an outdated belief), mark it:
- asserted: memory still presents it as how the person is now;
- past_only: memory mentions it only as the past ("used to", "was", a change log);
- absent: memory doesn't mention it.

Then score each criterion 0, 1 or 2 against the evidence:
{criteria}
2 = fully meets it, 1 = partly, 0 = fails. Give one short reason per score.

Submit everything with one submit_evaluation call."""

REPLY_JUDGE_SYSTEM = """\
You evaluate a reply an assistant wrote to a person's latest message, against a
reference checklist of what a good reply does, written in advance from what the person
told the assistant before. For each item mark it:
- follows: the reply clearly does it;
- partial: partly;
- ignores: the reply doesn't address it (neither follows nor goes against it);
- violates: the reply goes against it.
Also report asks_for_known (true if the reply asks the person for something the
checklist shows the assistant already knew) and give one short reason per item.
Submit everything with one submit_evaluation call."""


def load_checklists(paths: list[str | Path]) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for path in paths:
        out.update(json.loads(Path(path).read_text()))
    return out


def person_messages(item: BenchmarkItem) -> str:
    """The person's own words from every earlier session: the evidence a memory must match."""
    lines = []
    for i, s in enumerate(item.episode.sessions):
        turns = [t.content.strip().replace("\n", " ")[:TURN_CHARS] for t in s.turns if t.role == "user"]
        if turns:
            lines.append(f"[S-{i} {s.date[:10]}] " + " / ".join(turns))
    return "\n".join(lines)


def _numbered(prefix: str, items: list[str]) -> str:
    return "\n".join(f"{prefix}{i}. {text}" for i, text in enumerate(items, 1)) or "(none)"


def _spec(name: str, item_props: dict, extra: dict | None = None) -> ToolSpec:
    return ToolSpec("submit_evaluation", f"Submit the {name} evaluation.", {
        "type": "object", "properties": {**item_props, **(extra or {})}, "required": list(item_props)})


def _marks(ids: list[str], marks: tuple[str, ...]) -> dict:
    return {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string", "enum": ids}, "mark": {"type": "string", "enum": list(marks)},
        "evidence": {"type": "string"}}, "required": ["id", "mark"]}}


def _call(model: ModelAdapter, request: ModelRequest) -> tuple[dict | None, float]:
    for _ in range(2):
        response = model.complete(request)
        call = next((c for c in response.tool_calls if c.name == "submit_evaluation"), None)
        if call is not None:
            return call.arguments, cost_usd(response.usage, model.settings.model) or 0.0
    return None, cost_usd(response.usage, model.settings.model) or 0.0


def _majority(votes: list[str], order: tuple[str, ...]) -> str:
    """Most common mark; ties go to the worse mark (later in `order` is worse for contain/reply)."""
    counts = Counter(votes)
    return max(order, key=lambda m: (counts[m], order.index(m))) if votes else order[-2]


def evaluate_memory(model: ModelAdapter, item: BenchmarkItem, memory: str, checklist: dict[str, list[str]],
                    repeats: int = 3) -> dict[str, Any]:
    contain, stale = checklist.get("must_contain", []), checklist.get("must_not_assert", [])
    c_ids, s_ids = [f"M{i}" for i in range(1, len(contain) + 1)], [f"N{i}" for i in range(1, len(stale) + 1)]
    text = (f"<evidence>\n{person_messages(item)}\n</evidence>\n\n<memory>\n{memory or '(empty)'}\n</memory>\n\n"
            f"<must_contain>\n{_numbered('M', contain)}\n</must_contain>\n\n"
            f"<must_not_assert>\n{_numbered('N', stale)}\n</must_not_assert>")
    spec = _spec("memory", {
        "must_contain": _marks(c_ids, CONTAIN_MARKS),
        "must_not_assert": _marks(s_ids, STALE_MARKS),
        "criteria": {"type": "array", "items": {"type": "object", "properties": {
            "criterion": {"type": "string", "enum": list(CRITERIA)}, "score": {"type": "integer", "enum": [0, 1, 2]},
            "reason": {"type": "string"}}, "required": ["criterion", "score"]}},
    })
    system = MEMORY_JUDGE_SYSTEM.format(criteria="\n".join(f"- {k}: {v}" for k, v in CRITERIA.items()))
    request = ModelRequest(phase="eval_memory", system=system, blocks=(TextBlock(text),), tools=(spec,))
    votes: dict[str, list] = {i: [] for i in c_ids + s_ids + list(CRITERIA)}
    evidence: dict[str, str] = {}
    cost = 0.0
    for _ in range(repeats):
        args, c = _call(model, request)
        cost += c
        for key, marks in (("must_contain", CONTAIN_MARKS), ("must_not_assert", STALE_MARKS)):
            for m in (args or {}).get(key, []):
                if m.get("id") in votes and m.get("mark") in marks:
                    votes[m["id"]].append(m["mark"])
                    evidence.setdefault(m["id"], m.get("evidence", ""))
        for s in (args or {}).get("criteria", []):
            if s.get("criterion") in CRITERIA and s.get("score") in (0, 1, 2):
                votes[s["criterion"]].append(s["score"])
                evidence.setdefault(s["criterion"], s.get("reason", ""))
    rows = ([{"id": i, "item": t, "kind": "must_contain", "mark": _majority(votes[i], CONTAIN_MARKS), "votes": votes[i],
              "evidence": evidence.get(i, "")} for i, t in zip(c_ids, contain)]
            + [{"id": i, "item": t, "kind": "must_not_assert", "mark": _majority(votes[i], ("absent", "past_only", "asserted")),
                "votes": votes[i], "evidence": evidence.get(i, "")} for i, t in zip(s_ids, stale)])
    scores = {k: (median(votes[k]) if votes[k] else None) for k in CRITERIA}
    agree = [Counter(v).most_common(1)[0][1] / len(v) for k, v in votes.items() if v]
    contain_rows = [r for r in rows if r["kind"] == "must_contain"]
    stale_rows = [r for r in rows if r["kind"] == "must_not_assert"]
    return {
        "checklist": rows,
        "contain_rate": round(sum(r["mark"] == "present" for r in contain_rows) / len(contain_rows), 3) if contain_rows else None,
        "stale_rate": round(sum(r["mark"] == "asserted" for r in stale_rows) / len(stale_rows), 3) if stale_rows else None,
        "criteria": scores,
        "criteria_reasons": {k: evidence.get(k, "") for k in CRITERIA},
        "agreement": round(sum(agree) / len(agree), 3) if agree else None,
        "judge_cost_usd": round(cost, 5),
    }


def evaluate_reply(model: ModelAdapter, item: BenchmarkItem, reply: str, checklist: dict[str, list[str]],
                   repeats: int = 3) -> dict[str, Any]:
    should = checklist.get("reply_should", [])
    ids = [f"R{i}" for i in range(1, len(should) + 1)]
    convo = "\n".join(f"{t.role}: {t.content.strip()[:1500]}" for t in item.episode.probe.turns[-3:])
    text = (f"<known_before>\n{_numbered('K', checklist.get('must_contain', []))}\n</known_before>\n\n"
            f"<conversation_end>\n{convo}\n</conversation_end>\n\n<reply>\n{reply}\n</reply>\n\n"
            f"<reply_should>\n{_numbered('R', should)}\n</reply_should>")
    spec = _spec("reply", {"reply_should": _marks(ids, REPLY_MARKS)}, {"asks_for_known": {"type": "boolean"}})
    request = ModelRequest(phase="eval_reply", system=REPLY_JUDGE_SYSTEM, blocks=(TextBlock(text),), tools=(spec,))
    votes: dict[str, list] = {i: [] for i in ids}
    asks: list[bool] = []
    evidence: dict[str, str] = {}
    cost = 0.0
    for _ in range(repeats):
        args, c = _call(model, request)
        cost += c
        for m in (args or {}).get("reply_should", []):
            if m.get("id") in votes and m.get("mark") in REPLY_MARKS:
                votes[m["id"]].append(m["mark"])
                evidence.setdefault(m["id"], m.get("evidence", ""))
        if isinstance((args or {}).get("asks_for_known"), bool):
            asks.append(args["asks_for_known"])
    rows = [{"id": i, "item": t, "mark": _majority(votes[i], REPLY_MARKS), "votes": votes[i],
             "evidence": evidence.get(i, "")} for i, t in zip(ids, should)]
    agree = [Counter(v).most_common(1)[0][1] / len(v) for v in votes.values() if v]
    return {
        "checklist": rows,
        "follow_rate": round(sum(r["mark"] == "follows" for r in rows) / len(rows), 3) if rows else None,
        "violate_rate": round(sum(r["mark"] == "violates" for r in rows) / len(rows), 3) if rows else None,
        "asks_for_known": sum(asks) * 2 > len(asks) if asks else None,
        "agreement": round(sum(agree) / len(agree), 3) if agree else None,
        "judge_cost_usd": round(cost, 5),
    }
