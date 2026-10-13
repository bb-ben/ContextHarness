"""MemConflict adapter: does the agent use the right memory when memories disagree?

MemConflict (Tao et al. 2026, https://github.com/TaoZhen1110/MemConflict) gives each
person a chain of ~53 dated sessions and attaches questions to sessions. Three conflict
types: dynamic (a real update: the current state must win), static (a later false
contradiction: the true fact must survive, ideally with the conflict noticed), and
conditional (a preference that holds under a condition).

Here a memory method replays the chain; after each session that carries questions, a
sample of them is answered from the method's memory alone. A judge applies
MemConflict's published metric definitions (answer accuracy, update awareness for
dynamic, conflict recognition for static) plus one check of our own: does the memory
itself hold the evidence for the reference answer (MemConflict's white-box metrics rank
retrieved items, which a profile or summary doesn't have).

The repository states no license, so its data is read from a local clone and never
copied into this repo.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from ..models.base import ModelAdapter, ToolSpec
from ..types import Session, Turn
from .common import call_tool, memory_text, open_method

TYPES = ("dynamic_conflict", "static_conflict", "conditional_conflict")

ANSWER_SYSTEM = """\
You answer questions about a user from your memory of them.
Rules:
1. Use only what your memory says.
2. Do not invent facts the memory doesn't support.
3. If the memory is insufficient, say you cannot confirm.
4. If the memory holds inconsistent statements, briefly mention the inconsistency first,
   then give the best-supported answer.
5. Keep the answer concise and direct."""

JUDGE_SYSTEM = """\
You are a strict evaluator for memory-conflict questions. Use the reference answer as
the gold standard; do not require exact wording.

Score:
- answer_accuracy: for dynamic and static questions, 1.0 if the model answer captures
  the core correct fact (the current state after an update, or the true fact despite a
  false contradiction), 0.5 if partially correct (a correct key element but incomplete
  or vague), 0.0 if wrong, contradictory, missing the key fact, or overly uncertain. For
  conditional questions, 1.0 if it gives a correct core condition, else 0.0.
- update_awareness (dynamic only, else 0): 1 if the answer shows that a change happened,
  in the right old-to-new direction.
- conflict_recognition (static only, else 0): 1 if the answer recognizes the
  inconsistency or uncertainty in the case.
- memory_support: 1 if the memory shown contains information supporting the reference
  answer, else 0.
Give a short reason, and submit with the score tool."""


def load_people(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sessions_of(person: dict[str, Any]) -> list[Session]:
    out = []
    for i, s in enumerate(person["Full_Session_Chain"]):
        turns = []
        for _, pair in sorted(s.get("Session_Dialogue", {}).items(), key=lambda kv: int(kv[0].rsplit("_", 1)[-1])):
            for t in pair:
                if t.get("role") in ("user", "assistant") and t.get("content"):
                    turns.append(Turn(t["role"], t["content"]))
        out.append(Session(index=i, source_index=i, date=s["Date"], scenario="", turns=tuple(turns)))
    return out


def sample_questions(person: dict[str, Any], per_person: int, seed: int = 0) -> dict[int, list[dict[str, Any]]]:
    """Questions to ask after each session index, stratified by conflict type.

    Dynamic questions make up ~80% of the benchmark; the sample gives each type an equal
    share (or all of a type, if fewer), spread over the chain, so every type is measured.
    """
    rng = random.Random(f"{person['ID']}:{seed}")
    by_type: dict[str, list[tuple[int, int, dict]]] = {t: [] for t in TYPES}
    for i, s in enumerate(person["Full_Session_Chain"]):
        for j, q in enumerate(s.get("Session_Questions", [])):
            if q.get("conflict_type") in by_type:
                by_type[q["conflict_type"]].append((i, j, q))
    share = max(1, per_person // len(TYPES))
    chosen = sorted((x[:2], x[2]) for items in by_type.values() for x in rng.sample(items, min(share, len(items))))
    picked: dict[int, list[dict]] = {}
    for (i, _), q in chosen:  # in chain order, and in each session's own order
        picked.setdefault(i, []).append(q)
    return picked


def _judge_spec() -> ToolSpec:
    return ToolSpec("score", "Submit the scores.", {"type": "object", "properties": {
        "answer_accuracy": {"type": "number", "enum": [0, 0.5, 1]},
        "update_awareness": {"type": "integer", "enum": [0, 1]},
        "conflict_recognition": {"type": "integer", "enum": [0, 1]},
        "memory_support": {"type": "integer", "enum": [0, 1]},
        "reason": {"type": "string"}}, "required": ["answer_accuracy", "memory_support", "reason"]})


def _answer_spec() -> ToolSpec:
    return ToolSpec("answer", "Submit your answer.", {"type": "object", "properties": {
        "answer": {"type": "string"}}, "required": ["answer"]})


def run_person(person: dict[str, Any], method_name: str, model: ModelAdapter, judge: ModelAdapter,
               out_dir: Path, per_person: int) -> list[dict[str, Any]]:
    method, trace = open_method(method_name, model, out_dir)
    questions = sample_questions(person, per_person)
    rows = []
    saved = (out_dir / "questions.jsonl").open("w")  # row by row, so a run stopped by its cap keeps them
    for session in sessions_of(person):
        method.observe(session)
        for q in questions.get(session.index, []):
            memory = memory_text(method)
            args, answer_cost, text = call_tool(model, "memconflict_answer", ANSWER_SYSTEM,
                                                f"<memory>\n{memory or '(empty)'}\n</memory>\n\nQuestion: {q['question']}",
                                                _answer_spec())
            answer = (args or {}).get("answer") or text or ""
            graded, judge_cost, _ = call_tool(judge, "memconflict_judge", JUDGE_SYSTEM, (
                f"Conflict type: {q['conflict_type']}\nQuestion: {q['question']}\nReference answer: {q['answer']}\n"
                f"Model answer: {answer}\n\n<memory>\n{memory or '(empty)'}\n</memory>"), _judge_spec())
            graded = graded or {}
            rows.append({
                "person": person["ID"], "session": session.index, "date": session.date,
                "question_id": q["question_id"], "conflict_type": q["conflict_type"], "difficulty": q.get("difficulty"),
                "question": q["question"], "reference": q["answer"], "answer": answer,
                "answer_accuracy": float(graded.get("answer_accuracy", 0) or 0),
                "update_awareness": int(graded.get("update_awareness", 0) or 0),
                "conflict_recognition": int(graded.get("conflict_recognition", 0) or 0),
                "memory_support": int(graded.get("memory_support", 0) or 0),
                "reason": graded.get("reason", ""), "answer_cost_usd": round(answer_cost, 5),
                "judge_cost_usd": round(judge_cost, 5),
            })
            saved.write(json.dumps(rows[-1]) + "\n")
            saved.flush()
    trace.close()
    saved.close()
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """MemConflict's headline numbers per conflict type, from our rows."""
    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    out: dict[str, Any] = {"n": len(rows), "answer_accuracy": mean([r["answer_accuracy"] for r in rows])}
    for t in TYPES:
        rs = [r for r in rows if r["conflict_type"] == t]
        out[t] = {"n": len(rs), "answer_accuracy": mean([r["answer_accuracy"] for r in rs]),
                  "memory_support": mean([r["memory_support"] for r in rs])}
        if t == "dynamic_conflict":
            out[t]["update_awareness"] = mean([r["update_awareness"] for r in rs])
        if t == "static_conflict":
            out[t]["conflict_recognition"] = mean([r["conflict_recognition"] for r in rs])
    return out
