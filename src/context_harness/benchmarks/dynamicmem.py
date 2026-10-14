"""DynamicMem adapter: does the user model stay accurate over months?

DynamicMem (https://github.com/wenyaxie023/DynamicMem, MIT; data on Hugging Face at
xiewenya/dynamicmem) gives each user 1,000-2,600 app-activity logs over ~15 months and
5 checkpoints. At each checkpoint a memory system must (A) fill in the user's current
state for 17-48 habits and preferences, from templates, and (C) carry out personalized
service requests that depend on that state.

Here a memory method replays the logs grouped into weekly events (one event per log
would be ~10x the calls), split so every checkpoint falls on an event boundary. At each
checkpoint, State Completion is one call over all the checkpoint's templates and
Personalized Service one call over all its requests, both answered from the method's
memory alone. Predictions are written in DynamicMem's own format and scored with its
own evaluator (`python -m evaluation.eval_tce`, run from a local clone).
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..models.base import ModelAdapter, ToolSpec
from ..tracing import TracedModel
from ..types import Session, Turn
from .common import call_tool, memory_text, open_method

LOG_CHARS = 420  # per rendered log line; most logs are shorter

# DynamicMem's headline numbers, as its evaluator names them in tce_eval.json "summary".
# Its default LLM judge is the holistic one: per field, 0.8 * core correct + 0.2 * detail/2.
HEADLINE = {
    "state_completion": "snapshot_holistic_score_mean_on_expected_mean",
    "personalized_service": "rq3_apply_holistic_score_mean_mean",
    "state_value_f1": "snapshot_value_f1_mean_on_expected_mean",  # rule-based slot match
}

STATE_SYSTEM = """\
You fill in a user's CURRENT state from your memory of them: their habits, routines and
preferences as they are now (the latest values, not earlier ones). For each requested
key, fill the template's blanks with the current value; leave a key out if memory has
nothing on it. Use the encodings the question gives (weekday 0 = Monday, times as HH:MM).
Submit everything with the submit tool."""

SERVICE_SYSTEM = """\
You are the user's personal assistant. For each request, do what it asks for this user,
using what your memory says about their current habits, routines and preferences. Be
specific: use their actual times, places, people and preferences. When a request gives
an output template, answer in that template's structure. Submit everything with the
submit tool."""


def load_user(user_dir: Path) -> tuple[list[dict], dict]:
    return json.loads((user_dir / "app_log_large.json").read_text()), json.loads((user_dir / "task_packs.json").read_text())


def _render_log(log: dict[str, Any]) -> str:
    body = json.dumps({"request": log.get("request"), "response": log.get("response")}, ensure_ascii=False,
                      separators=(",", ":"))
    line = f"{log['timestamp']} {log['app_name']}/{log['api_name']} [{log['app_log_id']}] {body}"
    return line if len(line) <= LOG_CHARS else line[: LOG_CHARS - 1] + "…"


def weekly_events(logs: list[dict], checkpoint_indices: list[int]) -> list[tuple[Session, int]]:
    """(event, index of its last log), one per ISO week, also split after each checkpoint log."""
    cuts = set(checkpoint_indices)
    events, batch, week = [], [], None
    for i, log in enumerate(logs):
        w = dt.datetime.fromisoformat(log["timestamp"]).isocalendar()[:2]
        if batch and w != week:
            events.append(batch)
            batch = []
        week = w
        batch.append(i)
        if i in cuts:
            events.append(batch)
            batch = []
    if batch:
        events.append(batch)
    out = []
    for k, idx in enumerate(events):
        text = "App activity from the user's phone (not a conversation):\n" + "\n".join(_render_log(logs[i]) for i in idx)
        out.append((Session(index=k, source_index=idx[0], date=logs[idx[0]]["timestamp"],
                            scenario="Logs of the user's own app activity this week.", turns=(Turn("user", text),)),
                    idx[-1]))
    return out


def _state_request(pack: dict[str, Any]) -> tuple[str, list[str]]:
    keys = list(pack["keys"])
    lines = [f"- {k}: {pack['keys'][k]['question_text']}" for k in keys]
    return "\n".join(lines), keys


def _service_items(checkpoint: dict[str, Any]) -> list[tuple[str, dict]]:
    qa = checkpoint.get("rq3_apply_service_qa") or {}
    return [(key, item) for key, block in (qa.get("keys") or {}).items() for item in block.get("items", [])]


def answer_checkpoint(model: ModelAdapter, memory: str, checkpoint: dict[str, Any]) -> tuple[dict, dict, float]:
    """State Completion and Personalized Service for one checkpoint, from memory alone."""
    requested, keys = _state_request(checkpoint["state_completion_pack"])
    state_spec = ToolSpec("submit", "Submit the filled templates.", {"type": "object", "properties": {
        "state": {"type": "object", "description": "state_key -> the filled template (JSON object)"}},
        "required": ["state"]})
    args, cost_a, _ = call_tool(model, "dynamicmem_state", STATE_SYSTEM,
                                f"<memory>\n{memory or '(empty)'}\n</memory>\n\nKeys to fill:\n{requested}", state_spec)
    raw = (args or {}).get("state") or {}
    snapshot = {k: v for k, v in raw.items() if k in keys and isinstance(v, dict)}
    # a template may come back wrapped in its own key: {"k": {"k": {...}}}
    snapshot = {k: (v[k] if set(v) == {k} and isinstance(v[k], dict) else v) for k, v in snapshot.items()}

    items = _service_items(checkpoint)
    listed = []
    for n, (key, item) in enumerate(items):
        tmpl = f"\n  Output template: {json.dumps(item['output_template'])}" if item.get("output_template") else ""
        listed.append(f"[{n}] ({key}, {item.get('qa_id', 'q1')}) Scenario: {item.get('scenario', '')}\n"
                      f"  Request: {item.get('task_instruction', '')}{tmpl}")
    service_spec = ToolSpec("submit", "Submit one answer per request.", {"type": "object", "properties": {
        "answers": {"type": "array", "items": {"type": "object", "properties": {
            "n": {"type": "integer"}, "answer": {}}, "required": ["n", "answer"]}}}, "required": ["answers"]})
    cost_c = 0.0
    answers: dict[int, Any] = {}
    if items:
        args, cost_c, _ = call_tool(model, "dynamicmem_service", SERVICE_SYSTEM,
                                    f"<memory>\n{memory or '(empty)'}\n</memory>\n\nRequests:\n" + "\n\n".join(listed),
                                    service_spec)
        answers = {a.get("n"): a.get("answer") for a in (args or {}).get("answers", []) if isinstance(a, dict)}
    service: dict[str, dict] = {}
    for n, (key, item) in enumerate(items):
        service.setdefault(key, {"items": []})["items"].append({
            "qa_id": item.get("qa_id", "q1"), "service_family": item.get("service_family", ""),
            "answer": answers.get(n, ""), "evidence": []})
    return snapshot, service, cost_a + cost_c


def run_user(user_dir: Path, method_name: str, model: ModelAdapter, out_dir: Path,
             max_checkpoints: int | None = None) -> dict[str, Any]:
    logs, packs = load_user(user_dir)
    checkpoints = packs["checkpoints"][:max_checkpoints]
    by_index = {c["as_of"]["log_index"]: c for c in checkpoints}
    method, trace = open_method(method_name, model, out_dir)
    predictions, answer_cost = [], 0.0
    artifact = {"task_contract_version": packs.get("task_contract_version"),
                "research_frame_version": packs.get("research_frame_version"),
                "user_id": packs.get("user_id"), "method": method_name, "predictions": predictions}
    for event, last in weekly_events(logs, list(by_index)):
        method.observe(event)
        if last in by_index:
            cp = by_index[last]
            snapshot, service, cost = answer_checkpoint(TracedModel(model, trace), memory_text(method), cp)
            answer_cost += cost
            predictions.append({"checkpoint_id": cp["checkpoint_id"], "snapshot_state": snapshot,
                                "evidence": {"app_log_id": "", "evidence_content": ""},
                                "rq3_apply_answers": service})
            (out_dir / f"memory_at_{cp['checkpoint_id']}.md").write_text(memory_text(method))
            # after every checkpoint, so a run stopped by its cost cap can still be scored
            (out_dir / "tce_results.json").write_text(json.dumps(artifact, indent=1, ensure_ascii=False))
            if len(predictions) == len(checkpoints):
                break  # nothing left to score; skip replaying the rest
    trace.close()
    return {"predictions": len(predictions), "answer_cost_usd": round(answer_cost, 4)}


def evaluate_with_dynamicmem(repo: Path, user_dir: Path, out_dir: Path, judge_model: str | None) -> dict[str, Any]:
    """Score tce_results.json with DynamicMem's own evaluator; returns its headline scores.

    judge_model None runs its rule-based scoring only (no model calls)."""
    output = out_dir / "tce_eval.json"
    cmd = [sys.executable, "-m", "evaluation.eval_tce", "--benchmark", str((user_dir / "task_packs.json").resolve()),
           "--prediction", str((out_dir / "tce_results.json").resolve()), "--output", str(output.resolve()),
           *(["--enable-llm-judge", "--llm-provider", "openai", "--llm-model", judge_model,
              "--llm-judge-input-output", str((out_dir / "judge_io.json").resolve())] if judge_model
             else ["--no-enable-llm-judge"])]
    done = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, check=False)
    (out_dir / "eval_stdout.txt").write_text(done.stdout + "\n" + done.stderr)
    if done.returncode != 0 or not output.exists():
        raise RuntimeError(f"DynamicMem evaluator failed ({done.returncode}); see {out_dir / 'eval_stdout.txt'}")
    result = json.loads(output.read_text())
    summary = result.get("summary", {})
    return {"checkpoints": result.get("evaluated_checkpoints"),
            **{name: summary.get(key) for name, key in HEADLINE.items() if summary.get(key) is not None}}
