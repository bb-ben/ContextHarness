"""Answer a finished MemConflict profile run's questions again on the same memory.

Only the answer step changes, so an answer-side change can be tested without replaying
50 sessions. The memory at each question is rebuilt from the run: profile and facts
ledger from profile_versions.jsonl, episode-log lines from the record calls in the trace.

Usage:
  python scenarios/replay_memconflict_answers.py --heading new --out runs/replay_new_p0 \
      runs/step3_mc_profile_p0/profile/person_0 [--max-cost-usd 2]
--heading old uses the log heading the step-3 runs had, to measure answer-to-answer noise;
new used "History and details. For what is true now, the lasting facts and the profile
above come first" (tested 2026-10-14: no effect, 0.60 vs 0.61; reverted).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from context_harness.benchmarks import memconflict as mc  # noqa: E402
from context_harness.methods import profile  # noqa: E402
from context_harness.models import ModelSettings, make_model  # noqa: E402
from context_harness.models.pricing import CostCap  # noqa: E402

OLD_HEADING = "\n\n## From the episode log: the observations most relevant here (oldest first)\n"
NEW_HEADING = ("\n\n## From the episode log: earlier observations most relevant here (oldest first)\n"
               "History and details. For what is true now, the lasting facts and the profile above come "
               "first; use these lines for details they don't hold.\n")


def memories(run: Path):
    """For each event index: (facts, profile, episode-log lines) as the method held them."""
    versions = {json.loads(line)["event_id"]: json.loads(line) for line in (run / "profile_versions.jsonl").open()}
    records = {}
    for line in (run / "trace.jsonl").open():
        r = json.loads(line)
        if r.get("event") == "model_call" and r.get("phase") == "profile_update" and r.get("tool_calls"):
            records[r["event_id"]] = r["tool_calls"][0]["arguments"]
    episodes, facts, text = [], "", ""
    for k in range(max(int(e.split("-")[1]) for e in versions) + 1):
        v = versions.get(f"S-{k}")
        if v:
            facts, text = v.get("facts", ""), v["profile"]
        args = records.get(f"S-{k}")
        date = v["date"] if v else ""
        episodes.append(f"- E-{k} | {date} | source: S-{k} | {(args or {}).get('episode', '(not recorded)').strip()}")
        obs = [o for o in (args or {}).get("observations", []) if isinstance(o, dict)][:profile.MAX_OBSERVATIONS]
        for n, o in enumerate(obs, 1):
            kind = o.get("kind") if o.get("kind") in profile.KINDS else "behavior"
            episodes.append(f"  - O-{k}.{n} | {kind} | {str(o.get('text', '')).strip()}")
        yield k, facts, text, list(episodes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="runs/<run>/profile/person_0")
    ap.add_argument("--heading", choices=["old", "new"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-cost-usd", type=float, default=2.0)
    args = ap.parse_args()
    profile.LOG_HEADING = OLD_HEADING if args.heading == "old" else NEW_HEADING
    model = CostCap(make_model(ModelSettings(model="gpt-5.4-mini", effort="medium")), args.max_cost_usd)
    judge = CostCap(make_model(ModelSettings(model="gpt-5.4-mini", effort="high")), args.max_cost_usd, shared_with=model)
    rows = [json.loads(line) for line in (args.run / "questions.jsonl").open()]
    by_event = {k: (facts, text, eps) for k, facts, text, eps in memories(args.run)}
    out = args.out / "profile" / args.run.name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "questions.jsonl").open("w") as f:
        for r in rows:
            facts, text, eps = by_event[r["session"]]
            memory = profile.profile_memory_text(facts, text, eps, r["question"])
            q = {"question_id": r["question_id"], "question": r["question"], "answer": r["reference"],
                 "conflict_type": r["conflict_type"], "difficulty": r.get("difficulty")}
            row = {"person": r["person"], "session": r["session"], "date": r["date"],
                   **mc.answer_and_grade(model, judge, memory, q)}
            f.write(json.dumps(row) + "\n")
            f.flush()
    print(f"{args.out}: {len(rows)} questions, accuracy {mc.summarize([json.loads(x) for x in (out / 'questions.jsonl').open()])['answer_accuracy']}, spent ${model.spent:.2f}")


if __name__ == "__main__":
    main()
