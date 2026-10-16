# ContextHarness

A research harness for the context-management and user-memory layer of **FoodBody**, a
longitudinal health and wellness agent. It replays long, dated histories of one person
through a memory design, event by event, then tests whether what the design kept is
correct, current and actually used.

The designs are compared on four testbeds:

| Testbed | What it tests |
|---|---|
| [HorizonBench](https://github.com/stellalisy/HorizonBench) | Long conversation histories in which preferences change after life events; a five-choice question at the end |
| FoodBody-style scenarios (`scenarios/`) | Health and food histories with known changes (a diagnosis, an injury, a new baby), graded against reference checklists |
| [MemConflict](https://github.com/TaoZhen1110/MemConflict) | Whether the right memory wins when memories disagree: real updates, false later claims, preferences that hold under a condition |
| [DynamicMem](https://github.com/wenyaxie023/DynamicMem) | Whether the user model stays accurate over ~15 months of app activity, and serves the user well |

Results and the reasoning behind each design change are recorded in the design log,
[`docs/MEMORY_LAYER.md`](docs/MEMORY_LAYER.md), with each round's tables in
`examples/<round>/RESULTS.txt`.

| Read | For |
|---|---|
| [`docs/MEMORY_LAYER.md`](docs/MEMORY_LAYER.md) | The designs in detail, the design log with every round's results, and known gaps |
| [`docs/DATASET_NOTES.md`](docs/DATASET_NOTES.md) | HorizonBench findings. **Read before interpreting accuracy:** picking the shortest option scores 78.2% on the benchmark |
| [`agent.md`](agent.md) | The structured agent's contract, used verbatim as its system prompt |

## Memory designs

**The current design is `profile`.** After each event, one model call returns a short
episode line, up to 8 observations, any new or changed lasting facts, and the whole
profile rewritten. The harness writes the files; the model never edits them directly.

| File | Holds | Written by | Read |
|---|---|---|---|
| `facts.md` | The facts ledger: lasting facts (identity, family, work, where they live, health, allergies, diet rules), with source and date, earlier values, and conflicting reports kept side by side | The harness, from facts the model reports. A different value replaces the old one only when the model marks a real change; otherwise both are kept as a conflict | Every update and every answer |
| `profile.md` | Who the person is now: Routines, How they want help, Goals, Right now, Changes; each claim cites observation IDs | The model rewrites it every event (8,000-character limit; over-long rewrites are shortened, and cut only as a last resort) | Every update and every answer |
| `memory/episodes.md` | The episode log: one line per event and its numbered observations (`O-12.3`) | The harness appends; nothing is edited | The latest 10,000 characters during updates; when answering, the lines most relevant to the question (the whole log if it fits, otherwise BM25 search) |

Answers and replies see the ledger, the profile and the relevant log lines, within a
48,000-character budget. Every version of the profile and ledger is kept in
`profile_versions.jsonl`.

Other designs, kept for comparison:

| `--method` | Memory | Model calls per event | Memory shown when answering |
|---|---|---|---|
| `profile` | Facts ledger, profile, episode log | 1 (plus shortening if over the limit) | Ledger + profile + relevant log lines, ≤ 48,000 chars |
| `summary` | One rolling summary rewritten each event | 1 (plus shortening if over the limit) | The summary, ≤ 8,000 chars |
| `recent` | None; keeps the raw sessions | 0 | The newest raw sessions, ≤ 48,000 chars |
| `agent` | The structured workspace in [`agent.md`](agent.md): `context.md`, state files, episodes, archive | 1–8 (file tools; `finish` ends an event) | `context.md` + state files + reads and lookups, ≤ 48,000 chars |

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q                  # 213 tests, offline, no credentials
export OPENAI_API_KEY=...         # real-model runs; the default model is gpt-5.4-mini
```

- **Cost control.** Per-token prices for gpt-5.4-mini and gpt-6.1-sol are in
  [`models/pricing.py`](src/context_harness/models/pricing.py). `--max-cost-usd`
  pauses a run once its estimated spend reaches the cap; a model needs a price there
  before a run on it can be capped.
- **Other providers.** Pass a `gemini-*` model ID (reads `GEMINI_API_KEY`) or an
  Anthropic model ID (reads `ANTHROPIC_API_KEY`) as `--model`. `--model fake` is an
  offline, deterministic stand-in for tests and smoke runs.
- **Data.** HorizonBench downloads from Hugging Face on first use (`--config sample`
  or `benchmark`). The benchmarks used here are synthetic. Check a provider's data
  terms before sending real FoodBody user data.

**MemConflict and DynamicMem** are read from local copies in `data/external/`, which
git ignores. MemConflict states no license, so its data is never copied into this
repository.

```bash
git clone --depth 1 https://github.com/TaoZhen1110/MemConflict data/external/MemConflict
git clone --depth 1 https://github.com/wenyaxie023/DynamicMem data/external/DynamicMem
uv run python -c "from huggingface_hub import snapshot_download as d; \
  d('xiewenya/dynamicmem', repo_type='dataset', local_dir='data/external/dynamicmem_data')"   # 80 MB
```

DynamicMem is scored by its own evaluator, run from that clone; it reads the same
`OPENAI_API_KEY`.

## Commands

```bash
# Run a design over histories (HorizonBench items, or a scenario file)
uv run context-harness run --method profile --dataset scenarios/foodbody_long.jsonl --items 0,1,2 --max-cost-usd 2
uv run context-harness run --method profile --config sample --items 0 --max-cost-usd 1
uv run context-harness run --method profile --model fake --config sample      # offline check
# A pause (cost cap, quota, outage) stops at the last completed event; continue with the same --run-name and --resume

# Grade finished runs' memory and replies against reference checklists (evaluation framework)
uv run context-harness evaluate runs/<run>/items/* --checklists scenarios/foodbody_long_checklists.json --max-cost-usd 2

# MemConflict: people 0-4, 30 questions each (10 per conflict type), both designs
uv run context-harness memconflict --people 0,1,2,3,4 --questions 30 --methods profile,summary --max-cost-usd 15

# DynamicMem: users, all 5 checkpoints, both designs, scored by DynamicMem's evaluator
uv run context-harness dynamicmem --users 001_user_001,002_user_002 --methods profile,summary --max-cost-usd 12

# Tables with the spread across runs, people and users
uv run python scenarios/summarize_evals.py runs/<evaluate run>/summary.json        # --by-run compares two versions
uv run python scenarios/summarize_benchmarks.py runs/<memconflict run> runs/<dynamicmem run>
```

More:

| Command | Does |
|---|---|
| `respond-eval` | Writes the reply each item's memory would give, plus replies with no memory and with the full history, and grades them against HorizonBench's labels |
| `judge-memory` | Grades a finished item's memory against HorizonBench's ground-truth preference attributes |
| `replay-answer` | Reruns only the answer step on a finished item's saved memory |
| `vet-items` | Checks that each HorizonBench item's labeled change shows up in the person's own words |
| `memory-timeline`, `inspect-memory` | What each event added, changed or removed; file access and rule violations (structured agent) |
| `summarize`, `inspect-data`, `audit-data` | Aggregate runs; inspect or audit the HorizonBench data |
| `scenarios/replay_memconflict_answers.py` | Re-answers a finished MemConflict run's questions on the same rebuilt memory, to test answer-side changes or measure answer noise |
| `scenarios/rescore_dynamicmem.py` | Rebuilds a DynamicMem run's predictions from its trace and scores them again |

Common options: `--model`, `--effort` (reasoning effort, default `medium`),
`--max-cost-usd`, `--run-name`, `--out` (default `runs/`), `--timeout`. For
benchmark commands, `--max-cost-usd` is one budget shared by memory updates, answers
and our judge; DynamicMem's evaluator runs its own judge outside it. Run
`context-harness <command> -h` for the full list.

## How a run works

```
for each item (one person's history):
    for each event, oldest first:
        append it to raw_sessions.jsonl          hash-chained; the design never reads it back
        the design updates its memory            profile: one call; agent: a tool loop; summary: one call
        checkpoint                               a pause resumes here
    answer or reply from memory alone            labels and answer keys are never shown
    score                                        evaluator side only
```

Per item, a run writes `trace.jsonl` (every model call with its usage, every memory
diff, every fact added, changed or kept as a conflict), `result.json`, the memory
before and after (`snapshots/`), and for the profile `profile_versions.jsonl`.
`run_config.json` records the settings.

## How quality is measured

- **Evaluation framework** (`memeval.py`, the `evaluate` command). Before a run, a
  person writes a reference checklist per scenario: facts and current preferences
  the memory must contain, outdated beliefs it must no longer assert, and what a good
  reply does. An LLM judge marks each item and scores six criteria (faithfulness,
  currency, recall, change tracking, provenance, consistency) against the person's
  own messages. Each judgment is made three times and the majority kept, with their
  agreement reported.
- **HorizonBench**: five-choice accuracy. Read the dataset notes first: the shortest
  option is right 78.2% of the time.
- **MemConflict**: answer accuracy per conflict type, update awareness and conflict
  recognition (the benchmark's published definitions, applied by a judge), plus
  whether the memory holds the evidence for the answer.
- **DynamicMem**: its own evaluator's Core+Detail scores for State Completion and
  Personalized Service, plus a rule-based value F1.

Judges default to gpt-5.4-mini and stay fixed when the design or model changes. They
haven't yet been checked against a person's grading.

## Repository layout

| Path | Contents |
|---|---|
| `src/context_harness/methods/` | The designs: `profile.py` (with `log_search.py`), `baselines.py` (summary, recent), `agent.py` |
| `src/context_harness/benchmarks/` | MemConflict and DynamicMem adapters |
| `src/context_harness/models/` | OpenAI, Gemini, Anthropic and fake adapters; prices and cost caps |
| `src/context_harness/` | Orchestrator, tracing, evaluation framework (`memeval.py`), judges, the structured agent's workspace, tools and checks |
| `scenarios/` | FoodBody-style scenarios and their checklists, result-table scripts |
| `docs/` | Design log and dataset notes |
| `examples/` | Each round's results (`RESULTS.txt`) and sample memories |

## Tests

| File | Covers |
|---|---|
| `test_profile.py` | One call per event, numbered evidence, saved versions; over-long rewrites shortened before any cut; the facts ledger (changes, conflicts, copied lines, survival across rewrites); log search and what answers see |
| `test_benchmarks.py` | MemConflict and DynamicMem adapters: replay order, questions answered from memory as it stood, stratified sampling, weekly events, predictions in each benchmark's format, a shared cost cap, rows kept when a run stops |
| `test_memeval.py` | The evaluation framework: the judge sees only the person's words; majority of repeats; checklists cover every scenario |
| `test_openai_adapter.py`, `test_gemini_adapter.py`, `test_anthropic_adapter.py` | Requests and responses against each SDK's types; errors that pause a run; cost and cost caps |
| `test_leakage.py` | Future sessions, the question and every label are kept from every design |
| `test_resume.py` | A pause mid-item, mid-event or mid-answer resumes to the same end state |
| `test_respond.py`, `test_replay.py`, `test_judge.py`, `test_vet.py` | Replies from memory, answer-only replay, the memory judge, item vetting |
| `test_checks.py`, `test_context.py`, `test_workspace.py`, `test_supersede.py` | The structured agent's contract rules, working context, workspace and supersession |
| `test_dataset.py`, `test_audit.py`, `test_scoring.py`, `test_scenarios.py`, `test_end_to_end.py` | Data loading and ordering, the audit, scoring, scenario files, CLI runs |
