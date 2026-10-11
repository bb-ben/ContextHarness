# ContextHarness

The context-management and user-state layer for **FoodBody**, a longitudinal health
and wellness agent, developed and tested on
[HorizonBench](https://github.com/stellalisy/HorizonBench).

The layer keeps three kinds of memory across sessions: a bounded **working context**,
**episodic memory** of what happened, and the **current user state**. The agent
maintains all three itself, through a small set of file tools, under a written
contract. HorizonBench supplies long, dated conversation histories in which
preferences change after life events. That makes it a sequential testbed for
whether the layer keeps the user's state current.

**Status:** contract **v3.10**. The layer, instrumentation, and Gemini and OpenAI
adapters are built and tested offline (193 tests). A model that follows the
contract produces zero violations across all 901 events of the HorizonBench sample.

**Real runs so far:**
- **v1 (Gemini, 3½ events).** It followed the format and promoted a hypothesis
  correctly. It also showed three problems, which v2 fixed; see
  [`examples/real_run_v1_gemini/`](examples/real_run_v1_gemini).
- **v2 (gpt-5.4-mini, one full item, $1.02).** Memory stayed well-formed and
  bounded, but preferences only accumulated: every signal got a new topic, so none
  was ever superseded and the change in the person was missed. See
  [`examples/real_run_v2_openai/`](examples/real_run_v2_openai).
- **v3 (same item, $1.33).** Preference dimensions fixed the accumulation: all
  preferences sat on 7 seed dimensions, 4 were superseded, and the life event was
  linked to the decision-style value and revised it. The answer was still wrong,
  because the answering step leaned on the top of `context.md` rather than the
  revised value; see
  [`examples/real_run_v3_openai/`](examples/real_run_v3_openai).
- **Baselines on that item ($0.44)** are wrong too: the full history in one call
  answered D B B, recent history B B B, and the rolling summary A. The item is hard
  for this model whatever memory it gets; see
  [`examples/baselines_item0/`](examples/baselines_item0).
- **v3.1 on three items ($14.79).** No method, ours or a baseline, answers any of
  the three correctly. One item's label contradicts its own conversations, and
  another's options are ambiguous. On the valid one (item 6), the rolling summary
  holds the person's changed preferences better than our agent, because v3.1 filed
  a preference repeated across many tasks under each task's goal. No memory recorded
  any change. See [`examples/multi_item_v31/`](examples/multi_item_v31) and the
  [design log](docs/MEMORY_LAYER.md#v31-on-three-items).
- **v3.2 on item 6 ($5.85).** Preferences repeated across tasks are now collected,
  and goals close (1 active at the end, against 24). The memory scores the same,
  0.3, and still records no change: each revision of a preference merges new clauses
  into the old ones instead of replacing what changed. See
  [`examples/v32_item6/`](examples/v32_item6).
- **v3.3 on two FoodBody-style scenarios ($1.13 with the reply tests).** The agent
  recorded its first changes: it replaced "loose ideas, no numbers" with "structured
  weekly plans with carb counts" and noted what changed. Replies written from its
  memory used carb counts, which replies without memory never did. A rolling summary
  did as well or better on these short histories, at a tenth of the cost. See
  [`examples/v33_scenarios/`](examples/v33_scenarios) and the
  [design log](docs/MEMORY_LAYER.md#v32--v33-revisions-that-replace-and-checking-that-memory-is-used).
- **v3.4–v3.6 on four scenarios ($2.25).** Two new scenarios end on a neutral
  message, so replies without memory fit 0% of the changes and replies from memory
  50–100%. v3.4 and v3.5 kept the old preference standing behind temporary entries
  that expired; v3.6 gives how someone wants help a single home (preferences, with a
  review date if the change may not last), and both changes landed. Replies now use
  memory directly ("for you and your partner, Mediterranean-leaning, no grams"). A
  rolling summary still writes better-fitting replies on these short histories. See
  [`examples/scenarios_v34_v36/`](examples/scenarios_v34_v36) and the
  [design log](docs/MEMORY_LAYER.md#v34--v36-one-home-for-how-they-want-help).
- **v3.7–v3.8 on four scenarios ($1.78).** Repeated wishes are collected under a
  dimension and replies must account for all of memory; with both, replies from
  memory fit the changed preferences in 7 of 8 cases and never the old ones (no
  memory: 0 on the neutral-ending scenarios). A `foodbody` layout keeps diet,
  allergies and household in view. Run-to-run variance is now the main limit. See
  [`examples/scenarios_v37_v38/`](examples/scenarios_v37_v38).
- **v3.8 on item 6, 75 events ($7.44 with evaluations).** Memory still scores 0.3
  and replies follow the old style: the replies used memory faithfully, but every
  revision of the style value added to it ("now also…") instead of replacing it. What
  works on 8-event scenarios erodes over 75. See
  [`examples/v38_item6/`](examples/v38_item6).
- **v3.9–v3.10 on three scenarios, 54 runs (about $15 with vetting).** Reasoning
  effort didn't matter; run-to-run variance did. A review call after each update,
  where the agent re-reads its memory and fixes contradictions, cut stale values and
  raised replies that follow the new preferences from 0.67 to 0.83, at 63% more cost.
  See [`examples/v310_review_runs/`](examples/v310_review_runs).
- **The profile method ($2.97).** One profile rewritten each event, citing numbered
  evidence, under a 1,781-character prompt instead of the 25,200-character contract.
  Replies follow the new preferences 0.94 of the time on the neutral-ending scenarios
  (structured agent with review step: 0.83; rolling summary: 0.83), at about a tenth of
  the structured agent's cost. See [`examples/profile_method/`](examples/profile_method).
- **Memory judge on that item.** Graded against the ground-truth attributes, v3
  holds 0.38 of the person's current preferences, v2 0.0, and the rolling summary
  0.75. The summary's lead comes from role-play content recorded before the change,
  though, and no memory recorded the change itself; see
  [`docs/MEMORY_LAYER.md`](docs/MEMORY_LAYER.md#memory-judge).

| Read | For |
|---|---|
| [`agent.md`](agent.md) | The memory contract: what goes in each file, the entry format, how beliefs form, change and expire. It is the agent's system prompt, verbatim. |
| [`docs/MEMORY_LAYER.md`](docs/MEMORY_LAYER.md) | Layer design, the per-call prompt, every check, pause/resume, and the FoodBody extension points |
| [`docs/DATASET_NOTES.md`](docs/DATASET_NOTES.md) | HorizonBench findings. **Read before interpreting accuracy:** picking the shortest option scores 78.2% on the benchmark |

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q                     # 193 tests, offline, no credentials
export GEMINI_API_KEY=...            # for real-model runs (default model: gemini-3.7-flash)
```

Your free-tier limits are on your [AI Studio rate-limit dashboard](https://aistudio.google.com/rate-limit).
On the free tier, Google may use prompts and outputs to improve its products. That's
fine for HorizonBench's synthetic data, but **not for real FoodBody user data**,
which needs a paid tier or another provider. Anthropic models also work: pass an
Anthropic model ID as `--model`.

## Commands

```bash
# Offline runs with the deterministic fake model
uv run context-harness run --model fake --dataset tests/fixtures/tiny_benchmark.jsonl
uv run context-harness run --model fake --config sample

# Real-model demo: sample item 0, all 29 events, paced and capped for the free tier
uv run context-harness run --model gemini-3.7-flash --items 0 --rpm 5 --timeout 240 --max-requests 40 --run-name demo_item0
# A daily quota, provider outage or the request cap pauses it at the last completed event. Later:
uv run context-harness run --model gemini-3.7-flash --items 0 --rpm 5 --timeout 240 --max-requests 40 --run-name demo_item0 --resume

# Quicker smoke test: only the 16 most recent events (still includes item 0's life event)
uv run context-harness run --model gemini-3.7-flash --items 0 --max-sessions 16 --rpm 5

# Rerun only the answer step against a finished item's saved memory (about one call each)
uv run context-harness replay-answer runs/<run>/items/<item> --model gpt-5.4-mini \
    --strategy default --strategy dimension-first --repeats 5 --max-cost-usd 0.50

# Grade a finished item's memory against the ground-truth preference attributes
# (three judgings, majority grade; evaluator side only)
uv run context-harness judge-memory runs/<run>/items/<item> --max-cost-usd 0.30

# Short FoodBody-style histories with one known change (cents per run)
uv run context-harness run --model gpt-5.4-mini --dataset scenarios/foodbody.jsonl --max-cost-usd 0.80

# Write the reply the agent would send, from its saved memory, and compare it with
# replies from no memory and the full history; a judge grades each against the labels
uv run context-harness respond-eval runs/<run>/items/<item> --condition memory \
    --condition none --condition full-history --repeats 2 --max-cost-usd 0.40

# Check that each item's labeled change shows up in the person's own words
uv run context-harness vet-items --config sample --items 0,2,3,5,6,7 --max-cost-usd 0.40

# Inspect what the agent read and wrote, event by event, and every violation
uv run context-harness inspect-memory runs/<run>/items/<item>

# Baselines on the same model settings and budgets
uv run context-harness run --method summary --model gemini-3.7-flash --items 0 --rpm 5
uv run context-harness run --method recent  --model gemini-3.7-flash --items 0 --rpm 5

# Aggregate runs; inspect or audit the dataset
uv run context-harness summarize runs/<run-a> runs/<run-b>
uv run context-harness inspect-data --config sample
uv run context-harness audit-data --config benchmark --out data/audit_benchmark
```

Other options:
- `--effort minimal|low|medium|high`: Gemini thinking level.
- `--context-chars`: the `context.md` limit (default 6,000).
- `--no-preload`: contract v1 behavior, where state files must be read explicitly.
- `--max-requests`: a hard cap per invocation (the run pauses).
- `--timeout`: seconds per response before a retry.
- `--layout general|with-decisions|foodbody`: the file set and seed preference
  dimensions (`general` is the default; `with-decisions` adds the decisions file;
  `foodbody` adds profile topics such as diet and allergies).
- `--max-cost-usd`: pauses once estimated spend reaches the cap (priced models).
- `--review-step`: after each update, one more short call in which the agent re-reads
  its memory and fixes contradictions (about 60% more cost; raised reply fit from
  0.67 to 0.83 on the scenarios).
- `--no-check-feedback`: don't show the agent the memory checks' open problems.
- `--answer-strategy dimension-first|default`: how the agent uses memory when it
  answers. `dimension-first` (the default) finds the dimension the candidate
  responses differ on and reads its current value first.

Two commands show how memory evolved:
- `context-harness memory-timeline <item>`: what each event added, changed,
  archived or removed in working context, episodic memory and state.
- `context-harness inspect-memory <item>`: file access and violations.
- `--order text`: the benchmark's original order (see the dataset notes).
- `--no-scenarios`, `--log-prompts`.

Run `context-harness run -h` for the full list.

## How a run works

```
for each item:
    for each event (a HorizonBench session), oldest first:
        append it to raw_sessions.jsonl                 hash-chained, read-only to the agent
        agent loop: fresh single-turn calls, ≤ 8        system = agent.md; ends on finish
        memory checks on the workspace                  access, provenance, consistency, survival
        checkpoint                                      a quota pause resumes here
    answer the five-choice question (memory read-only)
    score; trace the life event's evidence chain        answer key read only here
```

`run_config.json` records the run's settings and `agent.md`'s hash. Per item, the
run writes:

| File | Contents |
|---|---|
| `workspace/` | The agent's files |
| `trace.jsonl` | Every model call, tool call, file diff and memory check |
| `memory_report.json` | Per-event access table, violations, metrics, entry lifecycles |
| `snapshots/initial`, `snapshots/final` | The workspace before and after |
| `result.json` | Answer, correctness, tokens, file operations, latency, memory summary, evidence chain |
| `checkpoint.json` | Only present while an item is paused |

### Baselines and comparable budgets

| Budget | agent | summary | recent |
|---|---|---|---|
| Always-visible memory | `context.md` ≤ 6,000 chars | summary ≤ 8,000 | — |
| Memory text at answer time (48,000) | `context.md` + reads / lookups | the summary | newest raw sessions |
| Model calls per event | 1–8 (`finish` ends an event) | 1 | 0 |

All three methods share one model configuration and one answer step: a
`submit_answer` tool, with HorizonBench's letter extraction as a fallback.

## Results example

Fake model, all 10 sample items. These numbers check the pipeline and are **not
model results**. The fake follows the contract perfectly and answers by word
overlap. Files are in [`examples/results_example/`](examples/results_example).

```
  Overall            0.0%  [0.0, 0.0]     (n=10)
  Shortest-option   80.0%  (label-free reference heuristic)
  Memory layer: 901 events, 0 distinct violations (new ones in   0.0% of events, 0 unresolved at end); top: none
  Final provenance 100.0%; context covers  17.4% of live state
  Evolution evidence reached: in_state 10
```

The fake's accuracy moves whenever the prompt's wording does (it was 10% before
v2), so it says nothing about the layer.

## Tests

| File | Covers |
|---|---|
| `test_checks.py` | Every contract rule fires on a workspace that breaks it and stays quiet on one that follows it |
| `test_resume.py` | A pause mid-item, mid-event or mid-answer resumes to the same end state as an uninterrupted run |
| `test_context.py` | `agent.md` is the system prompt; `context.md` edits replace it on the next call; `finish` ends an event in one call; due reminders appear on time, and ignoring them is flagged |
| `test_workspace.py` | Layout, allowlist, traversal, symlinks, limits, `lookup`, `archive`, access records, approval denial, raw-log integrity |
| `test_gemini_adapter.py` | Requests and responses against the SDK's types; thinking text excluded; per-minute vs. per-day 429s; pacing |
| `test_leakage.py` | Future sessions, the question and every label are kept from the agent, for all methods |
| `test_dataset.py`, `test_audit.py` | Chronological ordering, the guarded parser, the dataset audit |
| `test_scoring.py`, `test_end_to_end.py`, `test_anthropic_adapter.py` | Scoring and aggregation; CLI runs; the Anthropic adapter |
| `test_judge.py` | The judge grades beliefs, not raw episodes; missing grades count as absent; repeats combine by majority |
| `test_vet.py` | Vetting reads only the person's turns, split at the change; verdicts separate memory grading from answer scoring |
| `test_supersede.py` | `supersede` links both entries, refuses revisions without `changed:` or over the length limit, and leaves files untouched when it refuses |
| `test_respond.py` | Replies from memory (no options shown), from text or nothing; the reply judge's majority; answers record the entries they rest on |
| `test_scenarios.py` | The scenario file matches its generator, and every scenario's labels are coherent |
| `test_replay.py` | Answer-only replay restores the saved memory, reruns only the answer step, applies the strategy, and leaves the source run untouched |

## Next steps

1. **Make the profile method the main line**: more scenarios and repeated runs on
   long histories, and decide which of the structured design's pieces (if any) to keep.
2. **Measure variance:** run each scenario several times per version and compare
   distributions, not single runs.
3. **Add more neutral-ending scenarios.** They are the only tests so far where replies
   without memory fail.
4. **Score answers only on `usable` items** (0, 2 and 7 in the sample), and grade
   memory on `memory_only` items too.
5. **Check the judge against a person's reading** of a few memories, and try a
   stronger judge model.
6. **Add a FoodBody event renderer and decision/outcome statuses** once a schema is
   supplied.
