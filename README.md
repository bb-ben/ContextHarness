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

**Status:** contract **v3**. The layer, instrumentation, and Gemini and OpenAI
adapters are built and tested offline (143 tests). A model that follows the
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

| Read | For |
|---|---|
| [`agent.md`](agent.md) | The memory contract: what goes in each file, the entry format, how beliefs form, change and expire. It is the agent's system prompt, verbatim. |
| [`docs/MEMORY_LAYER.md`](docs/MEMORY_LAYER.md) | Layer design, the per-call prompt, every check, pause/resume, and the FoodBody extension points |
| [`docs/DATASET_NOTES.md`](docs/DATASET_NOTES.md) | HorizonBench findings. **Read before interpreting accuracy:** picking the shortest option scores 78.2% on the benchmark |

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q                     # 143 tests, offline, no credentials
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
- `--layout general|with-decisions`: the file set and seed preference dimensions
  (`general` is the default; `with-decisions` adds the decisions file).
- `--max-cost-usd`: pauses once estimated spend reaches the cap (priced models).

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
  Overall           10.0%  [0.0, 30.0]    (n=10)
  Shortest-option   80.0%  (label-free reference heuristic)
  Memory layer: 901 events, 0.00 violations/event (  0.0% of events); top: none
  Final provenance 100.0%; context covers  17.4% of live state
  Evolution evidence reached: in_state 10
```

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

## Next steps

1. **Run item 0 under v2** with Gemini. Read its `inspect-memory` output and trace,
   especially the life event at S-14 and whether the old preference is superseded.
   Then adjust `agent.md`.
2. **Build a length-balanced evaluation subset** before comparing accuracy (see the
   dataset notes).
3. **Add an LLM judge** for meaning-level contradictions and for whether the final
   state matches the true preference.
4. **Add a FoodBody event renderer and decision/outcome statuses** once a schema is
   supplied.
