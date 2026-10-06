# ContextHarness

A small research harness for one question: **can an agent that edits its own short-term
context and user-state files track evolving user preferences across a long,
chronological conversation?** It replays [HorizonBench](https://github.com/stellalisy/HorizonBench)
histories to the agent one dated session at a time. After the last session it asks the
benchmark's five-choice question.

Milestone 1 covers memory behavior and final-question accuracy, for the agent and two
baselines. There are no external action tools, no web access, and no approval logic. The
interfaces for those exist but are empty (see [Extension points](#extension-points)).

> **Read [`docs/DATASET_NOTES.md`](docs/DATASET_NOTES.md) before interpreting results.**
> It records how the real data differs from the intended design. Two findings matter
> most. First, in 2 of 10 sample items, the benchmark's own history contains sessions
> dated *after* the question. Second, in the sample, the correct option is the shortest
> one in 8 of 10 items.

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # creates .venv with anthropic, pyarrow, huggingface_hub, pytest
uv run pytest -q             # 53 tests, offline, no credentials
```

Real-model runs use the Anthropic API. The SDK resolves credentials from
`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, or an `ant auth login` profile. The HF
dataset is public; the `sample` config (3.3 MB) downloads on first use and is then
cached.

## Commands

```bash
# Fake-model run, fully offline (synthetic fixture with leakage canaries)
uv run context-harness run --model fake --dataset tests/fixtures/tiny_benchmark.jsonl

# Fake-model run on the real 10-item sample (no credentials; downloads the sample once)
uv run context-harness run --model fake --method agent --config sample

# Real model on a small number of sample items
uv run context-harness run --method agent --model claude-opus-5-5 --config sample --max-items 2

# Cheapest real smoke test: 1 item, only its 10 most recent sessions
uv run context-harness run --method agent --model claude-opus-5-5 --items 0 --max-sessions 10

# Baselines with identical model settings and budgets
uv run context-harness run --method summary --model claude-opus-5-5 --config sample --max-items 2
uv run context-harness run --method recent  --model claude-opus-5-5 --config sample --max-items 2

# Re-aggregate and compare runs; inspect the data's ordering anomalies
uv run context-harness summarize runs/<run-a> runs/<run-b>
uv run context-harness inspect-data --config sample
```

Useful flags: `--order text` replays the benchmark's exact text order (see A1 and A2 in
the notes). `--no-scenarios` hides the generator's `Scenario:` lines. `--resume`
continues an interrupted run. `--log-prompts` stores full request text in traces.
`--effort`, `--max-tokens`, `--context-chars`, `--file-chars`, `--answer-budget-chars`,
`--summary-chars`, `--max-update-steps` and `--max-answer-steps` set budgets. Run
`uv run context-harness run -h` for the full list.

**Cost.** The agent makes one model loop per history session. The sample averages
about 90 sessions per item (29–142), so a real run makes a few hundred calls per item.
Measure cost with the smoke-test command first. Every `result.json` records exact token
usage per phase, including cache reads and writes, so you can extrapolate from it.

## Architecture

```
HF parquet / local file
        │  dataset.py: parse with HB's parser, order sessions, split
        ▼
 BenchmarkItem ──► Episode (agent-visible)      GoldLabel (evaluator-only)
                    sessions, probe, question     correct/distractor letters, values,
        │                                         has_evolved, domain, evolution, raw id
        ▼
 orchestrator.run_item  (sequential)                          │
   for session in episode.sessions:                           │
       raw_sessions.jsonl.append(session)   (hash-chained)    │
       method.observe_session(session)                        │
   method.answer(probe, question) ──► AnswerResult            │
   ─────────── labels are read only below this line ──────────┘
   evaluation.score / reference heuristics / diagnostics ─► result.json
```

| Module | Role |
|---|---|
| `dataset.py` | Load rows (HF or local), parse sessions, apply the ordering policy, build `Episode` + `GoldLabel` |
| `hb_compat.py` | `parse_conversations` and `extract_letter` vendored verbatim from HB (pinned commit) |
| `models/` | `ModelAdapter` protocol; `AnthropicAdapter` (all provider code); `FakeModel`/`ScriptedModel` |
| `workspace.py` | The 8 allowlisted files, templates, per-file character limits, unified diffs |
| `tools.py` | Tool specs, validating `ToolExecutor`, `ApprovalGate` seam, `submit_answer` |
| `methods/` | `agent` (workspace agent), `recent` (recent-history baseline), `summary` (rolling summary) |
| `prompts.py` | All prompt text and rendering, in one auditable place |
| `tracing.py` | JSONL trace, token accounting wrapper, immutable raw session log |
| `evaluation.py` | Scoring, reference heuristics, evolution-session diagnostic, aggregation |
| `orchestrator.py`, `cli.py` | Sequential runner and command line |

### The agent's context design

**Every model call is a fresh single-turn request** built from explicit state. The
harness never grows a multi-turn tool transcript.

```
system  (fixed per phase; cache breakpoint)
user:   [ <session> transcript ]                (fixed within the session; cache breakpoint)
        [ <working_context> = context.md now ]  (re-read from disk on every call)
        [ <workspace> file list with sizes ]
        [ <actions_so_far> tool calls + results for this session ]
```

- **`context.md` is the bounded working context.** It is the only file injected
  automatically, and it is re-read on every call. An edit therefore takes effect in the
  very next invocation and replaces the old content rather than piling up beside it.
  Writes over its limit (default 8,000 chars) are rejected with a message the agent sees.
- **Longer-term files are reached only by explicit reads.** These are
  `long_term_memory.md` and `user_state/{goals, persistent_state, transient_state,
  progress, observations, hypotheses}.md`, each limited to 20,000 chars by default.
  Their templates describe each file's purpose and contain no user content.
- **Tools.** During replay: `list_files`, `read_file`, `write_file`, `edit_file`,
  `append_file`. At question time: read-only tools plus `submit_answer`. The paths are
  an enum of the 8 files, so traversal, the raw log, and the traces are unreachable.
  Doing nothing is a valid response: the model can reply without calling a tool.
- **Why single-turn steps.** The model's input is exactly what the trace records. A
  `context.md` edit can replace the old version instead of sitting next to it in
  history. And never rewriting earlier turns keeps prompt caching effective, which is
  compatible with "preserved thinking" on current Claude models.
- **No domain rules.** Nothing writes preferences on the agent's behalf. The only
  guidance, which the summary baseline receives too, is generic: preferences can
  change, so record what is current and mark what is superseded.

### Leakage boundary

Methods receive `Session`, then `Probe` and `Question` objects only. Option `value`s,
gold letters, `has_evolved`, `preference_domain`, `preference_evolution`, and the raw
`id` all stay in `GoldLabel`. The raw `id` encodes the tested domain, so the agent sees
an opaque `item_key`. The probe (the truncated final conversation the options respond
to) and the question are shown only after replay. `tests/test_leakage.py` plants canary
strings in each of these fields and checks every request sent to the model.

### Baselines and comparable budgets

All methods share one `ModelSettings` (model, `max_tokens`, `effort`, fallbacks) and one
answer phase: the same `submit_answer` tool, with `extract_letter` as a text fallback.

| Budget (default) | agent | summary | recent |
|---|---|---|---|
| Always-visible memory | `context.md` ≤ 8,000 chars | summary ≤ 8,000 chars (truncated) | — |
| Total memory text at answer time (`--answer-budget-chars` 48,000) | `context.md` + reads ≤ 48,000 | ≤ 8,000 | most recent sessions ≤ 48,000 |
| Model calls per history session | ≤ 8 (`--max-update-steps`) | 1 | 0 |
| Answer-phase calls | ≤ 6 | ≤ 6 | ≤ 6 |

Default model: `claude-opus-5-5`, `effort=medium`, `max_tokens=16000`. Opus 5.5 does not
accept `temperature`, so runs are not greedy, unlike HB's `temperature=0` protocol.
Server-side refusal fallback (`fallbacks="default"`) is **on** by default, which keeps
an item from failing on a classifier refusal; turn it off with `--no-fallbacks`. Each
call's serving model is logged, and `stats.served_by` shows whether any fallback ran.
Everything is written to `run_config.json`, and `--resume` refuses to mix settings.

### Outputs

```
runs/<run>/
  run_config.json        model settings, budgets, dataset policy, HB commit
  results.jsonl          one evaluator record per item
  summary.json           aggregates (below)
  items/<idx>_<user>_<key>/
    workspace/           the agent's files (agent-visible)
    raw_sessions.jsonl   immutable, hash-chained observation log (read-only)
    trace.jsonl          item_start, session_start/end, model_call, tool_call,
                         file_diff, update_loop_end, question_start, answer, evaluation
    snapshots/initial/   templates; snapshots/final/ end state (or summary / window)
    result.json          answer, correctness, tokens, file ops, latency, diagnostics
```

`summary.json` reports:
- accuracy (bootstrap 95% CI) overall, for evolved items, for static items, and by generator;
- invalid answers and errors;
- the old-preference distractor rate, both among evolved items and as a share of errors (chance 25%);
- the shortest-option reference accuracy;
- for the agent, how often it wrote memory on the date of a ground-truth evolution event;
- mean file ops and reads, token totals, model calls, and latency.

## Results example

These numbers come from `--model fake` on the real 10-item sample. They check the
pipeline and are **not** model results. The fake chooses by word overlap between
options and memory, which favors long options, and the sample's correct options are
mostly the shortest ones; hence 0%. Files: [`examples/results_example/`](examples/results_example)
(summary, run config, one item's `result.json`, a trace excerpt, and the final workspace).

```
================================================================
  agent / fake  (sample_fake_agent)
================================================================
  Overall            0.0%  [0.0, 0.0]     (n=10)
  Evolved            0.0%  [0.0, 0.0]     (n=6)
  Static             0.0%  [0.0, 0.0]     (n=4)
  Chance            20.0%
  Shortest-option   80.0%  (label-free reference heuristic)
  Errors: 0   Invalid answers: 0
  Old-preference distractor chosen:  33.3% of 6 evolved items;  33.3% of 6 errors (chance 25.0%)
  Wrote memory on an evolution-event date: 100.0% of evolved items
  Mean file ops/item: 173.9   mean file reads/item: 1
  Model calls: 1822   tokens: {'input_tokens': 4842501, 'output_tokens': 122855, ...}
================================================================
```

## Tests

`uv run pytest -q` runs these suites:
- `test_dataset.py`: chronological ordering, including stable ties, post-probe drops, the `text` policy, and the real sample when it is cached.
- `test_leakage.py`: canaries for future sessions, the question, and labels, for all three methods; long-term files stay out of prompts until read.
- `test_context.py`: `context.md` edits replace the working context on the next call, across sessions and into the answer phase; doing nothing is allowed; the limit is enforced.
- `test_workspace.py`: allowlist, traversal, symlinks, per-item isolation, diffs, approval denial, and tamper detection on the raw log.
- `test_scoring.py`: letter extraction, scoring, aggregation, text fallback, and invalid answers.
- `test_end_to_end.py`: CLI runs for every method, determinism, and resume.
- `test_anthropic_adapter.py`: request parameters bind to the installed SDK signatures; tool-use, usage, refusal and fallback parsing on SDK-typed responses.

## Extension points

- **Domain tools:** subclass `tools.Tool`, set `requires_approval = True`, and pass an
  `ApprovalGate` to `ToolExecutor`. The default `NoApprover` denies every such call.
- **Human approval:** implement `ApprovalGate.review(call, tool) -> ApprovalDecision`.
- **Providers:** implement `ModelAdapter.complete(ModelRequest) -> ModelResponse`.
  Requests are provider-neutral and single-turn.
- **Methods:** implement `observe_session`, `answer`, and `memory_snapshot`, then
  register the class in `methods/__init__.py`.

## Known limitations and next steps

- **Memory evaluation is deterministic for now.** It counts file ops per session,
  sizes, and writes on the date of each evolution event. An evaluator-side LLM judge of
  the final memory against `preference_evolution` is the natural next step.
- **The length bias and the post-probe sessions are verified on the sample only.**
  Checking how widespread both are in the 4,245-item benchmark needs the ~1.8 GB download.
- **Replay is sequential and costs one loop per session.** Message Batches, or
  parallelism across items, would cut wall-clock time for larger runs.
- **The sample is 10 items.** Confidence intervals are wide at this size, so
  comparisons between methods need the full benchmark.
