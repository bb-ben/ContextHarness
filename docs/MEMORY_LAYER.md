# The memory layer

This repository builds the context-management and user-state layer for a
longitudinal health and wellness agent (FoodBody). HorizonBench is the current
testbed. It supplies long, dated, sequential conversations in which preferences
change after life events.

The layer has three parts:

| Part | Where | What it does |
|---|---|---|
| **Contract** | [`agent.md`](../agent.md) | What belongs in each file, the entry format, and how beliefs form and change. Loaded verbatim as the agent's system prompt, so the documented rules and the model's instructions are the same text. |
| **Workspace** | `workspace.py`, `tools.py` | The files, their limits, and the only operations the agent has on them. |
| **Instrumentation** | `entries.py`, `checks.py`, `evaluation.py` | Parses the files after every event and measures access, provenance, consistency and survival. |

## Three kinds of memory

| Kind | Files | Shown to the agent |
|---|---|---|
| Short-term working context | `context.md` (6,000 chars) | In full, re-read before every call |
| Long-term episodic memory | `memory/episodes.md` (one `E` line per event, with `O` observations beneath), `memory/archive.md` (closed entries and rolled-over episodes) | By `lookup` (or `read_file`) |
| Current user state | `state/profile.md`, `preferences.md`, `goals.md`, `transient.md`, `hypotheses.md` | In full, re-read before every call (contract v2; `--no-preload` restores v1, where these too need a read) |

`context.md` is a working set and an index: a short briefing on what matters right
now. Every bullet cites the ID of a live state entry or hypothesis; it never copies
one.

Evidence flows **event → observation → hypothesis → state**:
- A belief changes by adding a new entry that supersedes the old one. The old entry
  stays, marked `superseded`.
- Closed entries can move to the archive. They are never deleted.
- The layer itself does only three things on the agent's behalf:
  - it lists transient entries whose expiry or review date has arrived, under
    `<due>`;
  - it moves entries to the archive when asked;
  - it rolls the oldest episode blocks into the archive, unchanged, once the episode
    log passes 80% of its limit.

## What one model call contains

```
system   agent.md
user     <event id="S-14" type="conversation" date="2025-10-15"> … </event>   (cached for the event's loop)
         <working_context> context.md now </working_context>
         <state> the five state files, in full </state>
         <preference_dimensions> each dimension, its current value, open challenges </preference_dimensions>
         <workspace> every file's size / limit </workspace>
         <due> transient entries now due, verbatim </due>
         <actions_so_far> this event's tool calls and results </actions_so_far>
         instruction + calls remaining
tools    list_files, read_file, lookup, append_file, edit_file, write_file, archive, finish
```

Each call is a fresh single-turn request. A `context.md` edit replaces the old
version in the next call, and nothing persists between events except the files.
`finish` lets the agent end an event in the same turn as its edits. With state
already visible, most events take one call.

## Instrumentation

After every event, `MemoryMonitor` parses the workspace and records which files were
read and written. It then runs the checks below. Results go to the item's trace (a
`memory_check` record per event) and to `memory_report.json`. To see them, run
`context-harness inspect-memory <item_dir>`.

| Question | Check codes |
|---|---|
| Does each entry follow the format? | `unparsed_line`, `unparsed_entry`, `wrong_file`, `orphan_observation`, `duplicate_id`, `bad_status`, `bad_kind` |
| Are preferences disciplined? | `undeclared_dimension` (a topic outside the seed list with no `new_dimension:` reason), `near_duplicate_dimension`, `unsupported_promotion` (a preference resting on non-stated evidence from one event, with no confirmed hypothesis), `unused_stated_observation` (a `stated` observation cited by no state entry, hypothesis or goal note) |
| Does each entry have a source and date? | `future_id` (ID names an event that hasn't happened), `bad_date`, `date_mismatch` (date differs from its event's real date), `bad_source`, `parent_mismatch`, `missing_basis`, `dangling_ref`, `basis_not_observation`, `missing_topic`, `missing_end_date` |
| Do current-state files contradict each other? | `topic_conflict` (two `current` entries for one topic), `incomplete_supersession`, `supersession_mismatch`, `unresolved_confirmation`, `overdue_transient`, `archived_live`, `context_stale`, `context_uncited`, `context_dangling`, `context_inactive` |
| Does information survive across events? | `missing_episode` (an event left no `E` line), `entry_deleted`, `history_rewritten` (statement, date or kind changed, or basis lost), `status_reverted`; plus each entry's lifecycle (created, status changes, deleted) |
| Which files are read and edited? | Per-event, per-file reads, writes and errors, from tool calls |

These checks are deterministic and never shown to the agent. Two metrics summarize
each snapshot:
- `provenance_share`: the share of state entries and hypotheses that are dated, match
  their event, and cite evidence that exists;
- `context_coverage_of_live_state`: the share of live state that `context.md` cites.

One more measure is evaluator-side because it uses the answer key: the **evidence
chain**. For each true life event, it reports how far the evidence got:
- `not_replayed`
- `not_recorded`
- `recorded` (an episode line)
- `observed`
- `hypothesized`
- `in_state`

A fake model that follows the contract produces zero violations on real
HorizonBench items. `tests/test_checks.py` shows each rule firing on a workspace
that breaks it.

## Long runs on a daily quota

After every completed event the orchestrator writes `checkpoint.json`: the
workspace, the method's state, the stats, the memory-check history and the elapsed
time.

When a provider reports a per-day quota error, the adapter raises `QuotaExhausted`.
The run then pauses:
- the interrupted event's partial edits are discarded;
- the raw log is cut back to its last verified record;
- `--resume` continues from the next event.

Per-minute limits are handled differently: the run waits for the provider's
suggested delay, and `--rpm` paces requests.

## Extension points for FoodBody

No FoodBody data schema has been supplied, so none is assumed. These seams exist:

- **Events.** The layer consumes anything implementing `types.Event`: `event_id`,
  `event_type` and `date`. Supporting a new source means writing a renderer in
  `prompts.render_event`. The workspace, tools and checks don't depend on the event
  type.
- **Decisions and outcomes.** The `with-decisions` layout adds `memory/decisions.md`.
  `D` entries are the agent's recommendations or actions, with the evidence behind
  them. `R` entries are what followed, linked by `decision:`. Their statuses are not
  yet specified.
- **Action tools.** Subclass `tools.Tool` with `requires_approval = True`. The
  executor routes such calls through an `ApprovalGate`; the default approves nothing.
- **Layouts.** File sets and limits are data (`workspace.Layout`), so a deployment can
  add files without code changes.

## Design log

**v1 → v2 (2026-10-09).** The first real-model run (Gemini 3.7 Flash, 3½ events; see
[`examples/real_run_v1_gemini/`](../examples/real_run_v1_gemini)) followed the format
with zero violations. It also promoted a hypothesis to a preference across two
events, exactly as specified. Three problems drove v2:

| Observed in v1 | Change in v2 |
|---|---|
| The agent read every state file in every event and never used `lookup`, so each event cost two calls | State files are pre-loaded into every call; episodes and the archive stay on demand. In the answer phase the pre-loaded text counts against the shared 48,000-character budget |
| A stated preference became only a hypothesis, while a one-off request was tagged `stated` | Evidence kinds redefined with worked examples, including lasting vs. situational |
| About 730 chars of observations per event would fill the episode log at around event 80 | A "worth recording" rule (zero to two observations per event, about 150 chars each), plus layer-driven rollover of old episodes into the archive |

**v2 → v3 (2026-10-09).** The first full real run (gpt-5.4-mini, all 29 events and
the answer, $1.02; see [`examples/real_run_v2_openai/`](../examples/real_run_v2_openai))
kept memory well-formed and bounded. Its core finding: **preferences only
accumulated**. The agent coined a new topic for nearly every signal (11 preferences,
11 topics), so new evidence never met an old value and nothing was superseded. Several
later "new" preferences were in fact the person's change, filed as unrelated facets.
The answer was driven by the newest one, which had itself been promoted from a
single request.

| Observed in v2 | Change in v3 |
|---|---|
| A new topic per signal, so supersession never fired | **Preference dimensions.** The layout supplies a seed list; each dimension holds one current value; a `<preference_dimensions>` view in every call shows each value and any open challenge; the contract says how new evidence confirms, refines (supersede) or challenges (hypothesis) the current value. The agent may add a dimension only with a `new_dimension:` reason |
| The same episode block appended up to four times in one event, because the agent couldn't see its earlier appends | **Writes reject repeated IDs,** and tool results name the entry IDs added or removed (with a warning on removal) |
| A preference promoted from one request decided the answer | The `unsupported_promotion` check; the evidence rule is restated around dimensions |
| 174 "violations" were really 12 problems re-reported every event | **Each violation is counted once,** at first appearance, with `resolved_at` when it disappears |

Run against the v2 run's final memory, the new checks flag all 11 invented topics,
four as near-duplicates of seed dimensions, and the preference that decided the
answer as an unsupported promotion.

**v3 on the same item** (gpt-5.4-mini, $1.33; see
[`examples/real_run_v3_openai/`](../examples/real_run_v3_openai)):

| | v2 | v3 |
|---|---|---|
| Preference dimensions used | 11 invented | 7, all seed |
| Preferences superseded | 0 of 11 | 4 of 11 |
| State entries added / changed | 43 / 17 | 35 / 40 |
| Accident hypothesis | "cautious about motorcycle risk", open, linked to nothing | "cautious about risky decisions", `challenges` the decision-style value, confirmed one event later |
| Decision-style value at the end | none | evidence-backed, threshold-based, cautious about risky options (supersedes the pre-accident value) |
| Distinct violations | 12 | 4 |
| Calls per event | 3.5 | 3.1 |
| Answer | B (wrong) | B (wrong) |

The memory now tracks the change, but the answer did not follow. At answer time the
model made one quick call (185 reasoning tokens, no reads). It went with the top
bullet of `context.md` (collaboration and step-by-step format), which omitted decision
style. The options differ only in their final sentence. B ("explore step by step,
start by investigating his methods") fits the stored memory about as well as the
correct C ("weigh it through a few ethical lenses"). The remaining gap is in how
memory is *used* and in how finely it is worded, not in whether it is kept.

**Answer-only replays** (`context-harness replay-answer`; five answers per strategy
on each saved memory, about $0.002 each; see
[`examples/answer_replays/`](../examples/answer_replays)):

| Memory | `default` | `dimension-first` |
|---|---|---|
| v3 | D B B B D (the old-preference trap twice) | B B B B B |
| v2 | B B B B B | B B B B B |

- **A single run hides variance.** On the v3 memory, the default answering step chose
  the trap 2 times in 5.
- **`dimension-first` makes the answer consistent** and avoids the trap, but neither
  memory leads this model to the correct C.
- **The limit is now memory content.** v3 recorded the person's *stated* preference
  for "comparing tradeoffs and pros and cons rather than a single right answer"
  (`O-13.1`) as evidence but never carried it into state. Under one value per
  dimension, it fell between decision style and explanation style. Across the two
  runs, 3 of 26 (v2) and 4 of 23 (v3) stated observations never reached any state
  entry or hypothesis.

**Baselines on the same item and model** ($0.44; see
[`examples/baselines_item0/`](../examples/baselines_item0)):

| Method | Answers |
|---|---|
| Full history, all 29 sessions in one call (81K tokens) | D B B |
| Recent history (the newest sessions that fit the answer budget) | B B B |
| Rolling summary | A |

**No method gets this item right.** Even with the whole history in front of it,
gpt-5.4-mini answers the way our memory does (mostly B, once the trap D). So the
layer is not losing something the raw history keeps. The item is hard for this model,
or its label hinges on a distinction the history barely supports. Twenty-nine
answers on one item can't separate those, which is why the next evaluation spans
several items and grades memory directly.

### v3 → v3.1: apply what the person says

| Problem in the v3 run | Change |
|---|---|
| 4 of 23 stated observations were recorded and never applied, including the one that decided the question | Contract: apply every `stated` observation in the same event (a `basis:`, a hypothesis's `for:`/`against:`, or a goal progress note). Check: `unused_stated_observation`, which flags exactly those four on the v3 memory and the three on v2's |
| A statement that fit two dimensions fit neither | Contract: one observation can bear on two dimensions; apply it to each |
| Wishes about one project had nowhere to go | Goal progress notes cite the observations they apply |
| The answering step leaned on the top of `context.md` | `dimension-first` is the default answer strategy (`--answer-strategy default` restores the old step) |

v3.1 has not met a real model yet.

## Known gaps

- **The archive itself has no rollover yet** (200,000 chars). That's enough for
  HorizonBench, but not for years of FoodBody data.
- **Semantic contradictions aren't detected yet.** Two current entries with
  different topics that conflict would need an LLM judge.
- **Real-model data is thin.** It covers 3½ events under v1 and one item each
  under v2 and v3, all the same item. v3.1 has not met a real model yet.
- **Open choices left for later:**
  - per-event review of open hypotheses (v2's accident hypothesis sat untouched for
    14 events);
  - review dates for goals (seven stayed active, some weeks stale);
  - ending an event automatically once its episode and `context.md` are written (v2
    averaged 3.4 calls per event).
