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
| Long-term episodic memory | `memory/episodes.md` (one `E` line per event, with `O` observations beneath), `memory/archive.md` (closed entries) | By `read_file` or `lookup` |
| Current user state | `state/profile.md`, `preferences.md`, `goals.md`, `transient.md`, `hypotheses.md` | By `read_file` or `lookup`; sizes always shown |

`context.md` is a working set and an index. Every bullet cites the ID of a live
state entry or hypothesis; it never copies one.

Evidence flows **event → observation → hypothesis → state**:
- A belief changes by adding a new entry that supersedes the old one. The old entry
  stays, marked `superseded`.
- Closed entries can move to the archive. They are never deleted.
- The layer itself does only two things on the agent's behalf:
  - it lists transient entries whose expiry or review date has arrived, under
    `<due>`;
  - it moves entries to the archive when asked.

## What one model call contains

```
system   agent.md
user     <event id="S-14" type="conversation" date="2025-10-15"> … </event>   (cached for the event's loop)
         <working_context> context.md now </working_context>
         <workspace> every file's size / limit </workspace>
         <due> transient entries now due, verbatim </due>
         <actions_so_far> this event's tool calls and results </actions_so_far>
         instruction + calls remaining
tools    list_files, read_file, lookup, append_file, edit_file, write_file, archive, finish
```

Each call is a fresh single-turn request. A `context.md` edit replaces the old
version in the next call, and nothing persists between events except the files.
`finish` lets the agent end an event in the same turn as its edits. With a
batched turn, an event costs one call.

## Instrumentation

After every event, `MemoryMonitor` parses the workspace and records which files were
read and written. It then runs the checks below. Results go to the item's trace (a
`memory_check` record per event) and to `memory_report.json`. To see them, run
`context-harness inspect-memory <item_dir>`.

| Question | Check codes |
|---|---|
| Does each entry follow the format? | `unparsed_line`, `unparsed_entry`, `wrong_file`, `orphan_observation`, `duplicate_id`, `bad_status`, `bad_kind` |
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

## Known gaps

- **Episodes are append-only and capped at 60,000 characters.** That's enough for
  HorizonBench (at most 142 events), but months of FoodBody data will need
  rollover.
- **Semantic contradictions aren't detected yet.** Two current entries with
  different topics that conflict would need an LLM judge.
- **There is no real-model data yet.** Behavior under a real model, such as whether it
  reads before editing or keeps `context.md` current, is unmeasured.
