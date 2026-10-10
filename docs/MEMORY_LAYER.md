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
| Are preferences disciplined? | `undeclared_dimension` (a topic outside the seed list with no `new_dimension:` reason), `near_duplicate_dimension`, `unsupported_promotion` (a preference resting on non-stated evidence from one event, with no confirmed hypothesis), `missing_change_note` (a preference or fact superseding another without `changed:`), `long_value` (a current preference or fact over 240 characters), `unused_stated_observation` (a `stated` observation cited by no state entry, hypothesis or goal note) |
| Does each entry have a source and date? | `future_id` (ID names an event that hasn't happened), `bad_date`, `date_mismatch` (date differs from its event's real date), `bad_source`, `parent_mismatch`, `missing_basis`, `dangling_ref`, `basis_not_observation`, `missing_topic`, `missing_end_date` (a transient entry with no `expires:`/`review:`, or an active goal with no `due:`/`review:`) |
| Do current-state files contradict each other? | `topic_conflict` (two `current` entries for one topic), `incomplete_supersession`, `supersession_mismatch`, `unresolved_confirmation`, `overdue_transient`, `overdue_goal` (an active goal past its review or due date), `archived_live`, `context_stale`, `context_uncited`, `context_dangling`, `context_inactive` |
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

### Memory judge

A multiple-choice answer is one noisy bit. HorizonBench also records the person's
preference as a few attributes, with values before and after each life event.
`context-harness judge-memory` shows an LLM judge the ground truth and a finished
run's beliefs (state, superseded entries, hypotheses, working context; not raw
episodes). For each attribute it grades whether memory holds the current value
(`matches` / `partial` / `absent` / `contradicts`), whether it still asserts the old
one, and whether it recorded the change. It runs on evaluator-side data only, after
the item is finished.

At first the judge was inconsistent: one memory scored 0.13, 0.63 and 0.38 on
three judgings, because it credited off-subject entries (e.g. "likes checklists" as
low tolerance for ambiguity). A stricter rubric (credit only entries on the
attribute's own subject), high reasoning effort and a majority vote over three
judgings brought agreement to 92–100%. On item 0 (gpt-5.4-mini judge, $0.20; see
[`examples/judge_item0/`](../examples/judge_item0)):

| Memory | Holds the person as they are now | On the 3 changed attributes | Change recorded |
|---|---|---|---|
| v2 | 0.0 | 0.0 | 0 of 3 |
| v3 | 0.38 | 0.5 (evidence-driven: yes; low tolerance for ambiguity: partly; deliberate: no) | 0 of 3 |
| Rolling summary | 0.75 | 1.0 | 0 of 3 |

**The summary's lead is right for the wrong reason.** All its evidence for
"data-driven" and "low tolerance for ambiguity" comes from crisis role-plays in
S-1 to S-4, before the accident, when the ground truth says the person was neither.
It recorded what happened inside the role-plays as preferences; the contract tells
the agent to skip task details. No memory recorded the change on any attribute.
So "holds the current value" rewards ending up right, however it got there, and
"change recorded" is the measure of tracking change. (Item vetting, below, confirms
that the change itself is real: the person's own requests about how to be helped
shift at the accident. The words the summary picked up were in the role-plays' content.)

### v3.1 on three items

Contract v3.1 on sample items 0, 5 and 6 (29, 78 and 75 events), against the
rolling summary and the full history in one call. Model gpt-5.4-mini, $14.79 in all;
see [`examples/multi_item_v31/`](../examples/multi_item_v31).

| Item (change) | Answer: agent / summary / full history ×3 | Holds the person now: agent / summary | Change recorded |
|---|---|---|---|
| 0 (accident → deliberate, data-driven) | B / A / D B B (correct C) | 0.38 / 0.75 | 0 everywhere |
| 5 (relocation → longer answers) | E / E / E E E (correct D) | 0.0 / 0.0 | 0 everywhere |
| 6 (advisor role → diplomatic, authoritative) | D / D / D D D (correct A; D is the old-preference trap) | 0.3 / 0.6 | 0 everywhere |

**No method answers any of the three correctly, and two of the items are poor tests:**
- **Item 0's options are ambiguous.** v3.1 now applies the pros-and-cons statement:
  its decision-style value reads "likes to decide deliberately… weighing pros and
  cons", and it appears in `context.md`. The answer is still B, in all 10 replays.
  B ("investigate his actual methods versus the rumors, then industry standards") is
  the most evidence-driven option, and the person became evidence-driven too.
- **Item 5's label contradicts its conversations.** The ground truth says the person
  came to want longer answers. In the 21 sessions after the change they ask for
  "shorter replies", "bullet beats", "can't do chunky paragraphs", and the final turn
  is "strip it to pure bullets." Every method picks "bullet points only"; the labeled
  answer is also the shortest option. A faithful memory must score 0 here.
- **Item 6 is a valid test, and the summary beats the agent.** After the change the
  person asks, task after task, for formal, measured, persuasive, diplomatic writing.
  The agent recorded this about 15 times, but v3.1 says a wish about one task belongs
  in that goal's notes, so each went under its own goal and nothing looked across
  them. Its tone value stayed "casual, humorous", from before the change, because
  the person's own chat register stayed casual. The summary generalized to "measured,
  persuasive, authoritative".

**No memory, ours or the summary's, recorded a change on any of the 10 changed
attributes.**

**Cost and time.** The agent costs $0.046–0.068 per event (rising as memory grows),
4–6 times the rolling summary, and took about 65 minutes for a 75-event item. Item 6
ended with dozens of `active` goals, many past their due date.

### Item vetting

Generated items can contradict themselves, and an item like that measures its label,
not the memory. `context-harness vet-items` shows a model the person's own turns
before and after each labeled change. For each changed attribute, it reports which
value those turns show in each period, and whether the question's final request
conflicts with the labeled answer. It also flags, without a model, whether the
correct option is the shortest. Evaluator side only.

| Verdict | Meaning | Sample items |
|---|---|---|
| `usable` | The change shows in the person's words, and the question's end fits the label | 0, 2, 7 |
| `memory_only` | The change shows, but the final request conflicts with the labeled answer: grade the memory, not the letter | 3, 6 |
| `suspect` | The change doesn't show | 5 (all four attributes still show the old value after the change) |

One judgment per item (gpt-5.4-mini, high effort, $0.34 for the six;
[`examples/vet_sample.json`](../examples/vet_sample.json)). In 5 of the 6 the
labeled answer is also the shortest option. Vetting checks the labels and the
question's end, not whether the options separate cleanly, so item 0's ambiguity
between B and C is not caught.

### v3.1 → v3.2: preferences across tasks, and goals that close

| Problem in the v3.1 runs | Change |
|---|---|
| On item 6 the same wish ("formal, measured, persuasive") came up in about 15 different tasks; each went under its own goal and never became a preference | A wish about one task that could be general also opens a hypothesis tagged with a dimension (`topic:` on `H`). `<preference_dimensions>` lists open hypotheses under their dimension, so the next task with the same wish finds and supports it |
| `tone` held "casual, humorous" because the person chats casually, while everything they asked to have written was formal | New seed dimension `output-style` (how the work they ask for should read); `tone` is now how to talk with them |
| 24 goals were still `active` at the end of item 6, 17 of them past their date and 7 with none | Every active goal needs `due:` or `review:`. Past-due goals appear under `<due>` to close (`achieved`, `abandoned`, or the new `lapsed`) or to push back. Checks: `missing_end_date` now covers goals; `overdue_goal` |

### v3.2 on item 6

$5.85 for the run and $0.20 for the judge; see
[`examples/v32_item6/`](../examples/v32_item6).

| | v3.1 | v3.2 |
|---|---|---|
| Holds the person now (judge) | 0.3 | 0.3 (tone moved from "contradicts" to "partial") |
| Change recorded | 0 of 3 | 0 of 3 |
| Goals at the end | 24 active (17 past their date, 7 with none) | 1 active, 15 lapsed, 1 achieved |
| Cost per event | $0.071 | $0.078 |
| Answer (unreliable on this item, per vetting) | D, the old-preference trap | D |

**The new mechanisms work.** A hypothesis tagged `output-style` ("may want drafted
scripts to sound professional and firm") collected six observations from different
tasks and was promoted; conversational `tone` stayed separate ("real, lightly
humorous"); goals close.

**What still blocks tracking change is the merge rule.** After the change the
`output-style` value was superseded five times, and each revision was a merge that
kept every earlier clause and added new ones: "dramatic yet respectful, high-passion"
from before the change is still in the final value, now joined by "opening with the
big vision… casual, leaderly, neighbor-friendly; formal reports professional and
firm". "Refines it: supersede with one merged statement" lets a value grow but never
lets the old part go, so a change can't be told apart from an addition.

Bookkeeping slips rose with the extra supersessions: 6 one-sided supersession links
and 7 entries whose date was edited to the latest update, all left open at the end.

### v3.2 → v3.3: revisions that replace, and checking that memory is used

| Problem | Change |
|---|---|
| Revisions merged new clauses into old ones (item 6's output-style value reached 569 characters and kept its pre-change wording) | Contract: a new value describes the person now; keep what still holds, drop what is replaced, about 200 characters. A worked example contrasts an addition with a change |
| Nothing recorded what changed | `changed:` on every preference or fact that supersedes another; the dimensions view shows each value's last change; check `missing_change_note` |
| Hand-made revisions left one-sided links and edited dates | A `supersede` tool writes the new entry and closes the old one in one step; it refuses a revision without `changed:` or over 240 characters |
| No way to tell whether a response used memory | `submit_answer` takes the entry IDs a choice rests on (`applied`); results show which were live, replaced or unknown |
| A multiple-choice letter is not the reply the person gets | `respond-eval`: the agent writes its reply from memory; the same reply is written from a summary, the full history or nothing; a judge grades each against the new and old values |

**FoodBody-style scenarios.** Two short hand-written histories in
[`scenarios/foodbody.py`](../scenarios/foodbody.py): meal planning after a type 2
diabetes diagnosis (loose ideas without numbers → structured plans with carb counts;
quick and playful stay), and coaching after a stress fracture (brutal and
performance-focused → gentle and recovery-focused; tables in km stay). About 8
events each, labels that match the conversations, and the correct option is never
the shortest. A full run costs cents.

**v3.3 results** (gpt-5.4-mini, $1.13 in all; see
[`examples/v33_scenarios/`](../examples/v33_scenarios) and
[`examples/respond_hb/`](../examples/respond_hb)):

| | Diagnosis | Injury |
|---|---|---|
| Answer: agent / summary | B / B (both correct) | B / B (both correct) |
| Memory on changed attributes: agent / summary | 1.0 / 1.0 | 0.5 / 1.0 |
| Changes recorded: agent / summary | **2 of 2** / 2 of 2 | 1 of 2 / 2 of 2 |
| Reply follows the new values (two replies): agent memory / none / full history / summary | 0.5, 1.0 / 0, 0 / 0.5, 0.5 / 1.0, 1.0 | 1.0 in every condition |
| Cost of the run: agent / summary | $0.25 / $0.017 | $0.19 / $0.014 |

- **The first recorded changes.** In the diagnosis scenario the agent replaced the
  format value ("structured meal plans for the week with carb counts per meal",
  `changed: now wants structured week plans, not loose ideas`) and the numbers value,
  instead of merging.
- **Memory is used.** Replies from the agent's memory rest on live entries only (3–5
  each, no replaced ones) and give carb counts; replies without memory never do.
- **Replacing can drop too much.** "Quick" (under 20 minutes) still held but fell out
  of the new value, and "playful" never left hypothesis status, so no reply was
  playful.
- **The injury change was read as temporary.** The person said "go easier on me
  right now", and the agent kept "blunt" as the standing value with "gentler while
  recovering" as transient state and an open challenge. Its answer and replies still
  followed the new state; the label counts it as half tracked.
- **The rolling summary does as well or better here, at a tenth of the cost.** On short
  histories with stated changes, a plain summary holds everything. Where the layer
  differs is that its changes, evidence and use are inspectable, and that is not yet
  an advantage in outcomes.
- **Some tests can't separate memory from none.** The injury scenario's last message
  and HorizonBench item 0's ethics question pull every reply toward the new values.
  On item 6, replies from v3.2's memory followed the old format, because that memory
  still held "numbered steps" as current.

## Known gaps

- **The archive itself has no rollover yet** (200,000 chars). That's enough for
  HorizonBench, but not for years of FoodBody data.
- **Semantic contradictions aren't detected yet.** Two current entries with
  different topics that conflict would need an LLM judge.
- **The judge is one model grading against a coarse taxonomy.** Its grades are
  consistent, but they haven't been checked against a person's reading, and it
  can't tell a value tracked through a change from one that was there all along.
- **Real-model data is thin, and the items are noisy.** Three items under v3.1, one
  item under v2 and v3. Vetting finds 3 of the 6 sample items with a labeled change
  usable as answer tests; it is one model's judgment per item and hasn't been checked
  against a person's reading.
- **v3.3 is untested on a long history.** The scenarios have about 8 events; whether
  replacement holds up over 75 (item 6) is open.
- **Replacement can drop what still holds** (the diagnosis scenario lost "quick").
- **Lasting or temporary is a judgment call** the agent can make differently from the
  label (the injury scenario).
- **Open choices left for later:**
  - per-event review of open hypotheses (v2's accident hypothesis sat untouched for
    14 events);
  - review dates for goals (seven stayed active, some weeks stale);
  - ending an event automatically once its episode and `context.md` are written (v2
    averaged 3.4 calls per event).
