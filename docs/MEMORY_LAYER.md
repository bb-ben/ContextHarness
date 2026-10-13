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
| Are preferences disciplined? | `undeclared_dimension` (a topic outside the seed list with no `new_dimension:` reason), `near_duplicate_dimension`, `unsupported_promotion` (a preference resting on non-stated evidence from one event, with no confirmed hypothesis), `missing_change_note` (a preference or fact superseding another without `changed:`), `long_value` (a current preference or fact over 240 characters), `undecided_hypothesis` (supported by two events and still open), `overdue_preference_review`, `preference_into_transient` (a hypothesis about how they want help resolved into a transient entry), `uncollected_signals` (tagged signals on a dimension from two events that no belief cites), `unused_stated_observation` (a `stated` observation cited by no state entry, hypothesis or goal note) |
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

### v3.4 → v3.6: one home for how they want help

Two more scenarios end on a neutral message ("Any ideas for dinner this week?",
"Planning groceries for the week, can you help?"), so a reply without memory has
nothing to go on. They are the ones that separate memory from none. Results are in
[`examples/scenarios_v34_v36/`](../examples/scenarios_v34_v36); $2.25 for all three
rounds.

| Version | Change | What happened on the neutral-ending scenarios |
|---|---|---|
| v3.4 | Carry over what still holds when replacing; decide a hypothesis once two events support it | Both answered correctly, but in 3 of 4 scenarios the change went into a transient entry ("needs very short help in the newborn phase") with the old preference left standing; "cooks for two" lived only in a goal that closed; the burnout reply asked for cuisine and household |
| v3.5 | Decisions change preferences; temporary overrides name the preference they override | Worse: both changes went into transient entries that expired a week later ("three-line answers tonight"), the old preference returned, and the burnout answer chose the old-preference trap. Given the same event twice (a resume replayed it), the model once superseded and once wrote a transient override |
| v3.6 | How they want help lives only in preferences; a change that may not last still supersedes, with `review:`; transient entries hold facts and circumstances | Both correct. "Concise answers for now" (review in four weeks) and "no grams, calories or tracking" superseded the old values with `changed:` notes; "vegetarian" and "cooks for self and partner" are profile facts; the transient entries are circumstances ("signed off work with burnout") |

**Replies** (share of changed attributes the reply follows now):

| | New parent | Burnout |
|---|---|---|
| No memory | 0, 0 | 0, 0 |
| v3.4 memory | 0.5, 0.5 | 0.5, 1.0 |
| v3.6 memory | 0.5, 0.5 | 0.5, 0.5 |
| Full history | 0.5 | 1.0 |
| Rolling summary | 1.0, 1.0 | 1.0, 1.0 |

The v3.6 replies use memory directly: "for you and your partner,
Mediterranean-leaning, skipping grams, calories and tracking" (v3.4's asked for the
household and cuisine). What they miss: "one-pan, freezer-friendly", asked for in
three different tasks, never became a preference; and the burnout reply holds "gentle
check-ins instead of reports" in memory but doesn't check in. The summary still writes
replies that fit better.

**Two places for one kind of belief is a design flaw, not a wording problem.** With a
transient path for how someone wants help, the model's choice between "change" and
"for now" was close to a coin flip, and the "for now" branch expired quietly. Removing
the path fixed both scenarios where rewording v3.4 → v3.5 had made them worse.

### v3.6 → v3.8: collecting repeated wishes, and replies that account for all of memory

| Gap after v3.6 | Change |
|---|---|
| "One-handed", "freezer meals", "one-pan" were observed in three tasks and never linked; the agent doesn't see past observations at update time, and no seed dimension fitted | v3.7: preference signals are tagged with a dimension (`topic:` on `O` lines); the dimensions view lists each dimension's signals that no belief cites yet; check `uncollected_signals`; new seed dimension `constraints` |
| A reply used some of what memory held and silently skipped the rest (a wanted check-in) | v3.7: `respond` needs every current preference, fact and active circumstance listed as applied or skipped, and returns the reply if any is missing |
| v3.7 lost "still vegetarian, obviously", said in passing, and both replies suggested chicken | v3.8: a layout can carry profile topics (the `foodbody` layout: diet, allergies, health conditions, household, kitchen), shown in every call with their current fact or "(none yet)" |

**v3.7 on all four scenarios** ($1.32 with replies; see
[`examples/scenarios_v37_v38/`](../examples/scenarios_v37_v38)): all four answered
correctly, and replies from memory fit the changed attributes in 7 of 8 cases (the
eighth half), never the old values; every reply accounted for all of memory.

| Reply follows the new values | Diagnosis | Injury | New parent | Burnout |
|---|---|---|---|---|
| No memory (earlier runs) | 0, 0 | 1, 1 | 0, 0 | 0, 0 |
| v3.6 memory | (not rerun) | (not rerun) | 0.5, 0.5 | 0.5, 0.5 |
| **v3.7 memory** | **1, 1** | **1, 1** | **1, 1** | **1, 0.5** |
| Rolling summary | 1, 1 | 1, 1 | 1, 1 | 1, 1 |

What v3.7's memories held: diagnosis kept "under 45 g carbs **and under 20 minutes**"
and finally promoted "playful"; injury made "gentler, healing-focused coaching for now"
the current preference with a review date; new parent collected "limited time,
one-handed, one-pan" under `constraints`; burnout held no numbers, gentle check-ins,
Mediterranean and cooking for two. The cost per scenario rose to $0.25–0.38 (v3.6:
about $0.18).

**v3.8 on the new-parent scenario** ($0.46 with replies) recorded vegetarian at its
first mention and both replies kept to it, but this time "one-pan" was not collected
and the replies missed it. **Run-to-run variance is now the main limit:** v3.7 caught
one-pan and lost vegetarian, v3.8 the reverse. The structure makes the right outcome
reachable; gpt-5.4-mini reaches it inconsistently, so a version has to be judged on
repeated runs.

### v3.8 on item 6: the long-history test

$7.11 for the run and $0.33 for its evaluations; see
[`examples/v38_item6/`](../examples/v38_item6).

| | v3.1 | v3.2 | v3.8 |
|---|---|---|---|
| Memory holds the person now (judge) | 0.3 | 0.3 | 0.3 |
| Changes recorded | 0 of 3 | 0 of 3 | 1 of 3 (tone, partly) |
| Replies follow the new / the old values | not measured | 0.33 / 0.67 | 0.17 / 0.67 |

**The replies used memory faithfully, and memory was stale.** Both replies accounted
for every entry and rested on 14–19 live ones, none replaced, but the current values
still said "step-by-step checklists" and "dramatic, punchy", so the replies came out
as blunt numbered plans: the person before the change.

**The cause is accretion that passes the rules.** The `output-style` value was
superseded seven times, each with a `changed:` note reading "now also wants…"; the
pre-change "dramatic, punchy" survives in the final value beside "measured, formal
enough for the UN". On the 8-event scenarios a change arrives in one or two clear
messages and gets replaced. Over 75 events it arrives as dozens of task requests,
each of which looks like an addition. Requiring a revision to say what changed does
not stop "what changed" from being "one more thing".

The next design question is re-deriving a value from recent evidence rather than
patching the previous value, which is what a rolling summary does each event.

**How much of the 0.3 is the item?** Counting the person's own messages after the
change, the format evidence is even: step-by-step words in 24 of 57 sessions, summary
words in 23, both in 11. A one-sided vetting pass had called every change on this item
clear. Vetting now counts both sides ([`examples/vet_sample_two_sided.json`](../examples/vet_sample_two_sided.json),
$0.46 plus $0.15 for a pass that ran out of output tokens): item 5's four changes are
contradicted (after the change, 13 sessions still point to short answers against 7);
items 0 and 6 are supported, though item 6's format only narrowly. `judge-memory
--vet` scores a memory only on unchanged attributes and supported changes. Single-pass
LLM counts over 50+ sessions are rough; the keyword count and the model's count
disagreed in direction on item 6.

### v3.9 and the design-or-agent question: 18 runs

v3.9 revises a value from its dimension's recent evidence (shown in the view), refuses
"now also…" revisions, and allows scoped values. To separate the design from the
agent, it ran three times at medium and three times at high reasoning effort on the
three neutral-ending scenarios: new parent, burnout, and a new **meal-prep** scenario
in which client work turns professional and macro-labeled over several small requests
while family cooking stays casual. $7.96 for the round, including two-sided vetting;
see [`examples/v39_effort_runs/`](../examples/v39_effort_runs) and
[`scenarios/check_memory.py`](../scenarios/check_memory.py).

| | Medium (9 runs) | High (9 runs) |
|---|---|---|
| Multiple-choice answers correct | 9 / 9 | 9 / 9 |
| Replies follow the new values (mean) | 0.72 | 0.78 |
| Cost | $2.56 | $4.42 |

**Reasoning effort doesn't matter:** the same results at 1.7 times the cost. What
varies is the run: the same scenario under the same settings produced replies scoring
0.0, 0.5 and 1.0.

**Where the design failed:**
- **Overlapping dimensions.** `help-format` and `detail-level` both claimed answer
  length. In all six new-parent runs the agent recorded "very short, about 3 lines" on
  one while "detailed, thorough explanations" stayed current on the other; nothing
  compares values across dimensions, and 2 of 6 replies followed the stale one.
- **Scopes can dodge a conflict.** One run scoped "short answers" to "cooking help",
  which is every request this person makes, leaving "detailed" as the default.

**Where the design worked:** scoped values. In the meal-prep scenario the agent kept
client work (professional, macros per serving) apart from family cooking in most runs,
and all six replies to the client request followed the new values.

**Where the agent's judgment differs from the label:** burnout replies skipped the
"gentle check-ins" preference as irrelevant to a grocery request in 5 of 6; and
recording varies run to run ("one-pan" collected in 4 of 6).

### v3.10: non-overlapping dimensions, check feedback, and a review step

v3.10 moves answer length out of `help-format` (only `detail-level` has it), shows
the agent the memory checks' open problems in its next update, and can add a short
review call after each update ("re-read memory as your edits left it; fix
contradictions, scopes that cover every request, unrecorded statements or facts").
Same three scenarios, three runs per condition, medium effort; $7.28 for the round;
see [`examples/v310_review_runs/`](../examples/v310_review_runs).

| | v3.9 | v3.10 | v3.10 + review |
|---|---|---|---|
| Replies follow the new values (mean of 9) | 0.72 | 0.67 | **0.83** |
| New parent: "detailed" still current | 3 of 3 | 3 of 3 | **1 of 3** |
| Burnout: "exact numbers" still current | 1 of 3 | 2 of 3 | **0 of 3** |
| Answers correct | 9 of 9 | 9 of 9 | 9 of 9 |
| Cost for 9 runs | $2.56 | $2.63 | $4.30 |

- **Fixing the dimension overlap alone didn't help.** Both values now sat on
  `detail-level`, and the agent dodged the conflict with a scope instead: "shorter
  answers in late-night, one-handed situations", leaving "detailed, thorough" as the
  default.
- **The review step did.** It replaced the stale value ("very short answers, about 3
  lines, for now") and removed the stale "exact numbers", and replies rose from 0.67 to
  0.83, at 63% more cost. A second look at its own result catches what the agent
  misses in the first pass.
- **Still missed in every condition:** burnout replies treat "gentle check-ins" as
  irrelevant to a grocery request, and the meal-prep memory mostly loses "family
  cooking stays casual" (1 of 9 runs keeps it). Nine runs per condition is still a
  small sample.

### The profile method: simpler, and better so far

**Diagnosis.** Every structured-agent failure from v3.2 to v3.10 had the same shape:
an old belief survived because one of many per-entry decisions went the wrong way
(merge instead of replace, temporary instead of lasting, the wrong dimension, a scope
that dodged the conflict, "now also…"). Each fix closed one route and the failure
moved to the next, while the contract grew from 14,300 to 25,200 characters. The
rolling summary, which makes one decision per event (rewrite who this person is now),
kept scoring at or above the agent.

**The profile method** (`--method profile`, [`methods/profile.py`](../src/context_harness/methods/profile.py))
keeps what worked and drops the bookkeeping. One call per event, under a 1,781-character
prompt: append the episode and up to three observations (numbered by the harness), and
rewrite one profile (facts, how they want help, goals, right now, dated changes) that
cites observation IDs. Every version of the profile is kept as the change history.

**Results** (gpt-5.4-mini, medium effort; $2.97 for the round; see
[`examples/profile_method/`](../examples/profile_method)):

| | v3.10 + review | Rolling summary | **Profile** |
|---|---|---|---|
| Replies follow the new values, 3 neutral-ending scenarios (9 each) | 0.83 | 0.83 | **0.94** |
| Same, all 5 scenarios (15 each) | not run | 0.90 | **0.97** |
| Expected beliefs present (memory checks) | most | all | all |
| Cost per scenario run | about $0.48 | about $0.02 | about $0.05 |
| 75-event item: memory against ground truth | 0.3 (v3.8) | 0.6 (v3.1 round) | 0.6 |
| 75-event item: replies follow the new values | 0.17 (v3.8) | not run | 0.0 |
| 75-event item: cost | $7.11 (v3.8) | about $0.93 | $1.43 |

- **On the scenarios the profile is the best of the three**, at about a tenth of the
  structured agent's cost, and it keeps what the summary lacks: evidence for each
  claim and a dated record of what changed.
- **It keeps context-dependent preferences the way a person would write them**: "In
  the newborn period … very short; otherwise detailed, science-forward", and "client
  documents: professional, with macros; family meals: fun".
- **The 75-event item is still not solved.** The profile doubles the structured
  agent's memory score and gets the tone change right, but keeps "step-by-step" as
  the format, and the replies follow it. That format change is evenly split in the
  person's own messages (24 sessions against 23), so it is a weak test either way.

### Two outside benchmarks: MemConflict and DynamicMem

Both benchmarks replay a long history through the profile or the summary and ask
questions answered from that memory alone. MemConflict has about 52 chat sessions per
person and asks whether the right memory wins when memories disagree. DynamicMem has
about 15 months of app logs per user and asks for current habits and for service that
depends on them. See the README for setup, and
[`examples/benchmarks_pilot/RESULTS.txt`](../examples/benchmarks_pilot/RESULTS.txt).

**Pilot** (gpt-5.4-mini; MemConflict: 2 people, 30 questions each; DynamicMem: 1 user,
5 checkpoints):

| | Profile | Summary |
|---|---|---|
| MemConflict: dynamic (a real update) | 0.60 | 0.50 |
| MemConflict: static (a false later claim) | 0.05 | 0.05 |
| MemConflict: conditional (a preference under a condition) | 0.21 | 0.05 |
| DynamicMem: state completion | 0.10 | 0.24 |
| DynamicMem: personalized service | 0.34 | 0.29 |

- **Most misses are retention, not reasoning.** The evidence for the answer was in
  memory for 3 of 20 static questions and 7 of 19 conditional ones for the profile (2
  and 1 for the summary). When it was there, dynamic answers were right about 70% of
  the time. Background facts (name, birthplace, degree, siblings) and "when they like
  X" don't survive in a document organized around how to help.
- **Newest wins, even when the newest claim is false.** Static conflicts plant a later
  wrong statement; both designs adopt it. Neither keeps "these two sources disagree".
- **The profile has no place for routines.** DynamicMem's state questions are
  schedules (weekly budget review, Sundays 09:30, home office). The profile's
  sections don't hold them, so it scores below the summary on state completion even
  though it serves the user slightly better.
- **Both memories overflowed their 8,000-character limit**, and the hard cut dropped
  the end. For the profile, that means Right now and Changes. The profile was cut on
  26 of 53 MemConflict events and 44 of 68 DynamicMem events; the summary on 51 of 53
  and 56 of 67. On the earlier FoodBody runs this was 11% of profile updates.
  - Asking the model afterwards to rewrite under the limit trims only 5–10%, even
    with a concrete target.
  - Showing the current length and limit in every update prompt, plus the shorten
    step, halved the profile's cuts (13 of 53) but left the summary's high (38 of
    52).
  - On MemConflict person 0 the fix did not change the profile's accuracy (0.25
    before, 0.22 after, 30 questions), which fits the retention diagnosis above.

## Known gaps

- **Long histories outgrow an 8,000-character memory.** On MemConflict and DynamicMem
  both designs keep too little: background facts, conditional preferences, routines.
  A cut-off tail is the worst way to lose it. Candidates: a Routines section, a larger
  limit for long histories, and keeping "sources disagree" instead of newest-wins.

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
- **Changes don't hold up over long histories.** On the 75-event item, v3.8's values
  grew by addition ("now also…") and kept the pre-change wording.
- **Run-to-run variance, not reasoning effort.** High effort gave the same results as
  medium at 1.7 times the cost; the same settings still produce different memories on
  different runs.
- **Scopes can dodge a conflict**: a stale value can survive as the "default" while
  the change is scoped to a narrow situation. The review step catches some of these.
- **The review step costs 63% more**; whether its gain holds on long histories is
  untested.
- **A rolling summary still matches or beats the layer on these short scenarios**, at
  a fraction of the cost. The layer's edge so far is that what it believes, why and
  what it used are all inspectable.
- **Replacement can drop what still holds** (the diagnosis scenario lost "quick" in
  v3.3 and v3.4).
- **Open choices left for later:**
  - per-event review of open hypotheses (v2's accident hypothesis sat untouched for
    14 events);
  - review dates for goals (seven stayed active, some weeks stale);
  - ending an event automatically once its episode and `context.md` are written (v2
    averaged 3.4 calls per event).
