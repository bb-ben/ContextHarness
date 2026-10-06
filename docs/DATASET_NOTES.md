# HorizonBench: schema findings and design decisions

Inspected: GitHub `stellalisy/HorizonBench` @ `4b5076b147c952499b7921f20f811dd5aca7ef0b`
(2026-04-20): `README.md`, `evaluate.py`, `scripts/build_hf_dataset.py`,
`src/benchmark_prompts.py`, `src/timeline_generator.py`. Also inspected the HF
dataset `stellalisy/HorizonBench`, config `sample` (`sample/test-00000-of-00001.parquet`,
10 rows).

## Released schema (`sample` and `benchmark` configs)

| Field | Type | Who may see it in this harness |
|---|---|---|
| `id` | str, e.g. `Marcus Trevino_event31_20260327_turn3_facilitation_style_preferences_160_counterfactuals.json` | **Evaluator only.** It encodes the user's name, event index, probe turn index **and the tested preference domain**. The agent gets an opaque key. |
| `generator` | `sonnet-4.5` / `o3` / `gemini-3-flash` | Evaluator only (for aggregation) |
| `user_id` | `{generator}/user_{N}` | Evaluator only |
| `conversation` | str, 390K–730K chars in the sample (~100K–180K tokens) | Parsed into sessions; see below |
| `correct_letter` | `A`–`E` | Evaluator only |
| `options` | JSON str: list of `{letter, value, option}` | Agent sees `letter` + `option` text only. `value` (a snake_case attribute value) is evaluator-only |
| `has_evolved` | bool | Evaluator only |
| `preference_domain` | str | Evaluator only |
| `distractor_letter` | str, `""` if none | Evaluator only |
| `preference_evolution` | JSON str or `null` | Evaluator only |

`mental_state_graphs` (360 rows, ~560 MB) is not loaded by this milestone. If it is
used later, it must stay evaluator-side.

## How `conversation` is actually built

`src/benchmark_prompts.py::timeline_to_history` walks the user's
`conversation_record` **in stored order** (it never sorts by date). For each record
it writes `Date: <iso>` and an optional `Scenario: <text>` line, then `User: ` and
`Assistant: ` lines. When it reaches the target event, it stops at `turn_index`. The
whole string starts with `Conversation History:`. Multi-line turns carry on without a
prefix, which HB's `parse_conversations` handles by appending those lines to the
previous turn.

## Differences between the intended design and the real data

1. **"Original order" is not always chronological.** In 8/10 sample items, text
   order has non-decreasing dates. In **2/10 (`sonnet-4.5/user_112`, `sonnet-4.5/user_68`)**
   the first 22 and 29 sessions are short, scenario-less conversations with
   shuffled dates covering the user's whole 6-month timeline. **13 and 19 of them
   are dated after the probe session.** These are evolution-event conversations
   (`timeline_generator._generate_evolution_event_conversation`), the 2–4-turn
   chats in which the user mentions the life event that changes their preferences.
   For these two users they sit at the front of `conversation_record`. As a result,
   (a) the benchmark's own prompt contains **future events** for these items, and
   (b) life events appear *before* the preferences they later change.
2. **Timestamps are not unique.** Sessions on the same day share an identical
   synthetic timestamp (10–54 duplicates per item), so date alone can't order
   them. Ties keep text order.
3. **The last segment is not a session. It is the probe.** It is the target
   conversation cut off at `turn_index` (the `turnN` in the id), and in every
   sample item it ends with a user turn. It holds 1–21 turns in the sample. The
   five options are candidate *next assistant responses* to it.
4. **The question is not in `conversation`.** HB builds it from `options` with a
   fixed stem ("Which of the following responses would align with the user's
   preferences the most?").
5. **`Scenario:` is generator narration, not a user or assistant turn.** Example:
   *"The user is seeking help with X and values the AI assistant's expertise in Y."*
   Every normal session has one. Evolution-event sessions do not, so a *missing*
   scenario reveals that a session is a life-event session.
6. **Option `value`s don't always belong to one attribute.** For example, item 8 has
   values `External-accountability, False, Tough-love-critique, True,
   Objective-realism`. The correct value is the one that was expressed in the
   original turn.
7. **The distractor is defined only for some items.** `distractor_letter` is empty
   for static items. In the full benchmark it can also be empty for evolved items,
   because it is set only when a counterfactual is flagged
   `is_pre_evolution_distractor` and its value matches an option. All 6 evolved
   sample items have one. The distractor is the *original* value, even when there
   were intermediate evolutions (item 2: storytelling → analytical → personal
   anecdotes; the distractor is storytelling).
8. **`preference_evolution`** has `original_attributes`, `current_attributes`,
   `changed_attributes`, and `evolution_history[]` with
   `{date, event_name, event_category, trigger, attribute_changes}`. Item 7 has 4
   evolution steps. Each evolution date matches the date of a scenario-less
   session, which makes an evaluator-side "did the agent react to the life event?"
   diagnostic possible.
9. **Text artifacts**, kept as they are:
   - item 0 has the same user turn twice in a row;
   - item 8 has a user line labelled `Assistant:`.

   No content line was mis-split by the prefix parser (all `Date:` values parse as ISO).
10. **Scale.** History has 29–142 sessions per sample item (820–3,690 turns).
    Sessions range from 450 to 65K characters. A per-session agent loop
    therefore costs ~30–140 update loops per item.
11. **HB's own protocol** is one call with the full history, `temperature=0`,
    `max_tokens=64`, and regex letter extraction. Claude Opus 5.5 does not accept
    `temperature`, so runs here are not greedy. Answers go through a
    `submit_answer` tool, with HB's `extract_letter` as a text fallback.

12. **Option texts sometimes state the preference value outright.** Item 5's
    options include *"your ideal count of 8 sentences"* and *"honoring your
    1-sentence preference"*. This is a counterfactual-generation artifact, and the
    text is agent-visible because it is part of the question.
13. **Length bias in the sample: the correct option is the shortest of the five in
    8 of 10 items.** The correct option is the originally generated turn, while the
    four alternatives are rewrites that tend to run longer. A label-free "pick the
    shortest" rule scores 80% on the sample, against 52.8% for the best model in the
    HorizonBench paper. *Not yet checked on the full `benchmark` config.* Every
    summary therefore reports a shortest-option reference line.

## Decisions (ambiguities resolved)

| # | Ambiguity | Decision (default) | Alternative (flag) |
|---|---|---|---|
| A1 | Replay order | `chronological`: stable sort by timestamp; ties keep text order | `--order text` reproduces the benchmark's exact information set |
| A2 | Sessions dated after the probe | **Dropped** in `chronological`, because they are future information. The count is logged per item | Kept in `--order text` |
| A3 | Is the probe prefix replayed as a session? | No. It is shown only at question time, as the "current conversation" together with the options | — |
| A4 | Are `Scenario:` lines observations? | Yes. They are shown as session metadata, as HB shows them to every model | `--no-scenarios` hides them (and the "missing scenario" tell) |
| A5 | Tools at answer time | Agent may `list_files`/`read_file`, not write. It answers with `submit_answer` | — |
| A6 | "Immutable raw session log" | Written by the harness as an append-only, hash-chained `raw_sessions.jsonl` *outside* the workspace, chmod read-only at the end. Not reachable by agent tools | — |
| A7 | "Memory evaluation" in milestone 1 | Deterministic: per-file edit counts and sizes, no-op sessions, plus an evaluator-side check of whether the agent wrote to memory during the evolution-event session(s) | LLM-judge over final memory (next milestone) |
| A8 | Item identity shown to agent | Opaque `item_key` (hash). The raw `id` would leak the domain | — |
| A9 | Length artifact (finding 13) | Every summary reports the accuracy of a label-free shortest-option heuristic as a reference | — |
