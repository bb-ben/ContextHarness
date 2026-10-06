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

   **Full benchmark (audit, see below):** this happens only with the `sonnet-4.5`
   generator, in **347 of its 1,052 items (33%), from 31 of its 106 users**. That
   is 8.2% of all items and 3,294 post-question sessions in total. No `o3` or
   `gemini-3-flash` item is affected. No item's *tested* evolution event is dated
   after its question, so the label itself is never in the future.
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

   In the sample, no turn text was mis-split by the prefix parser. In the full
   benchmark, some was: see finding 14.
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
13. **Length bias: the correct option is usually the shortest.** In the sample, the
    correct option is the shortest in 8 of 10 items. **On the full benchmark, a label-free "pick the shortest" rule scores
    78.2% (95% CI 76.9–79.5%)**, against 52.8% for the best model in the
    HorizonBench paper and 20% chance:

    | Group | Items | Shortest-option rule | Correct option's length rank 1 / 2 / 3 / 4 / 5 |
    |---|---|---|---|
    | All | 4,245 | 78.2% | 78.7 / 11.2 / 4.5 / 3.1 / 2.4 % |
    | gemini-3-flash | 2,212 | 83.4% | 83.6 / 8.9 / 3.0 / 2.0 / 2.5 % |
    | o3 | 981 | 79.8% | 79.8 / 12.4 / 3.5 / 2.3 / 1.9 % |
    | sonnet-4.5 | 1,052 | 65.8% | 67.2 / 15.0 / 8.8 / 6.2 / 2.8 % |
    | evolved | 2,484 | 79.9% | 80.4 / 10.2 / 3.9 / 3.1 / 2.3 % |
    | static | 1,761 | 75.8% | 76.3 / 12.7 / 5.4 / 3.1 / 2.6 % |

    Rank 1 means shortest; tied lengths share the better rank. The median correct
    option is 0.70× the median length of the other four. Counting words instead of
    characters gives 76.3%, and "pick the longest" scores 2.5%. The old-preference
    distractor is shorter than the correct option in only 8.4% of the 1,492 evolved
    items that have one, so length does not explain models choosing it. Every
    summary reports the shortest-option rule as a reference line.

    The measurement was reproduced from the raw parquet files by a separate script
    that uses no harness code. The paper checks position bias (§6.1) but, as far as
    we can tell, not option length. **The cause is not established.** Two candidates:
    - *Generation:* the correct option is the turn originally generated in the
      conversation, and the four alternatives are rewrites that may run longer.
    - *Filtering:* items are kept only if 5 LLMs all answer wrong *without* the
      history. If those LLMs favor fuller, more elaborate answers, the filter would
      discard items whose correct option is long. "Pick the longest" scoring 2.5%,
      far below chance, fits this explanation.

    Telling them apart needs the unfiltered items, which are not released.
14. **HB's parser mis-splits 147 items (3.5%).** `parse_conversations` treats any
    line beginning `Date: ` as a new session, and any line beginning `Scenario: `
    as the session's scenario, even inside turn text:

    | What the turn text contains | Upstream result | Items |
    |---|---|---|
    | `Date: YYYY-MM-DD`, `Date: [counting backward]` | invents a session with an unparseable date | 44 |
    | `Scenario: …` (e.g. a fantasy-football reply) | overwrites the session's real scenario; the line is lost | 98 |
    | `Date: 1943-04-15` (a quoted archive record) | invents a session dated 1943, which date-sorting would move to the front | 5 |

    The harness uses `hb_compat.parse_sessions`, which starts a session only on a
    full `YYYY-MM-DDTHH:MM:SS` timestamp and reads `Scenario:` only before a
    session's first turn. Its output is identical to upstream's on the other 4,098
    items. HB's headline evaluation (`build_prompt`) passes the raw text to the
    model and never calls this parser, so **the paper's reported scores are not
    affected**; the bug matters only to methods that split conversations with
    `parse_conversations`, as HB suggests for retrieval. A turn line that begins `User: ` or `Assistant: ` still can't be told
    apart from a real turn; item 8's mislabelled line may be one.

### Reproducing the audit

```bash
uv run context-harness audit-data --config benchmark --out data/audit_benchmark
```

It takes about 20 seconds after the ~1.6 GB download. It writes one record per
item to `audit_items.jsonl` and the aggregates to `audit_summary.json`; a copy of
the summary is in `docs/benchmark_audit_summary.json`. The audit reads labels, so
it is evaluator-side only.

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
| A10 | Parser mis-splits (finding 14) | `parse_sessions`: sessions start only on full timestamps; `Scenario:` only before the first turn | `hb_compat.parse_conversations` kept verbatim for comparison |
