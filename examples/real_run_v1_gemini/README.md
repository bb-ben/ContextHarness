# Real run, contract v1: Gemini 3.7 Flash, HorizonBench sample item 0

These are the first real-model results: 3 complete events plus part of a 4th, before
Gemini's free tier returned repeated 503 "high demand" errors. They show how a model
actually uses the memory layer. They say nothing about preference change; item 0's
life event is at event 15.

| Event | Call 1 | Call 2 | Thinking tokens | Violations |
|---|---|---|---|---|
| S-0 | read 4 files | wrote 4 files + context.md, finish | 3,596 | 0 |
| S-1 | read all 6 state/episode files | wrote 5 files + context.md, finish | 3,842 | 0 |
| S-2 | read all 6 | wrote/edited 4 files + context.md, finish | 7,009 | 0 |
| S-3 | read 4 files | read 1 more, then the outage | — | — |

**What worked:**
- **The format:** zero violations, and every entry is dated and cites evidence that
  exists.
- **The evidence ladder:** a request repeated across two events was opened as a
  hypothesis (`H-1.1`), then confirmed and promoted to a preference (`P-2.1`).
- **`context.md`:** stayed a small index (742 chars).

**What motivated contract v2:**
1. **The agent read every state file in every event** and never used `lookup`. That
   costs one extra request per event, and the reads grow with memory. v2 pre-loads
   the state files.
2. **Evidence kinds were applied inconsistently.** A stated preference ("prefers a
   casual tone", `O-2.4`) became only a low-confidence hypothesis, while a one-off
   request was tagged `stated`. v2 adds worked examples, including lasting vs.
   situational.
3. **Episodes grew about 730 chars per event,** which would overflow the 60,000-char
   log at around event 80. v2 adds a "worth recording" rule, and the layer now rolls
   old episodes into the archive.

Files:
- `trace.jsonl`: every model call, tool call, file diff and memory check;
- `memory_report.json`: per-event access and checks;
- `workspace_after_event_3/`: the agent's memory as it stood.
