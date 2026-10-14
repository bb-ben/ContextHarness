"""Pick the episode-log lines worth showing when answering.

The profile and the facts ledger hold what is true now; details mentioned once
(a preference under a condition, a routine's exact time, a named possession) live only
in the episode log. When the whole log fits the answer budget it is shown in full;
otherwise BM25 against the question picks the lines, newest first among equals, and
they are shown oldest first with their dates.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_EPISODE = re.compile(r"^- (E-\d+) \| (\S+) \|(?: source: \S+ \|)? ?(.*)$")
_OBSERVATION = re.compile(r"^- (O-\d+\.\d+) \| (.*)$")
_STOP = set("""a an and are as at be been but by can could did do does for from had has have he her his how
i if in into is it its me my of on or our she so than that the their them they this to up was we were what
when where which while who why will with would you your user users""".split())


def _terms(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in _STOP]


def dated_entries(log_lines: list[str]) -> list[str]:
    """Episode-log lines as self-contained dated lines: '- 2022-03-09 | O-6.2 | stated | ...'."""
    out, date = [], ""
    for raw in log_lines:
        line = raw.strip()
        if m := _EPISODE.match(line):
            date = m.group(2)
            out.append(f"- {date} | {m.group(1)} | {m.group(3)}")
        elif m := _OBSERVATION.match(line):
            out.append(f"- {date} | {m.group(1)} | {m.group(2)}")
    return out


def relevant_log(log_lines: list[str], query: str, budget_chars: int) -> str:
    entries = dated_entries(log_lines)
    if sum(len(e) + 1 for e in entries) <= budget_chars:
        return "\n".join(entries)
    docs = [_terms(e) for e in entries]
    n, avg = len(docs), max(1.0, sum(map(len, docs)) / max(1, len(docs)))
    df = Counter(w for d in docs for w in set(d))
    query_terms = set(_terms(query))

    def bm25(d: list[str], k1: float = 1.2, b: float = 0.75) -> float:
        tf, s = Counter(d), 0.0
        for w in query_terms & tf.keys():
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg))
        return s

    scores = [bm25(d) for d in docs]
    chosen, used = [], 0
    for i in sorted(range(n), key=lambda i: (-scores[i], -i)):  # relevance, then newest
        if used + len(entries[i]) + 1 <= budget_chars:
            chosen.append(i)
            used += len(entries[i]) + 1
    return "\n".join(entries[i] for i in sorted(chosen))
