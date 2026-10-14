"""Deterministic checks of what a scenario run's final memory holds, for comparing many runs.

For each scenario: the beliefs that should be current at the end, and the old ones that
should be gone. Matching is by keywords over live preferences, facts and transient
entries, so it is rough, but it is the same for every run, which is what a comparison
of conditions needs. Usage:

    python scenarios/check_memory.py runs/<run>/items/<item> [...]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

LIVE = re.compile(r"^- ([PFT]-\d+\.\d+) \| (current|active) \|(.*)$")

# scenario key -> (expected current beliefs, stale beliefs that should be gone), as regexes
EXPECT = {
    "dani": ({
        "brief answers": r"short|brief|concise|three lines|3 lines|quick answers",
        "one-pan / freezer": r"one[- ]pan|freezer|batch",
        "vegetarian": r"vegetarian",
        "spicy": r"spic|chili",
    }, {
        "detailed answers still standing": r"\b(lots of detail|in-depth|detailed|deep)\b(?!.*(short|brief|concise))",
    }),
    "lena": ({
        "no numbers": r"(no|without|avoid\w*|skip\w*)\b[^|]*\b(grams|calories|numbers|tracking|macros)",
        "gentle check-ins": r"check[- ]?ins?",
        "Mediterranean": r"mediterranean",
        "cooks for two": r"partner|for two|two people|self and",
    }, {
        "exact numbers still standing": r"exact (grams|macros|numbers)|grams per meal|weekly (numeric )?reports? (broken|by day)",
    }),
    "marisol": ({
        "professional for clients": r"client[^|]*(professional|formal|polished)|(professional|formal|polished)[^|]*client",
        "macros for clients": r"(macro|calorie|protein)",
        "family stays casual": r"(family|kids)[^|]*(fun|playful|silly|casual|emoji)|(fun|playful|silly|casual|emoji)[^|]*(family|kids)",
    }, {
        "no numbers still standing (unscoped)": r"no nutrition|without nutrition|no (numbers|calorie)",
    }),
}


def live_beliefs(snapshot: Path) -> list[tuple[str, str]]:
    """Current beliefs, one string each: state entries for the agent, the profile's lines
    (without its Changes section, which names old values on purpose), or the summary."""
    if (snapshot / "profile.md").exists():
        text = (snapshot / "profile.md").read_text().split("## Changes")[0]
        facts = (snapshot / "facts.md").read_text() if (snapshot / "facts.md").exists() else ""
        # a fact's current value only: "; earlier:" names old values on purpose, like ## Changes
        current = [line.split("; earlier:")[0] for line in facts.splitlines() if line.strip()]
        return [("facts", line) for line in current] + [("profile", line) for line in text.splitlines() if line.strip()]
    if (snapshot / "summary.md").exists():
        return [("summary", line) for line in (snapshot / "summary.md").read_text().splitlines() if line.strip()]
    out = []
    for name in ("state/preferences.md", "state/profile.md", "state/transient.md"):
        path = snapshot / name
        if path.exists():
            for line in path.read_text().splitlines():
                if m := LIVE.match(line):
                    out.append((m.group(1), m.group(3)))
    return out


def check(item_dir: Path) -> dict:
    key = next(k for k in EXPECT if k in item_dir.name)
    expected, stale = EXPECT[key]
    beliefs = live_beliefs(item_dir / "snapshots" / "final")
    text = [b for _, b in beliefs]
    found = {name: any(re.search(rx, t, re.I) for t in text) for name, rx in expected.items()}
    still = {name: any(re.search(rx, t, re.I) for t in text) for name, rx in stale.items()}
    result = json.loads((item_dir / "result.json").read_text()) if (item_dir / "result.json").exists() else {}
    return {
        "scenario": key,
        "expected_present": found,
        "stale_present": still,
        "scoped_values": sum("scope:" in t for t in text),
        "answer_correct": result.get("correct"),
        "cost_usd": round(result.get("estimated_cost_usd") or 0, 3),
        "finished": bool(result),
    }


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(json.dumps({"item": arg, **check(Path(arg))}))
