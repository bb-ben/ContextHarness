"""Synthetic HorizonBench-format rows with canary strings for leakage tests.

`conversation` is assembled exactly like HorizonBench's
src/benchmark_prompts.py::timeline_to_history: records in *stored* order, each
`Date:` / optional `Scenario:` / `User:` / `Assistant:` lines, the target
conversation cut at turn_index, and a trailing blank line.
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tiny_benchmark.jsonl"

# Strings that must never reach the agent (anywhere) ...
FORBIDDEN_ALWAYS = [
    "CANARY_ITEM_ID",
    "canary_domain",
    "CANARY_EVOLUTION_EVENT",
    "canary_value_",
    "CANARY_FUTURE_SESSION",  # dated after the probe; dropped by chronological policy
]
# ... and strings that may appear only once the question is asked.
FORBIDDEN_DURING_REPLAY = ["CANARY_PROBE_TURN", "CANARY_OPTION_TEXT", "Which of the following responses"]


def build_conversation(records: list[dict]) -> str:
    out = "Conversation History:\n"
    for rec in records:
        out += f"Date: {rec['date']}\n"
        if rec.get("scenario"):
            out += f"Scenario: {rec['scenario']}\n"
        for role, text in rec["turns"]:
            out += f"{'User' if role == 'user' else 'Assistant'}: {text}\n"
    return out + "\n"


def make_row(
    item_id: str,
    records: list[dict],
    options: list[tuple[str, str, str]],
    correct: str,
    distractor: str = "",
    evolution: dict | None = None,
    generator: str = "fixture-gen",
    user_id: str = "fixture-gen/user_0",
    domain: str = "canary_domain",
) -> dict:
    return {
        "id": item_id,
        "generator": generator,
        "user_id": user_id,
        "conversation": build_conversation(records),
        "correct_letter": correct,
        "options": json.dumps([{"letter": l, "value": v, "option": t} for l, v, t in options]),
        "has_evolved": evolution is not None,
        "preference_domain": domain,
        "distractor_letter": distractor,
        "preference_evolution": json.dumps(evolution),
    }


def _day(d: int) -> str:
    return f"2025-10-{d:02d}T09:00:00.000001"


def evolved_row() -> dict:
    """Evolved item whose stored order mimics the sonnet-4.5 artifact: an
    evolution-event chat placed first, plus a chat dated after the probe."""
    records = [
        {  # stored first but dated 10-12: evolution-event conversation (no scenario)
            "date": _day(12),
            "turns": [
                ("user", "SESSION_MARKER_2 Hey, so I just moved to Tokyo for a demanding new job. Can you help me find a gym?"),
                ("assistant", "Congrats! Look for gyms near your office with early hours."),
            ],
        },
        {  # stored second but dated after the probe: future information
            "date": _day(28),
            "turns": [
                ("user", "CANARY_FUTURE_SESSION I just switched careers again."),
                ("assistant", "Exciting news."),
            ],
        },
        {
            "date": _day(2),
            "scenario": "The user wants help planning a weekend and likes detailed explanations.",
            "turns": [
                ("user", "SESSION_MARKER_0 Please give me long, detailed explanations with lots of background."),
                ("assistant", "Of course. Here is a thorough plan:\n1. Morning hike\n2. Long lunch"),
                ("user", "Perfect, I love detailed answers with background."),
                ("assistant", "Glad it helps."),
            ],
        },
        {  # same timestamp as the previous record: tie keeps text order
            "date": _day(2),
            "scenario": "The user asks for a recipe.",
            "turns": [
                ("user", "SESSION_MARKER_1 Recipe for a quick pasta please, detailed steps."),
                ("assistant", "Boil water, salt it, cook pasta 9 minutes, toss with sauce."),
            ],
        },
        {
            "date": _day(15),
            "scenario": "The user needs a quick reply between meetings.",
            "turns": [
                ("user", "SESSION_MARKER_3 Busy week. Keep it short: brief bullet points only, no background."),
                ("assistant", "- Noted\n- Short bullets from now on"),
            ],
        },
        {  # probe: target conversation cut at turn_index=1 (ends with a user turn)
            "date": _day(20),
            "scenario": "The user asks for advice before a presentation.",
            "turns": [("user", "CANARY_PROBE_TURN I present in ten minutes, any advice?")],
        },
    ]
    options = [
        ("A", "canary_value_long", "CANARY_OPTION_TEXT_A Here is a long, detailed explanation with background on presentation theory, history, and many examples."),
        ("B", "canary_value_short", "CANARY_OPTION_TEXT_B - Breathe\n- Brief bullet points\n- Short and focused"),
        ("C", "canary_value_story", "CANARY_OPTION_TEXT_C Let me tell you a story about my grandmother."),
        ("D", "canary_value_question", "CANARY_OPTION_TEXT_D What do you think the audience wants?"),
        ("E", "canary_value_humor", "CANARY_OPTION_TEXT_E Tell a joke first, everyone loves a joke."),
    ]
    evolution = {
        "original_attributes": {"length": "canary_value_long"},
        "current_attributes": {"length": "canary_value_short"},
        "changed_attributes": {"length": {"original": "canary_value_long", "current": "canary_value_short"}},
        "evolution_history": [
            {
                "date": _day(12),
                "event_name": "CANARY_EVOLUTION_EVENT relocation",
                "event_category": "career",
                "trigger": "evolution_event",
                "attribute_changes": {"length": {"from": "canary_value_long", "to": "canary_value_short"}},
            }
        ],
    }
    return make_row(
        "Fixture User_event9_20260101_turn1_canary_domain_1_counterfactuals.json CANARY_ITEM_ID",
        records,
        options,
        correct="B",
        distractor="A",
        evolution=evolution,
    )


def static_row() -> dict:
    records = [
        {
            "date": _day(3),
            "scenario": "The user asks about houseplants.",
            "turns": [
                ("user", "My ferns keep dying. I prefer practical numbered steps."),
                ("assistant", "1. Water weekly\n2. Indirect light\n3. Humidity tray"),
            ],
        },
        {
            "date": _day(9),
            "scenario": "The user follows up on plants.",
            "turns": [
                ("user", "The numbered steps worked. Numbered steps again please for my cactus."),
                ("assistant", "1. Water monthly\n2. Bright light"),
            ],
        },
        {
            "date": _day(11),
            "scenario": "The user asks about orchids.",
            "turns": [("user", "Orchid tips?")],
        },
    ]
    options = [
        ("A", "canary_value_essay", "An orchid is a fascinating plant with a long history in Victorian collecting."),
        ("B", "canary_value_steps", "1. Water weekly with ice cubes\n2. Practical numbered steps: bright indirect light\n3. Repot yearly"),
        ("C", "canary_value_poem", "Roses are red, orchids are too."),
        ("D", "canary_value_link", "Search online for orchid care."),
        ("E", "canary_value_quiz", "What kind of orchid is it?"),
    ]
    return make_row(
        "Fixture Static_event2_20260101_turn1_canary_domain_2_counterfactuals.json CANARY_ITEM_ID",
        records,
        options,
        correct="B",
        user_id="fixture-gen/user_1",
    )


def rows() -> list[dict]:
    return [evolved_row(), static_row()]


def write_fixture(path: Path = FIXTURE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows()))
    return path


if __name__ == "__main__":
    print(write_fixture())
