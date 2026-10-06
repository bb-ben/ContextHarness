"""Parsing helpers vendored verbatim from HorizonBench's evaluate.py, plus
`parse_sessions`, a guarded variant the harness uses (see its docstring).

Source: https://github.com/stellalisy/HorizonBench (Apache-2.0),
commit 4b5076b147c952499b7921f20f811dd5aca7ef0b. Vendored rather than imported
because HorizonBench's evaluate.py is a top-level script, not an installable
package, and importing it pulls in `datasets`/`litellm`. Keep these byte-for-byte
compatible with upstream so parsing and letter extraction match the benchmark's.
"""

import re

HB_COMMIT = "4b5076b147c952499b7921f20f811dd5aca7ef0b"

QUESTION_STEM = "Which of the following responses would align with the user's preferences the most?"


def parse_conversations(conversation_text: str) -> list[dict]:
    segments = []
    current_segment = None

    for line in conversation_text.split("\n"):
        if line.startswith("Conversation History:"):
            continue
        elif line.startswith("Date: "):
            if current_segment is not None:
                segments.append(current_segment)
            current_segment = {"date": line[6:], "scenario": "", "turns": []}
        elif line.startswith("Scenario: ") and current_segment is not None:
            current_segment["scenario"] = line[10:]
        elif line.startswith("User: ") and current_segment is not None:
            current_segment["turns"].append({"role": "user", "content": line[6:]})
        elif line.startswith("Assistant: ") and current_segment is not None:
            current_segment["turns"].append({"role": "assistant", "content": line[11:]})
        elif current_segment is not None and current_segment["turns"]:
            current_segment["turns"][-1]["content"] += "\n" + line

    if current_segment is not None:
        segments.append(current_segment)
    return segments


_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def parse_sessions(conversation_text: str) -> list[dict]:
    """parse_conversations with two guards against text inside turns.

    Upstream treats any line starting with "Date: " as a new session and any
    "Scenario: " line as the current session's scenario. About 1% of benchmark
    items have turn text such as "Date: YYYY-MM-DD", which upstream splits into
    bogus sessions. Here a "Date: " line starts a session only when it carries an
    ISO timestamp, and "Scenario: " is recognized only before a session's first
    turn. On every other item the output is identical to upstream's.
    """
    segments = []
    current_segment = None

    for line in conversation_text.split("\n"):
        if line.startswith("Conversation History:"):
            continue
        elif line.startswith("Date: ") and _TIMESTAMP.match(line[6:]):
            if current_segment is not None:
                segments.append(current_segment)
            current_segment = {"date": line[6:], "scenario": "", "turns": []}
        elif line.startswith("Scenario: ") and current_segment is not None and not current_segment["turns"]:
            current_segment["scenario"] = line[10:]
        elif line.startswith("User: ") and current_segment is not None:
            current_segment["turns"].append({"role": "user", "content": line[6:]})
        elif line.startswith("Assistant: ") and current_segment is not None:
            current_segment["turns"].append({"role": "assistant", "content": line[11:]})
        elif current_segment is not None and current_segment["turns"]:
            current_segment["turns"][-1]["content"] += "\n" + line

    if current_segment is not None:
        segments.append(current_segment)
    return segments


def extract_letter(response_text: str) -> str:
    """Extract the selected letter (A-E) from a model response."""
    text = response_text.strip()
    if text and text[0] in "ABCDE":
        return text[0]
    match = re.search(r"\b([A-E])\b", text)
    return match.group(1) if match else ""
