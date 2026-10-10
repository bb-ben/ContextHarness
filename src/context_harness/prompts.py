"""Prompt text and rendering.

The agent's system prompt is agent.md, loaded verbatim, so the documented memory
contract and the instructions the model receives cannot drift apart. Phase
instructions (process an event vs. respond) go in the per-call dynamic block.
Baseline prompts live here too and share the generic tracking guidance.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .entries import Entry
from .types import Probe, Question, Session

CONTRACT_PATH = Path(__file__).resolve().parents[2] / "agent.md"


@lru_cache(maxsize=4)
def load_contract(path: str | None = None) -> str:
    target = Path(path) if path else CONTRACT_PATH
    if not target.exists():
        raise FileNotFoundError(f"Memory contract not found at {target}")
    return target.read_text()


UPDATE_INSTRUCTION = (
    "Process the event above as the contract describes, in one turn if you can: append its "
    "episode block, resolve anything under <due>, update hypotheses and state, rewrite "
    "context.md, and call finish together with those edits."
)


def answer_instruction(read_budget_left: int) -> str:
    return (
        "You are now responding to the person in <current_conversation>. Memory is read-only. "
        f"You can read at most {read_budget_left} more characters with read_file and lookup. "
        "Choose the candidate response that best fits this person as they are now, then call "
        "submit_answer with its letter."
    )


# How the agent should use memory when it responds. "default" adds nothing.
ANSWER_STRATEGIES: dict[str, str] = {
    "default": "",
    "dimension-first": (
        "Work in this order: (1) compare the candidate responses and find where they actually differ; "
        "(2) name the preference dimension or dimensions those differences fall on; (3) read the current "
        "value of each in <preference_dimensions>, and if it was revised, look up what it replaced and why; "
        "(4) choose the response that best fits those current values. Weigh the dimension the responses "
        "differ on, not whichever preference is most prominent in context.md."
    ),
}


TRACKING_GUIDANCE = (
    "People's preferences, goals and circumstances can change over time. Keep track of what "
    "is currently true. When something changes, record the new state as current and mark the "
    "old one as superseded, with dates."
)

ANSWER_TASK = (
    "You will be shown candidate next responses. Choose the one that best fits this user's "
    "preferences as they are now, then call submit_answer with its letter."
)

RECENT_ANSWER_SYSTEM = """\
You are an AI assistant with an ongoing relationship with one user. You are partway through a \
conversation with them, shown in <current_conversation>. <recent_history> contains the most \
recent earlier sessions that fit within {budget} characters; anything older is not shown.

{answer_task}"""

SUMMARY_ANSWER_SYSTEM = """\
You are an AI assistant with an ongoing relationship with one user. You are partway through a \
conversation with them, shown in <current_conversation>. Earlier sessions are not shown. \
<summary> is a running summary of what you know about the user, updated after every earlier \
session.

{answer_task}"""

SUMMARY_UPDATE_SYSTEM = """\
You maintain a running summary of what an AI assistant needs to know about one user, written \
from the user's conversation sessions with the assistant. Later, the assistant will rely on \
this summary alone to respond the way this user wants at that time.

You receive the current summary and the session that just ended. Write the complete updated \
summary. {tracking_guidance} Stay under {max_chars} characters; anything longer is cut off. \
Output only the summary text."""


def _speaker(role: str) -> str:
    return "User" if role == "user" else "Assistant"


def render_event(s: Session) -> str:
    """Render one event. Conversation sessions are the only event type so far."""
    lines = [f'<event id="{s.event_id}" type="{s.event_type}" date="{s.date[:10]}">']
    if s.scenario:
        lines.append(f"Scenario: {s.scenario}")
    lines += [f"{_speaker(t.role)}: {t.content}" for t in s.turns]
    lines.append("</event>")
    return "\n".join(lines)


def render_probe(p: Probe, event_id: str) -> str:
    lines = [f'<current_conversation id="{event_id}" date="{p.date[:10]}">']
    if p.scenario:
        lines.append(f"Scenario: {p.scenario}")
    lines += [f"{_speaker(t.role)}: {t.content}" for t in p.turns]
    lines.append("</current_conversation>")
    return "\n".join(lines)


def render_question(q: Question) -> str:
    body = "\n".join(f"{o.letter}: {o.text}" for o in q.options)
    return f"<question>\n{q.stem}\n{body}\n</question>"


def render_working_context(content: str, limit: int) -> str:
    return f'<working_context file="context.md" chars="{len(content)}" limit="{limit}">\n{content}\n</working_context>'


def render_workspace(files: list[dict]) -> str:
    rows = "\n".join(f"{f['path']}  ({f['chars']}/{f['limit']} chars)" for f in files)
    return f"<workspace>\n{rows}\n</workspace>"


def render_state(files: dict[str, str]) -> str:
    body = "\n".join(f'<file path="{path}">\n{content.rstrip()}\n</file>' for path, content in files.items())
    return f"<state>\n{body}\n</state>"


def render_due(due: list[tuple[Entry, str]]) -> str:
    body = "\n".join(line for _, line in due) if due else "(none)"
    return f"<due>\n{body}\n</due>"


def render_actions(actions_text: str) -> str:
    return f"<actions_so_far>\n{actions_text or '(none yet)'}\n</actions_so_far>"


# Baselines render sessions the same way the agent sees events.
render_session = render_event
