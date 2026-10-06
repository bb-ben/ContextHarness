"""All prompt text and rendering in one place, so prompt changes are easy to audit.

Prompts describe the task generically (keep memory of a user whose preferences
can change). They never mention the benchmark, multiple-choice evaluation, or
preference domains, and nothing here writes preferences on the agent's behalf.
"""

from __future__ import annotations

from .types import Probe, Question, Session
from .workspace import FILE_PURPOSES

TRACKING_GUIDANCE = (
    "People's preferences, goals and circumstances can change over time. Keep track of what "
    "is currently true. When something changes, record the new state as current and mark the "
    "old one as superseded, with dates."
)

AGENT_UPDATE_SYSTEM = """\
You manage the memory of an AI assistant that has an ongoing relationship with one user. You \
do not talk to the user. After each conversation session between the user and the assistant, \
you are shown that session and may update a small set of memory files. In later sessions the \
assistant will rely on this memory to respond the way this user wants at that time.

Each call shows you:
- <session>: the session that just ended (its date, an optional scenario note, and the turns).
- <working_context>: the full current contents of context.md. It is the only file shown \
automatically, and it is refreshed after every edit. Limit: {context_max_chars} characters.
- <workspace>: every memory file with its size and limit. Use read_file to see any other file.
- <actions_so_far>: the tools you have already called for this session, with their results.

Memory files:
{file_list}

How to work:
- {tracking_guidance}
- Tie what you record to the session date. Write concise, specific notes, not transcript copies.
- Doing nothing is fine when a session adds nothing worth remembering.
- You may make several tool calls in one turn. When you are finished with this session, reply \
briefly without calling any tools. You have at most {max_steps} model calls per session.
- A write that would push a file over its limit is rejected."""

ANSWER_TASK = (
    "You will be shown candidate next responses. Choose the one that best fits this user's "
    "preferences as they are now, then call submit_answer with its letter."
)

AGENT_ANSWER_SYSTEM = """\
You are an AI assistant with an ongoing relationship with one user. You are partway through a \
conversation with them, shown in <current_conversation>. Earlier sessions are not shown. What \
you know about the user comes from your memory files: context.md is shown in \
<working_context>, and you can read the other files with read_file. Memory is read-only now, \
and you can read at most {read_budget} characters in total.

{answer_task}"""

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


def agent_update_system(context_max_chars: int, max_steps: int) -> str:
    file_list = "\n".join(f"- {path}: {purpose}" for path, purpose in FILE_PURPOSES.items())
    return AGENT_UPDATE_SYSTEM.format(
        context_max_chars=context_max_chars,
        file_list=file_list,
        tracking_guidance=TRACKING_GUIDANCE,
        max_steps=max_steps,
    )


def _speaker(role: str) -> str:
    return "User" if role == "user" else "Assistant"


def render_session(s: Session) -> str:
    lines = [f'<session date="{s.date}">']
    if s.scenario:
        lines.append(f"Scenario: {s.scenario}")
    lines += [f"{_speaker(t.role)}: {t.content}" for t in s.turns]
    lines.append("</session>")
    return "\n".join(lines)


def render_probe(p: Probe) -> str:
    lines = [f'<current_conversation date="{p.date}">']
    if p.scenario:
        lines.append(f"Scenario: {p.scenario}")
    lines += [f"{_speaker(t.role)}: {t.content}" for t in p.turns]
    lines.append("</current_conversation>")
    return "\n".join(lines)


def render_question(q: Question) -> str:
    body = "\n".join(f"{o.letter}: {o.text}" for o in q.options)
    return f"<question>\n{q.stem}\n{body}\n</question>"


def render_working_context(content: str, limit: int) -> str:
    return (
        f'<working_context file="context.md" chars="{len(content)}" limit="{limit}">\n'
        f"{content}\n</working_context>"
    )


def render_workspace(files: list[dict]) -> str:
    rows = "\n".join(f"{f['path']}  ({f['chars']}/{f['limit']} chars)" for f in files)
    return f"<workspace>\n{rows}\n</workspace>"


def render_actions(actions_text: str) -> str:
    return f"<actions_so_far>\n{actions_text or '(none yet)'}\n</actions_so_far>"
