"""The profile method: an evidence log plus one profile rewritten every event.

The structured agent built current state by editing fine-grained entries one event at
a time, and every failure had the same shape: an old belief survived because the agent
never made the one decision (replace, not merge; lasting, not temporary; this
dimension, not that one) that would have removed it. A rolling summary avoids that by
rewriting "who this person is now" every event, but keeps no evidence trail.

This method does both with one call per event and a short prompt:
- an append-only episode log, with observations numbered by the harness
  (`O-<event>.<n>`), so every claim can cite its evidence;
- one profile document, rewritten each event to describe the person now, with dated
  "Changes" lines and observation IDs cited inline;
- every version of the profile, kept by the harness as the change history.
"""

from __future__ import annotations

import difflib
import json
from typing import Any

from .. import prompts
from ..models.base import ModelRequest, TextBlock, ToolSpec
from ..tools import SubmitAnswer, ToolExecutor
from ..types import AnswerResult, Event, Probe, Question, Session
from .base import MethodContext, answer_with_tools, fit_to_limit

KINDS = ("stated", "behavior", "reaction", "event")
RECENT_EPISODE_CHARS = 5000  # the latest episode lines shown with each update

PROFILE_SYSTEM = """\
You keep the long-term memory an assistant has of one person, in two parts:
- an episode log: what happened in each conversation, kept as evidence;
- a profile: who the person is now, which the assistant reads before every reply.

After each conversation, call record once with:
1. episode: one line on what the conversation was about.
2. observations: 0 to 3 things it showed about the person (their preferences,
   reactions, circumstances, life events, lasting facts). Each has a kind: stated
   (they said it about themselves or how they want things), behavior (what they asked
   for or did), reaction (how they responded), or event (something in their life).
   They are numbered {next_ids} in the order you list them.
3. profile: the complete rewritten profile.

The profile describes the person as they are now, in these sections:
## Facts: lasting facts such as diet, health, household, work. Keep a fact even if it
   came up only in passing.
## How they want help: format, length, tone, and what kind of suggestions fit them.
   Where it differs by context, say so ("client documents: ...; family meals: ...").
## Goals
## Right now: temporary circumstances, each with roughly when it should end. Drop
   the ones that have ended.
## Changes: one dated line per change, e.g. "2025-07-10: answers now short (3 lines);
   were detailed." Keep the most recent ones.

Rules:
- When the evidence shows they have changed, replace the old description; don't keep
  both. Record the change under Changes.
- Cite the observations behind each claim in brackets, e.g. [O-5.1, O-6.2]. Weigh the
  recent episodes shown: repeated recent evidence outweighs older evidence.
- A one-off request about a single task doesn't change how they want help.
- Stay under {max_chars} characters."""

PROFILE_ANSWER_SYSTEM = """\
You are an AI assistant with an ongoing relationship with one user. You are partway through a \
conversation with them, shown in <current_conversation>. Earlier sessions are not shown. \
<profile> is your memory of who this user is now, updated after every earlier session, with \
the evidence for each point cited by observation ID.

{answer_task}"""


def _record_spec() -> ToolSpec:
    return ToolSpec("record", "Record this conversation and the rewritten profile.", {
        "type": "object",
        "properties": {
            "episode": {"type": "string", "description": "One line on what the conversation was about."},
            "observations": {"type": "array", "items": {
                "type": "object",
                "properties": {"kind": {"type": "string", "enum": list(KINDS)}, "text": {"type": "string"}},
                "required": ["kind", "text"]}},
            "profile": {"type": "string", "description": "The complete rewritten profile."},
        },
        "required": ["episode", "observations", "profile"],
    })


class ProfileMemory:
    name = "profile"
    has_workspace = False

    def __init__(self, ctx: MethodContext):
        self.ctx = ctx
        self.profile = ""
        self.episodes: list[str] = []
        self.max_chars = ctx.config.summary_max_chars
        self.versions_path = ctx.item_dir / "profile_versions.jsonl"

    def _recent_episodes(self) -> str:
        lines, used = [], 0
        for line in reversed(self.episodes):
            if used + len(line) > RECENT_EPISODE_CHARS:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(reversed(lines)) or "(none yet)"

    def observe(self, event: Event) -> None:
        k = event.index
        next_ids = ", ".join(f"O-{k}.{n}" for n in (1, 2, 3))
        request = ModelRequest(
            phase="profile_update",
            system=PROFILE_SYSTEM.format(next_ids=next_ids, max_chars=self.max_chars),
            blocks=(TextBlock(
                f"<profile>\n{self.profile or '(empty)'}\n</profile>\n\n"
                f"<recent_episodes>\n{self._recent_episodes()}\n</recent_episodes>\n\n"
                + prompts.render_event(event)
                + "\n\nCall record with the episode, its observations and the complete rewritten profile."),),
            tools=(_record_spec(),),
            metadata={"event_id": event.event_id},
        )
        args = None
        for _ in range(2):  # one retry if the model answers without the tool
            response = self.ctx.model.complete(request)
            call = next((c for c in response.tool_calls if c.name == "record"), None)
            if call is not None and isinstance(call.arguments.get("profile"), str):
                args = call.arguments
                break
        date = event.date[:10]
        observations = [o for o in (args or {}).get("observations", []) if isinstance(o, dict)][:3]
        self.episodes.append(f"- E-{k} | {date} | source: {event.event_id} | "
                             f"{(args or {}).get('episode', '(not recorded)').strip()}")
        for n, o in enumerate(observations, 1):
            kind = o.get("kind") if o.get("kind") in KINDS else "behavior"
            self.episodes.append(f"  - O-{k}.{n} | {kind} | {str(o.get('text', '')).strip()}")
        if not args or not args["profile"].strip():
            self.ctx.trace.emit("profile_unchanged", event_id=event.event_id)
            return
        new, fit = fit_to_limit(self.ctx, args["profile"].strip(), self.max_chars, "profile", event.event_id)
        diff = "".join(difflib.unified_diff(self.profile.splitlines(keepends=True), new.splitlines(keepends=True),
                                            "a/profile", "b/profile"))
        self.ctx.trace.emit("profile_update", event_id=event.event_id, chars=len(new), **fit,
                            observations=len(observations), diff=diff)
        self.profile = new
        with self.versions_path.open("a") as f:  # the profile's history, one version per event
            f.write(json.dumps({"event_id": event.event_id, "date": date, "profile": new}) + "\n")

    def answer(self, probe: Probe, question: Question, probe_event_id: str) -> AnswerResult:
        submit = SubmitAnswer(question.letters)
        fixed = (f"<profile>\n{self.profile or '(empty)'}\n</profile>\n\n" + prompts.render_probe(probe, probe_event_id)
                 + "\n\n" + prompts.render_question(question))
        return answer_with_tools(
            ctx=self.ctx, executor=ToolExecutor([submit], self.ctx.trace), submit=submit,
            system=PROFILE_ANSWER_SYSTEM.format(answer_task=prompts.ANSWER_TASK), fixed_text=fixed,
            render_dynamic=lambda actions, calls_left: prompts.render_actions(actions)
            + "\n\nCall submit_answer with your choice.",
        )

    def memory_snapshot(self) -> dict[str, str]:
        return {"profile.md": self.profile, "memory/episodes.md": "\n".join(self.episodes) + "\n"}

    def export_state(self) -> dict[str, Any]:
        return {"profile": self.profile, "episodes": self.episodes}

    def restore_state(self, state: dict[str, Any], history: tuple[Session, ...]) -> None:
        self.profile, self.episodes = state["profile"], list(state["episodes"])
