"""Typed records shared across the harness.

The split between `Episode` (agent-visible) and `GoldLabel` (evaluator-only) is
the leakage boundary: methods only ever receive pieces of an `Episode`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["user", "assistant"]


class Event(Protocol):
    """Anything the memory layer observes, in time order.

    HorizonBench supplies conversation sessions. Other sources (e.g. FoodBody
    logs, decisions, outcomes) can implement this protocol plus a renderer in
    prompts.render_event, without changing the workspace or checks.
    """

    @property
    def event_id(self) -> str: ...

    @property
    def event_type(self) -> str: ...

    @property
    def date(self) -> str: ...


@dataclass(frozen=True)
class Turn:
    role: Role
    content: str


@dataclass(frozen=True)
class Session:
    """One dated historical session, replayed to the agent as an observation."""

    index: int  # position in replay order
    source_index: int  # position in the original conversation text
    date: str  # ISO timestamp exactly as given in the data
    scenario: str  # generator narration; "" if absent or hidden
    turns: tuple[Turn, ...]

    event_type = "conversation"

    @property
    def event_id(self) -> str:
        return f"S-{self.index}"


@dataclass(frozen=True)
class Probe:
    """The truncated target conversation the question is asked about.

    Shown only at question time; it ends with the user turn the options respond to.
    """

    date: str
    scenario: str
    turns: tuple[Turn, ...]


@dataclass(frozen=True)
class Option:
    letter: str
    text: str


@dataclass(frozen=True)
class Question:
    stem: str
    options: tuple[Option, ...]

    @property
    def letters(self) -> tuple[str, ...]:
        return tuple(o.letter for o in self.options)


@dataclass(frozen=True)
class Episode:
    """Everything the agent may ever see for one benchmark item."""

    item_key: str  # opaque; the raw HB id leaks the preference domain
    sessions: tuple[Session, ...]
    probe: Probe
    question: Question


@dataclass(frozen=True)
class GoldLabel:
    """Evaluator-only fields. Never passed to a method, model, or tool."""

    item_id: str
    generator: str
    user_id: str
    correct_letter: str
    distractor_letter: str | None
    has_evolved: bool
    preference_domain: str
    option_values: dict[str, str]
    preference_evolution: dict[str, Any] | None


@dataclass(frozen=True)
class OrderingReport:
    policy: str
    n_history_sessions_in_text: int
    n_replayed: int
    text_order_monotonic: bool
    n_moved: int  # sessions whose replay position differs from text position
    dropped_after_probe: tuple[str, ...]  # dates of sessions dated after the probe
    truncated_to_last: int | None = None  # set when --max-sessions limited replay


@dataclass(frozen=True)
class BenchmarkItem:
    index: int
    episode: Episode
    gold: GoldLabel
    ordering: OrderingReport
    conversation_sha256: str


@dataclass
class AnswerResult:
    letter: str | None
    via: Literal["tool", "text_fallback", "none"]
    raw_text: str = ""
    details: dict[str, Any] = field(default_factory=dict)
