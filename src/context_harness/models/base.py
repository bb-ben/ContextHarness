"""Provider-neutral model interface.

Every model invocation in this harness is a *single-turn* request: a system
prompt plus one user message made of text blocks. Multi-step tool use is
driven by the caller, which rebuilds the request from explicit state (session,
current context.md, actions taken so far) on every step. That keeps each model
input fully determined by logged state, lets edits to context.md take effect on
the very next call, and never rewrites conversation history (which would break
prompt caching and preserved thinking on current Claude models).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class TextBlock:
    text: str
    cache: bool = False  # place a prompt-cache breakpoint after this block


@dataclass(frozen=True)
class ModelRequest:
    phase: str  # e.g. "memory_update", "answer", "summary_update"; for logging only
    system: str
    blocks: tuple[TextBlock, ...]
    tools: tuple[ToolSpec, ...] = ()
    # Harness bookkeeping (session index, step). Never contains labels; adapters
    # must not send it to the provider.
    metadata: dict[str, Any] = field(default_factory=dict)

    def prompt_text(self) -> str:
        return self.system + "\n" + "\n".join(b.text for b in self.blocks)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(*(a + b for a, b in zip(asdict(self).values(), asdict(other).values())))

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens


@dataclass(frozen=True)
class ModelResponse:
    text: str
    tool_calls: tuple[ToolCall, ...]
    stop_reason: str
    usage: Usage
    model: str  # model that actually served the request (differs on fallback)
    latency_s: float
    request_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSettings:
    """Explicit, logged generation settings shared by every method in a run."""

    model: str
    max_tokens: int = 16000
    effort: str | None = "medium"  # Anthropic output_config.effort / Gemini thinking_level; None to omit
    thinking: dict[str, Any] | None = None  # None = provider/model default
    temperature: float | None = None  # rejected by Claude Opus 5.5; leave None
    fallbacks: bool = True  # server-side refusal fallback ("default" routing); Anthropic only
    max_retries: int = 5
    timeout_s: float = 600.0
    requests_per_minute: float | None = None  # client-side pacing, e.g. for free-tier quotas


class PauseRun(Exception):
    """A condition that should pause the run rather than fail the item.

    The orchestrator stops at the last completed event; --resume continues
    from there.
    """


class QuotaExhausted(PauseRun):
    """The provider's quota for the current period (or this run's request cap) is used up."""


class ProviderUnavailable(PauseRun):
    """The provider kept failing (overload, server errors, dropped connections) through all retries."""


class ModelAdapter(Protocol):
    settings: ModelSettings

    def complete(self, request: ModelRequest) -> ModelResponse: ...


class RequestCap:
    """Hard ceiling on model requests for one run, as a guard against runaway usage.

    Raises QuotaExhausted once `limit` requests have been made, so the run pauses
    at the last completed event and can be continued with --resume.
    """

    def __init__(self, inner: "ModelAdapter", limit: int):
        self.inner, self.limit, self.settings, self.count = inner, limit, inner.settings, 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        if self.count >= self.limit:
            raise QuotaExhausted(f"request cap of {self.limit} reached for this run")
        self.count += 1
        return self.inner.complete(request)
