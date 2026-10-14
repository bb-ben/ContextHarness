"""Published per-token prices, used to estimate run cost and enforce --max-cost-usd.

Prices are USD per 1M tokens: (input, cached input, output). Reasoning tokens are
billed as output. Add a model here before capping a run's cost on it.
"""

from __future__ import annotations

from .base import ModelAdapter, ModelRequest, ModelResponse, QuotaExhausted, Usage

PRICES_PER_MTOK: dict[str, tuple[float, float, float]] = {
    # developers.openai.com/api/docs/models/gpt-5.4-mini, checked 2026-10-09
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    # developers.openai.com/api/docs/models/gpt-6.1-sol, checked 2026-10-13
    "gpt-6.1-sol": (2.00, 0.10, 10.00),
}


def cost_usd(usage: Usage, model: str) -> float | None:
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        return None
    inp, cached, out = prices
    return (
        (usage.input_tokens + usage.cache_creation_input_tokens) * inp
        + usage.cache_read_input_tokens * cached
        + usage.output_tokens * out
    ) / 1_000_000


class CostCap:
    """Stops a run once its estimated spend reaches `max_usd`, by pausing it.

    Checked before each request, so the overshoot is at most one request. Caps made with
    `shared_with` draw on the same budget (e.g. an answering model and its judge).
    """

    def __init__(self, inner: ModelAdapter, max_usd: float, shared_with: CostCap | None = None):
        if inner.settings.model not in PRICES_PER_MTOK:
            raise ValueError(f"No published price recorded for {inner.settings.model}; add it to models/pricing.py")
        self.inner, self.max_usd, self.settings = inner, max_usd, inner.settings
        self._ledger = shared_with._ledger if shared_with else [0.0]

    @property
    def spent(self) -> float:
        return self._ledger[0]

    def complete(self, request: ModelRequest) -> ModelResponse:
        if self.spent >= self.max_usd:
            raise QuotaExhausted(f"cost cap of ${self.max_usd:.2f} reached (spent ${self.spent:.3f})")
        response = self.inner.complete(request)
        self._ledger[0] += cost_usd(response.usage, self.settings.model) or 0.0
        return response
