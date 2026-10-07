import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import fixture_data  # noqa: E402

from context_harness.dataset import parse_item  # noqa: E402
from context_harness.methods import MethodConfig  # noqa: E402
from context_harness.models.base import ModelSettings  # noqa: E402
from context_harness.orchestrator import RunConfig  # noqa: E402


@pytest.fixture
def evolved_item():
    return parse_item(fixture_data.evolved_row(), index=0)


@pytest.fixture
def static_item():
    return parse_item(fixture_data.static_row(), index=1)


@pytest.fixture
def items(evolved_item, static_item):
    return [evolved_item, static_item]


def run_config(method: str = "agent", **method_kwargs) -> RunConfig:
    return RunConfig(
        method=method,
        settings=ModelSettings(model="fake", effort=None, fallbacks=False),
        method_config=MethodConfig(**method_kwargs),
    )


def ev(request) -> int | None:
    """Replay index of the event a model request was made for (None in the answer phase)."""
    event = request.metadata.get("event_id")
    return int(event.split("-")[1]) if event else None
