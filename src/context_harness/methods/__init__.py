from .agent import WorkspaceAgent
from .base import MemoryMethod, MethodConfig, MethodContext
from .baselines import RecentHistory, RollingSummary
from .profile import ProfileMemory

METHODS = {cls.name: cls for cls in (WorkspaceAgent, RecentHistory, RollingSummary, ProfileMemory)}


def make_method(name: str, ctx: MethodContext) -> MemoryMethod:
    try:
        return METHODS[name](ctx)
    except KeyError:
        raise ValueError(f"Unknown method {name!r}; choose from {sorted(METHODS)}") from None


__all__ = ["METHODS", "MemoryMethod", "MethodConfig", "MethodContext", "make_method"]
