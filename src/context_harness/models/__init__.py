from .base import (
    ModelAdapter,
    ModelRequest,
    ModelResponse,
    ModelSettings,
    QuotaExhausted,
    RequestCap,
    TextBlock,
    ToolCall,
    ToolSpec,
    Usage,
)


def make_model(settings: ModelSettings) -> ModelAdapter:
    """`fake` is the offline model; `gemini-*` uses Google; anything else is an Anthropic model id."""
    if settings.model == "fake":
        from .fake import FakeModel

        return FakeModel(settings)
    if settings.model.startswith("gemini-"):
        from .gemini_adapter import GeminiAdapter

        return GeminiAdapter(settings)
    from .anthropic_adapter import AnthropicAdapter

    return AnthropicAdapter(settings)


__all__ = [
    "ModelAdapter",
    "ModelRequest",
    "ModelResponse",
    "ModelSettings",
    "QuotaExhausted",
    "RequestCap",
    "TextBlock",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "make_model",
]
