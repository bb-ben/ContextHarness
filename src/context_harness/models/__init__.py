from .base import ModelAdapter, ModelRequest, ModelResponse, ModelSettings, TextBlock, ToolCall, ToolSpec, Usage


def make_model(settings: ModelSettings) -> ModelAdapter:
    """`fake` selects the deterministic offline model; anything else is an Anthropic model id."""
    if settings.model == "fake":
        from .fake import FakeModel

        return FakeModel(settings)
    from .anthropic_adapter import AnthropicAdapter

    return AnthropicAdapter(settings)


__all__ = [
    "ModelAdapter",
    "ModelRequest",
    "ModelResponse",
    "ModelSettings",
    "TextBlock",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "make_model",
]
