from .base import (
    ModelAdapter,
    ModelRequest,
    ModelResponse,
    ModelSettings,
    PauseRun,
    ProviderUnavailable,
    QuotaExhausted,
    RequestCap,
    TextBlock,
    ToolCall,
    ToolSpec,
    Usage,
)


def make_model(settings: ModelSettings) -> ModelAdapter:
    """`fake` is the offline model; `gpt-*`/`o*` use OpenAI; `gemini-*` uses Google; else Anthropic."""
    if settings.model == "fake":
        from .fake import FakeModel

        return FakeModel(settings)
    if settings.model.startswith(("gpt-", "o1", "o3", "o4")):
        from .openai_adapter import OpenAIAdapter

        return OpenAIAdapter(settings)
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
    "PauseRun",
    "ProviderUnavailable",
    "QuotaExhausted",
    "RequestCap",
    "TextBlock",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "make_model",
]
