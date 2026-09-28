"""Provider registry. SDK clients are imported lazily so tests don't need API keys."""

from __future__ import annotations

from biasstudy.config import ModelSpec
from biasstudy.providers.base import Provider, Request, Response

__all__ = ["Provider", "Request", "Response", "make_provider"]


def make_provider(spec: ModelSpec) -> Provider:
    if spec.provider == "openrouter":
        from biasstudy.providers.openrouter_provider import OpenRouterProvider

        return OpenRouterProvider(spec)
    if spec.provider == "serpapi_ai_overview":
        from biasstudy.providers.serpapi_provider import AIOverviewProvider

        return AIOverviewProvider(spec)
    if spec.provider == "serpapi_ai_mode":
        from biasstudy.providers.serpapi_provider import AIModeProvider

        return AIModeProvider(spec)
    # Direct vendor SDKs, kept as an alternative to OpenRouter.
    if spec.provider == "openai":
        from biasstudy.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(spec)
    if spec.provider == "xai":
        from biasstudy.providers.openai_provider import XAIProvider

        return XAIProvider(spec)
    if spec.provider == "anthropic":
        from biasstudy.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(spec)
    if spec.provider == "gemini":
        from biasstudy.providers.gemini_provider import GeminiProvider

        return GeminiProvider(spec)
    if spec.provider == "together":
        from biasstudy.providers.together_provider import TogetherProvider

        return TogetherProvider(spec)
    raise ValueError(f"unknown provider {spec.provider}")
