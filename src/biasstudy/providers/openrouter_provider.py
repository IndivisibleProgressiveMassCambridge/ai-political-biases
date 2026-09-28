"""All chat models through OpenRouter (one key), pinned to the first-party provider.

Search arm uses OpenRouter's web plugin with engine="native", so each vendor's own search runs
(OpenAI, Anthropic, Google, xAI). We never let it fall back to a third-party search engine.
"""

from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI

from biasstudy.config import ModelSpec
from biasstudy.providers.base import Request, Response

BASE_URL = "https://openrouter.ai/api/v1"


def openrouter_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=BASE_URL, api_key=os.environ.get("OPENROUTER_API_KEY"))


def routing(route: str | None) -> dict[str, Any]:
    return {"provider": {"order": [route], "allow_fallbacks": False}} if route else {}


def _citations(message: dict) -> list[dict]:
    out = []
    for ann in message.get("annotations") or []:
        if ann.get("type") == "url_citation":
            c = ann.get("url_citation") or ann
            out.append({"url": c.get("url"), "title": c.get("title")})
    return out


class OpenRouterProvider:
    def __init__(self, spec: ModelSpec, client: AsyncOpenAI | None = None):
        if not spec.route:
            raise ValueError(f"{spec.key}: set `route` so requests stay on the first-party provider")
        self.spec = spec
        self.client = client or openrouter_client()

    async def complete(self, req: Request) -> Response:
        extra_body = routing(self.spec.route)
        if req.search:
            extra_body["plugins"] = [{"id": "web", "engine": "native"}]
        kwargs: dict = {
            "model": self.spec.model_id,
            "messages": [{"role": "user", "content": req.prompt}],
            "max_tokens": req.max_output_tokens,
            "extra_body": extra_body,
        }
        if req.temperature is not None and self.spec.supports_temperature:
            kwargs["temperature"] = req.temperature
        r = await self.client.chat.completions.create(**kwargs)
        raw = r.model_dump(mode="json")
        choice = raw["choices"][0]
        return Response(
            text=choice["message"].get("content") or "",
            citations=_citations(choice["message"]),
            model_id_reported=raw.get("model"),
            finish_reason=choice.get("finish_reason"),
            served_by=raw.get("provider"),
            usage=raw.get("usage") or {},
            raw=raw,
        )
