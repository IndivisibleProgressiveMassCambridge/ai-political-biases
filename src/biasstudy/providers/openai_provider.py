"""OpenAI (Responses API) and xAI (OpenAI-compatible Responses API)."""

from __future__ import annotations

import os

from openai import AsyncOpenAI

from biasstudy.config import ModelSpec
from biasstudy.providers.base import Request, Response


class OpenAIProvider:
    def __init__(self, spec: ModelSpec, client: AsyncOpenAI | None = None):
        self.spec = spec
        self.client = client or AsyncOpenAI()

    async def complete(self, req: Request) -> Response:
        kwargs: dict = {
            "model": self.spec.model_id,
            "input": req.prompt,
            "max_output_tokens": req.max_output_tokens,
        }
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.search:
            kwargs["tools"] = [{"type": "web_search"}]
        r = await self.client.responses.create(**kwargs)

        citations = []
        for item in r.output or []:
            if getattr(item, "type", None) != "message":
                continue
            for part in item.content or []:
                for ann in getattr(part, "annotations", None) or []:
                    if getattr(ann, "type", None) == "url_citation":
                        citations.append({"url": ann.url, "title": getattr(ann, "title", None)})
        return Response(
            text=r.output_text or "",
            citations=citations,
            model_id_reported=r.model,
            finish_reason=r.status,
            usage=r.usage.model_dump() if r.usage else {},
            raw=r.model_dump(mode="json"),
        )


class XAIProvider(OpenAIProvider):
    def __init__(self, spec: ModelSpec, client: AsyncOpenAI | None = None):
        super().__init__(
            spec,
            client or AsyncOpenAI(base_url="https://api.x.ai/v1", api_key=os.environ.get("XAI_API_KEY")),
        )
