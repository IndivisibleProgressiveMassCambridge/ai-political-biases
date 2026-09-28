"""Open-weights Llama via Together's OpenAI-compatible chat completions. No search arm."""

from __future__ import annotations

import os

from openai import AsyncOpenAI

from biasstudy.config import ModelSpec
from biasstudy.providers.base import Request, Response


class TogetherProvider:
    def __init__(self, spec: ModelSpec, client: AsyncOpenAI | None = None):
        self.spec = spec
        self.client = client or AsyncOpenAI(
            base_url="https://api.together.xyz/v1", api_key=os.environ.get("TOGETHER_API_KEY")
        )

    async def complete(self, req: Request) -> Response:
        if req.search:
            raise ValueError(f"{self.spec.key} has no native search; search arm not supported")
        kwargs: dict = {
            "model": self.spec.model_id,
            "messages": [{"role": "user", "content": req.prompt}],
            "max_tokens": req.max_output_tokens,
        }
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        r = await self.client.chat.completions.create(**kwargs)
        choice = r.choices[0]
        return Response(
            text=choice.message.content or "",
            model_id_reported=r.model,
            finish_reason=choice.finish_reason,
            usage=r.usage.model_dump() if r.usage else {},
            raw=r.model_dump(mode="json"),
        )
