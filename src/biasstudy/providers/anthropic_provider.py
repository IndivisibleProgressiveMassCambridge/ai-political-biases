"""Anthropic Messages API, with the server-side web search tool for the search arm."""

from __future__ import annotations

from anthropic import AsyncAnthropic

from biasstudy.config import ModelSpec
from biasstudy.providers.base import Request, Response

WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": 5}
MAX_CONTINUATIONS = 5


class AnthropicProvider:
    def __init__(self, spec: ModelSpec, client: AsyncAnthropic | None = None):
        self.spec = spec
        self.client = client or AsyncAnthropic()

    async def complete(self, req: Request) -> Response:
        messages: list[dict] = [{"role": "user", "content": req.prompt}]
        kwargs: dict = {"model": self.spec.model_id, "max_tokens": req.max_output_tokens}
        # Current Claude models reject temperature; only pass it where the config allows.
        if req.temperature is not None and self.spec.supports_temperature:
            kwargs["temperature"] = req.temperature
        if req.search:
            kwargs["tools"] = [WEB_SEARCH_TOOL]

        content: list = []
        raws: list[dict] = []
        r = None
        # Server tools can end a turn with pause_turn; resend to let the model continue.
        for _ in range(MAX_CONTINUATIONS):
            r = await self.client.messages.create(messages=messages, **kwargs)
            raws.append(r.model_dump(mode="json"))
            content.extend(r.content)
            if r.stop_reason != "pause_turn":
                break
            messages = [*messages, {"role": "assistant", "content": r.content}]

        text_parts, citations = [], []
        for block in content:
            if block.type != "text":
                continue
            text_parts.append(block.text)
            for c in getattr(block, "citations", None) or []:
                if getattr(c, "type", None) == "web_search_result_location":
                    citations.append({"url": c.url, "title": getattr(c, "title", None)})
        return Response(
            text="".join(text_parts),
            citations=citations,
            model_id_reported=r.model,
            finish_reason=r.stop_reason,
            usage=r.usage.model_dump() if r.usage else {},
            raw={"turns": raws},
        )
