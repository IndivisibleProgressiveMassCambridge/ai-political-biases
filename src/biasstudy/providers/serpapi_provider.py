"""Google AI Overviews and AI Mode, captured through SerpApi.

There is no official API for either product. SerpApi runs the real Google search (from the
race's location) and returns the AI answer as structured blocks, which we render to text.
When Google shows no AI Overview, the response is recorded with shown=False. That is a result
in itself, and is NOT the same as the AI omitting a controversy.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from biasstudy.config import ModelSpec
from biasstudy.providers.base import Request, Response

SEARCH_URL = "https://serpapi.com/search.json"


def render_blocks(blocks: list[dict], depth: int = 0) -> list[str]:
    """Flatten SerpApi text_blocks (paragraph/heading/list/table/expandable/...) to markdown lines."""
    lines: list[str] = []
    indent = "  " * depth
    for b in blocks or []:
        kind = b.get("type")
        if kind == "heading":
            lines.append(f"## {b.get('snippet', '')}")
        elif kind == "paragraph":
            lines.append(b.get("snippet", ""))
        elif kind == "list":
            for item in b.get("list", []):
                head = " ".join(x for x in (item.get("title"), item.get("snippet")) if x)
                lines.append(f"{indent}- {head}")
                if item.get("list"):
                    lines.extend(render_blocks([{"type": "list", "list": item["list"]}], depth + 1))
        elif kind == "expandable":
            if b.get("title"):
                lines.append(f"## {b['title']}")
            lines.extend(render_blocks(b.get("text_blocks", []), depth))
        elif kind == "table":
            lines.extend(" | ".join(row) for row in b.get("table", []))
        elif kind == "comparison":
            labels = b.get("product_labels", [])
            for row in b.get("comparison", []):
                vals = "; ".join(f"{l}: {v}" for l, v in zip(labels, row.get("values", [])))
                lines.append(f"- {row.get('feature', '')}: {vals}")
        elif b.get("snippet"):
            lines.append(b["snippet"])
        # top_stories, videos, images: links rather than AI text; captured in raw only
    return [l for l in lines if l.strip()]


def _references(obj: dict) -> list[dict]:
    return [{"url": r.get("link"), "title": r.get("title"), "source": r.get("source")}
            for r in obj.get("references") or []]


class _SerpApiBase:
    def __init__(self, spec: ModelSpec, client: httpx.AsyncClient | None = None):
        self.spec = spec
        self.client = client or httpx.AsyncClient(timeout=120)
        self.api_key = os.environ.get("SERPAPI_API_KEY", "")

    async def _get(self, params: dict[str, Any]) -> dict:
        # no_cache: each repetition must be a fresh Google search, not SerpApi's cached copy.
        r = await self.client.get(SEARCH_URL, params={**params, "api_key": self.api_key,
                                                      "no_cache": "true", "gl": "us", "hl": "en"})
        r.raise_for_status()
        data = r.json()
        if data.get("error") and "hasn't returned any results" not in data["error"]:
            raise RuntimeError(f"SerpApi error: {data['error']}")
        return data

    @staticmethod
    def _strip_key(data: dict) -> dict:
        meta = dict(data.get("search_parameters") or {})
        meta.pop("api_key", None)
        return {**data, "search_parameters": meta}


class AIOverviewProvider(_SerpApiBase):
    async def complete(self, req: Request) -> Response:
        if not req.search:
            raise ValueError("AI Overviews only exist in the search arm")
        page = await self._get({"engine": "google", "q": req.prompt, "location": req.location})
        aio = page.get("ai_overview") or {}
        raws = {"search": self._strip_key(page)}
        # Google sometimes loads the AI Overview separately; the token expires in ~1 minute.
        if aio.get("page_token") and not aio.get("text_blocks"):
            follow = await self._get({"engine": "google_ai_overview", "page_token": aio["page_token"]})
            raws["ai_overview"] = self._strip_key(follow)
            aio = follow.get("ai_overview") or {}
        shown = bool(aio.get("text_blocks")) and not aio.get("error")
        return Response(
            text="\n".join(render_blocks(aio.get("text_blocks", []))) if shown else "",
            citations=_references(aio),
            model_id_reported="google-ai-overview",
            finish_reason="shown" if shown else ("error: " + aio["error"] if aio.get("error") else "not_shown"),
            shown=shown,
            served_by="serpapi",
            raw=raws,
        )


class AIModeProvider(_SerpApiBase):
    async def complete(self, req: Request) -> Response:
        if not req.search:
            raise ValueError("AI Mode only exists in the search arm")
        data = await self._get({"engine": "google_ai_mode", "q": req.prompt, "location": req.location})
        text = data.get("reconstructed_markdown") or "\n".join(render_blocks(data.get("text_blocks", [])))
        return Response(
            text=text,
            citations=_references(data),
            model_id_reported="google-ai-mode",
            finish_reason="shown" if text.strip() else "not_shown",
            shown=bool(text.strip()),
            served_by="serpapi",
            raw=self._strip_key(data),
        )
