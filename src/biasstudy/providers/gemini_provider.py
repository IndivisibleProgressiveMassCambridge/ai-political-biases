"""Google Gemini via google-genai, with Google Search grounding for the search arm."""

from __future__ import annotations

from google import genai
from google.genai import types

from biasstudy.config import ModelSpec
from biasstudy.providers.base import Request, Response


class GeminiProvider:
    def __init__(self, spec: ModelSpec, client: genai.Client | None = None):
        self.spec = spec
        self.client = client or genai.Client()  # reads GEMINI_API_KEY

    async def complete(self, req: Request) -> Response:
        # Default safety settings on purpose: we are measuring out-of-the-box behavior.
        config = types.GenerateContentConfig(
            max_output_tokens=req.max_output_tokens,
            temperature=req.temperature,
            tools=[types.Tool(google_search=types.GoogleSearch())] if req.search else None,
        )
        r = await self.client.aio.models.generate_content(
            model=self.spec.model_id, contents=req.prompt, config=config
        )
        cand = (r.candidates or [None])[0]
        citations = []
        meta = getattr(cand, "grounding_metadata", None)
        for chunk in (getattr(meta, "grounding_chunks", None) or []):
            web = getattr(chunk, "web", None)
            if web is not None:
                citations.append({"url": web.uri, "title": web.title})
        return Response(
            text=r.text or "",
            citations=citations,
            model_id_reported=getattr(r, "model_version", None),
            finish_reason=str(cand.finish_reason) if cand and cand.finish_reason else None,
            usage=r.usage_metadata.model_dump() if r.usage_metadata else {},
            raw=r.model_dump(mode="json"),
        )
