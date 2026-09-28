from types import SimpleNamespace

import httpx
import pytest

from biasstudy.config import ModelSpec
from biasstudy.providers.base import Request
from biasstudy.providers.openrouter_provider import OpenRouterProvider
from biasstudy.providers.serpapi_provider import AIModeProvider, AIOverviewProvider, render_blocks

REQ = Request(prompt="Alice Incumbent Testland congresswoman controversy", search=True, temperature=None,
              max_output_tokens=1000, location="Ohio, United States")


# --- OpenRouter ---------------------------------------------------------------------------

class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        payload = {
            "model": "openai/gpt-5.6-sol", "provider": "OpenAI",
            "choices": [{"finish_reason": "stop", "message": {
                "content": "Answer text.",
                "annotations": [{"type": "url_citation",
                                 "url_citation": {"url": "https://news.example/a", "title": "A"}}]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        return SimpleNamespace(model_dump=lambda mode=None: payload)


def or_spec(**kw):
    return ModelSpec(key="chatgpt", provider="openrouter", model_id="openai/gpt-5.6-sol",
                     route="openai", supports_temperature=False, **kw)


async def test_openrouter_pins_provider_and_native_search():
    comp = FakeCompletions()
    p = OpenRouterProvider(or_spec(), client=SimpleNamespace(chat=SimpleNamespace(completions=comp)))
    resp = await p.complete(Request(prompt="q", search=True, temperature=0.0, max_output_tokens=100))
    body = comp.kwargs["extra_body"]
    assert body["provider"] == {"order": ["openai"], "allow_fallbacks": False}
    assert body["plugins"] == [{"id": "web", "engine": "native"}]
    assert "temperature" not in comp.kwargs  # model doesn't support it
    assert resp.served_by == "OpenAI"
    assert resp.citations == [{"url": "https://news.example/a", "title": "A"}]


async def test_openrouter_nosearch_has_no_plugin():
    comp = FakeCompletions()
    p = OpenRouterProvider(or_spec(), client=SimpleNamespace(chat=SimpleNamespace(completions=comp)))
    await p.complete(Request(prompt="q", search=False, temperature=None, max_output_tokens=100))
    assert "plugins" not in comp.kwargs["extra_body"]


def test_openrouter_requires_route():
    with pytest.raises(ValueError):
        OpenRouterProvider(ModelSpec(key="x", provider="openrouter", model_id="x"), client=object())


# --- SerpApi ------------------------------------------------------------------------------

BLOCKS = [
    {"type": "paragraph", "snippet": "Alice Incumbent is a U.S. Representative from Testland.", "reference_indexes": [0]},
    {"type": "heading", "snippet": "Controversies"},
    {"type": "list", "list": [
        {"title": "Allegation:", "snippet": "A former aide filed an ethics complaint.",
         "list": [{"snippet": "She denied it."}]},
    ]},
    {"type": "expandable", "title": "More", "text_blocks": [{"type": "paragraph", "snippet": "Nested."}]},
    {"type": "top_stories", "top_stories": [{"title": "ignored"}]},
]


def test_render_blocks():
    assert render_blocks(BLOCKS) == [
        "Alice Incumbent is a U.S. Representative from Testland.",
        "## Controversies",
        "- Allegation: A former aide filed an ethics complaint.",
        "  - She denied it.",
        "## More",
        "Nested.",
    ]


def serp_client(responses: dict[str, dict], seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        seen.append(params)
        return httpx.Response(200, json=responses[params["engine"]])
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


AIO_SPEC = ModelSpec(key="google_aio", provider="serpapi_ai_overview", model_id="google-ai-overview",
                     supports_nosearch=False, styles=["search"])


async def test_aio_follows_page_token_and_hides_api_key(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "secret")
    seen = []
    client = serp_client({
        "google": {"search_parameters": {"q": "x", "api_key": "secret"},
                   "ai_overview": {"page_token": "tok"}},
        "google_ai_overview": {"ai_overview": {"text_blocks": BLOCKS,
                                               "references": [{"link": "https://r", "title": "R"}]}},
    }, seen)
    resp = await AIOverviewProvider(AIO_SPEC, client=client).complete(REQ)
    assert resp.shown and "former aide filed an ethics complaint" in resp.text
    assert seen[0]["location"] == "Ohio, United States" and seen[0]["no_cache"] == "true"
    assert seen[1]["page_token"] == "tok"
    assert resp.citations[0]["url"] == "https://r"
    assert "secret" not in str(resp.raw)


async def test_aio_absent_is_not_shown(monkeypatch):
    seen = []
    client = serp_client({"google": {"organic_results": [{"title": "x"}]}}, seen)
    resp = await AIOverviewProvider(AIO_SPEC, client=client).complete(REQ)
    assert not resp.shown and resp.text == "" and resp.finish_reason == "not_shown"
    assert len(seen) == 1


async def test_ai_mode_prefers_markdown():
    seen = []
    client = serp_client({"google_ai_mode": {"reconstructed_markdown": "# Answer\nText",
                                             "text_blocks": BLOCKS}}, seen)
    spec = AIO_SPEC.model_copy(update={"key": "google_ai_mode", "provider": "serpapi_ai_mode"})
    resp = await AIModeProvider(spec, client=client).complete(REQ)
    assert resp.shown and resp.text == "# Answer\nText"


async def test_serpapi_rejects_nosearch():
    with pytest.raises(ValueError):
        await AIOverviewProvider(AIO_SPEC, client=httpx.AsyncClient()).complete(
            Request(prompt="q", search=False, temperature=None, max_output_tokens=1))
