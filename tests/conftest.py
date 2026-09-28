from __future__ import annotations

import pytest

from biasstudy.config import Study


@pytest.fixture
def study() -> Study:
    """Small self-contained study so tests don't depend on the evolving real config."""
    return Study.model_validate({
        "models": [
            {"key": "m_a", "provider": "openai", "model_id": "a", "training_cutoff": "2025-06-01"},
            {"key": "m_b", "provider": "anthropic", "model_id": "b", "training_cutoff": "2025-06-01",
             "supports_temperature": False},
            {"key": "m_c", "provider": "together", "model_id": "c", "training_cutoff": "2025-06-01",
             "supports_search": False},
            {"key": "aio", "provider": "serpapi_ai_overview", "model_id": "google-ai-overview",
             "supports_nosearch": False, "supports_temperature": False, "styles": ["search"]},
            {"key": "off", "provider": "openrouter", "model_id": "x", "route": "x", "enabled": False},
        ],
        "judges": [],
        "races": [{
            "id": "xx01_2026", "year": 2026, "district_desc": "Testland's 1st district", "group": "index",
            "location": "Testland, United States",
            "candidates": [
                {"id": "inc", "name": "Alice Incumbent", "party": "R", "incumbent": True,
                 "office_desc": "U.S. Representative", "search_hint": "Testland congresswoman",
                 "aliases": ["Incumbent"]},
                {"id": "chal", "name": "Bob Challenger", "party": "D", "incumbent": False,
                 "office_desc": "candidate", "aliases": ["Challenger"]},
            ],
        }],
        "controversies": [
            {"id": "inc_scandal", "candidate_id": "inc", "description": "Alleged misuse of funds.",
             "date": "2024-01-01", "severity": 2, "status": "verified",
             "sources": ["https://a.example", "https://b.example"], "patterns": [r"misuse of (campaign )?funds"]},
        ],
        "prompts": [
            {"id": "race_1", "tier": "race", "text": "Race {year}: {candidate_a} vs {candidate_b} in {district_desc}."},
            {"id": "profile_1", "tier": "profile", "text": "Profile {candidate}, {office_desc}."},
            {"id": "profile_2", "tier": "profile", "enabled": False, "text": "Unused {candidate}."},
            {"id": "s_race", "tier": "race", "style": "search", "text": "{candidate_a} vs {candidate_b}"},
            {"id": "s_profile", "tier": "profile", "style": "search", "text": "{candidate} {search_hint}"},
        ],
    })
