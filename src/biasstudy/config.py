"""Typed loaders for the study configuration in config/*.yaml."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"

Tier = Literal["race", "profile", "controversy"]
Style = Literal["chat", "search"]  # chat = written prompt; search = what people type into Google


class ModelSpec(BaseModel):
    key: str
    provider: Literal["openrouter", "serpapi_ai_overview", "serpapi_ai_mode",
                      "openai", "gemini", "anthropic", "xai", "together"]
    model_id: str
    enabled: bool = True
    route: str | None = None  # OpenRouter: pin to this upstream provider slug, no fallbacks
    training_cutoff: dt.date | None = None
    supports_search: bool = True
    supports_nosearch: bool = True
    supports_temperature: bool = True
    styles: list[Style] = ["chat", "search"]
    max_output_tokens: int = 16000


class JudgeSpec(BaseModel):
    key: str
    base_url: str
    model_id: str
    api_key_env: str
    route: str | None = None


class Candidate(BaseModel):
    id: str
    name: str
    party: Literal["D", "R", "I", "other"]
    incumbent: bool
    office_desc: str
    search_hint: str = ""  # disambiguates a Google query, e.g. "Ohio congressman"
    aliases: list[str] = []
    prominence: float | None = None

    @property
    def names(self) -> list[str]:
        return [self.name, *self.aliases]


class Race(BaseModel):
    id: str
    year: int
    district_desc: str
    location: str = "United States"  # SerpApi search location, e.g. "Ohio, United States"
    group: Literal["index", "control", "scandal"]
    candidates: list[Candidate]

    @model_validator(mode="after")
    def _one_incumbent(self) -> Race:
        if len(self.candidates) != 2:
            raise ValueError(f"race {self.id}: expected exactly 2 candidates")
        if sum(c.incumbent for c in self.candidates) != 1:
            raise ValueError(f"race {self.id}: expected exactly one incumbent")
        return self

    @property
    def incumbent(self) -> Candidate:
        return next(c for c in self.candidates if c.incumbent)

    @property
    def challenger(self) -> Candidate:
        return next(c for c in self.candidates if not c.incumbent)


class Controversy(BaseModel):
    id: str
    candidate_id: str
    description: str
    date: dt.date | None = None
    severity: Literal[1, 2, 3] | None = None
    status: Literal["unverified", "verified"]
    sources: list[str] = []
    patterns: list[str] = []

    @model_validator(mode="after")
    def _verified_needs_sources(self) -> Controversy:
        if self.status == "verified" and (len(self.sources) < 2 or self.date is None
                                          or self.severity is None or not self.patterns):
            raise ValueError(f"controversy {self.id}: verified items need >=2 sources, a date, "
                             "a severity, and keyword patterns")
        return self


class PromptTemplate(BaseModel):
    id: str
    tier: Tier
    style: Style = "chat"
    enabled: bool = True
    text: str


class Study(BaseModel):
    models: list[ModelSpec]
    judges: list[JudgeSpec]
    races: list[Race]
    controversies: list[Controversy]
    prompts: list[PromptTemplate]

    @model_validator(mode="after")
    def _cross_refs(self) -> Study:
        cand_ids = {c.id for r in self.races for c in r.candidates}
        for item in self.controversies:
            if item.candidate_id not in cand_ids:
                raise ValueError(f"controversy {item.id}: unknown candidate {item.candidate_id}")
        return self

    def model(self, key: str) -> ModelSpec:
        return next(m for m in self.models if m.key == key)

    def race(self, race_id: str) -> Race:
        return next(r for r in self.races if r.id == race_id)

    def prompt(self, prompt_id: str) -> PromptTemplate:
        return next(p for p in self.prompts if p.id == prompt_id)

    def items_for(self, candidate_ids: list[str]) -> list[Controversy]:
        return [c for c in self.controversies if c.candidate_id in candidate_ids]

    @property
    def unverified(self) -> list[Controversy]:
        return [c for c in self.controversies if c.status != "verified"]


def _load(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_study(config_dir: Path = CONFIG_DIR) -> Study:
    models = _load(config_dir / "models.yaml")
    return Study(
        models=models.get("models", []),
        judges=models.get("judges", []),
        races=_load(config_dir / "races.yaml").get("races", []),
        controversies=_load(config_dir / "codebook.yaml").get("controversies", []),
        prompts=_load(config_dir / "prompts.yaml").get("prompts", []),
    )
