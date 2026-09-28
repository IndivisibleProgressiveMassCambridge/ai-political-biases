"""Regex-based mention detection against the codebook. Transparent baseline for the judges."""

from __future__ import annotations

import re
from functools import lru_cache

from biasstudy.config import Controversy, Study


@lru_cache(maxsize=None)
def _compile(patterns: tuple[str, ...]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def item_hits(text: str, item: Controversy) -> list[str]:
    return [m.group(0) for p in _compile(tuple(item.patterns)) if (m := p.search(text))]


def relevant_items(study: Study, race_id: str, candidate_id: str | None) -> list[Controversy]:
    """Race prompts are scored on both candidates' items; profile/controversy on the target only."""
    race = study.race(race_id)
    ids = [c.id for c in race.candidates] if candidate_id is None else [candidate_id]
    return study.items_for(ids)


def score_record(study: Study, record: dict) -> list[dict]:
    rows = []
    for item in relevant_items(study, record["race_id"], record["candidate_id"]):
        hits = item_hits(record["text"], item)
        rows.append({
            "key": record["key"],
            "item_id": item.id,
            "kw_mentioned": int(bool(hits)),
            "kw_matches": hits,
        })
    return rows
