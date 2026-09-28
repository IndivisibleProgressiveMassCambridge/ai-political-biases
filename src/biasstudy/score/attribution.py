"""Attribute words in a race-overview response to the incumbent, the challenger, or neither.

Heuristic, deterministic, and auditable:
  - A heading (markdown '#', bold-only line, or a short line ending in ':') that names exactly one
    candidate sets the section context; a heading naming both or neither clears it.
  - A sentence naming exactly one candidate goes to that candidate.
  - A sentence naming neither inherits the current section context, or the previous sentence's
    candidate if it opens with a pronoun; otherwise it is 'neutral'.
  - A sentence naming both is 'both' (comparisons), excluded from the share denominator.
The share of words landing in 'neutral'/'both' is reported so readers can judge coverage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from biasstudy.config import Race

_HEADING = re.compile(r"^\s*(#{1,6}\s+.*|\*\*[^*]+\*\*:?\s*|[^.!?]{1,80}:)\s*$")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[*])")
_PRONOUN_START = re.compile(r"^\W*(he|she|they|his|her|their|him|them)\b", re.IGNORECASE)
_WORD = re.compile(r"\b[\w'-]+\b")


@dataclass
class Attribution:
    words: dict[str, int] = field(default_factory=dict)  # candidate_id | 'both' | 'neutral'

    def share(self, cand_id: str, other_id: str) -> float | None:
        a, b = self.words.get(cand_id, 0), self.words.get(other_id, 0)
        return a / (a + b) if a + b else None

    @property
    def total(self) -> int:
        return sum(self.words.values())


def _name_patterns(race: Race) -> dict[str, re.Pattern]:
    return {
        c.id: re.compile(r"\b(" + "|".join(re.escape(n) for n in c.names) + r")\b", re.IGNORECASE)
        for c in race.candidates
    }


def _units(text: str) -> list[tuple[bool, str]]:
    """Split into (is_heading, text) units: headings whole, other lines into sentences."""
    units = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if _HEADING.match(line):
            units.append((True, line))
        else:
            units.extend((False, s) for s in _SENT_SPLIT.split(line.strip()) if s.strip())
    return units


def attribute(text: str, race: Race) -> Attribution:
    pats = _name_patterns(race)
    out = Attribution(words={c.id: 0 for c in race.candidates} | {"both": 0, "neutral": 0})
    section: str | None = None
    prev: str | None = None
    for is_heading, unit in _units(text):
        named = [cid for cid, p in pats.items() if p.search(unit)]
        if is_heading:
            section = named[0] if len(named) == 1 else None
            prev = section
            continue  # headings themselves aren't counted as content
        if len(named) == 1:
            label = named[0]
        elif len(named) > 1:
            label = "both"
        elif _PRONOUN_START.match(unit) and prev:
            label = prev
        elif section:
            label = section
        else:
            label = "neutral"
        out.words[label] += len(_WORD.findall(unit))
        prev = label if label in pats else prev
    return out
