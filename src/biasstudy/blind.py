"""Blind responses for grading: strip model self-identification, assign opaque IDs, shuffle.

Graders (LLM judges and humans) see only blind_id + text. The key map that links blind_id back
to model/arm is written separately and kept out of the repo until grading is complete.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import random
import re
from pathlib import Path

from biasstudy.store import Store

# Vendor/model names that could reveal which system wrote a response.
SELF_ID_PATTERNS = [
    r"\bChat\s?GPT\b", r"\bOpenAI\b", r"\bGPT-?\d[\w.\-]*", r"\bo\d(-mini|-pro)?\b(?=\s+model)",
    r"\bGemini\b", r"\bBard\b", r"\bGoogle\s+(AI|DeepMind)\b",
    r"\bClaude\b", r"\bAnthropic\b",
    r"\bGrok\b", r"\bxAI\b",
    r"\bLlama\b", r"\bMeta\s+AI\b",
    r"\bAI\s+Overviews?\b", r"\bAI\s+Mode\b",
]
# Product boilerplate that would identify the source (removed entirely, not replaced).
_BOILERPLATE = re.compile(
    r"AI responses may include mistakes\.?(\s*Learn more)?|For legal advice, consult a professional\.?",
    re.IGNORECASE,
)
_SELF_ID = re.compile("|".join(SELF_ID_PATTERNS), re.IGNORECASE)
# Inline citation markers like [1] or 【3†source】 differ by vendor and would leak identity.
_CITATION_MARKERS = re.compile(r"【[^】]*】|\[\d+(?:,\s*\d+)*\]")
# Markdown links: keep the link text, drop the URL (search-arm answers inline links differently).
_MD_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")


def scrub(text: str) -> str:
    text = _BOILERPLATE.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _CITATION_MARKERS.sub("", text)
    text = _SELF_ID.sub("[AI]", text)
    return text


def blind_id(key: str, salt: str) -> str:
    return "b_" + hmac.new(salt.encode(), key.encode(), hashlib.sha256).hexdigest()[:12]


def blind(raw: Store, out: Store, key_map_path: Path, salt: str, seed: int = 0) -> int:
    if not salt:
        raise ValueError("BLIND_SALT must be set")
    records = list(raw)
    random.Random(seed).shuffle(records)
    done = out.keys()
    n = 0
    key_map_path.parent.mkdir(parents=True, exist_ok=True)
    new_map = not key_map_path.exists()
    with key_map_path.open("a", newline="") as f:
        w = csv.writer(f)
        if new_map:
            w.writerow(["blind_id", "key", "model_key", "arm", "temp_mode"])
        for r in records:
            bid = blind_id(r["key"], salt)
            if bid in done or not r.get("shown", True):
                continue  # no AI answer shown: nothing to grade (tracked as shown=0 in analysis)
            # Only what graders need: which race/candidate/prompt tier, and the scrubbed text.
            out.append({
                "key": bid,
                "race_id": r["race_id"],
                "candidate_id": r["candidate_id"],
                "prompt_id": r["prompt_id"],
                "text": scrub(r["text"]),
            })
            w.writerow([bid, r["key"], r["model_key"], r["arm"], r["temp_mode"]])
            n += 1
    return n
