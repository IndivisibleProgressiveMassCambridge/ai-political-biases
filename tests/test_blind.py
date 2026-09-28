import csv
import re

from biasstudy.blind import SELF_ID_PATTERNS, blind, scrub
from biasstudy.store import Store

LEAKY = ("As ChatGPT, I can't browse. I'm Claude, made by Anthropic. Gemini says [1] 【4†source】 "
         "per [the Plain Dealer](https://example.com/x). Grok by xAI, GPT-5, Llama.")


def test_scrub_removes_all_self_identification():
    out = scrub(LEAKY)
    for p in SELF_ID_PATTERNS:
        assert not re.search(p, out, re.IGNORECASE), p
    assert "https://" not in out and "【" not in out and "[1]" not in out
    assert "the Plain Dealer" in out


def test_blind_writes_opaque_ids_and_key_map(tmp_path):
    raw = Store(tmp_path / "raw.jsonl")
    for i, m in enumerate(["m_a", "m_b"]):
        raw.append({"key": f"k{i}", "model_key": m, "arm": "nosearch", "temp_mode": "default",
                    "race_id": "r", "candidate_id": None, "prompt_id": "race_1", "text": LEAKY})
    out = Store(tmp_path / "blinded.jsonl")
    km = tmp_path / "key_map.csv"
    assert blind(raw, out, km, salt="s") == 2
    assert blind(raw, out, km, salt="s") == 0  # idempotent
    recs = list(out)
    assert all(r["key"].startswith("b_") and "model_key" not in r and "arm" not in r for r in recs)
    with km.open() as f:
        assert len(list(csv.DictReader(f))) == 2
