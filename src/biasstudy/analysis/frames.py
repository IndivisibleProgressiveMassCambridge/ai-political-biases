"""Join raw responses, keyword scores, and (unblinded) judge grades into analysis frames."""

from __future__ import annotations

import math
from datetime import date

import pandas as pd

from biasstudy.config import Study
from biasstudy.score.attribution import attribute
from biasstudy.score.keywords import score_record


def consensus_mentioned(j1: float, j2: float, kw: int) -> tuple[int, bool]:
    """Pre-registered rule: judges agree -> their value; judges split -> keyword breaks the tie.
    Returns (value, was_disputed)."""
    if pd.isna(j1) or pd.isna(j2):
        present = [v for v in (j1, j2) if not pd.isna(v)]
        return (int(present[0]) if present else kw), True
    if j1 == j2:
        return int(j1), False
    return kw, True


def in_scope(item_date: date | None, cutoff: date | None, arm: str) -> bool:
    """Search-off: a model can only be scored on controversies public before its cutoff."""
    if arm == "search":
        return True
    if item_date is None or cutoff is None:
        return False
    return item_date <= cutoff


def item_frame(study: Study, raw: list[dict], judges: pd.DataFrame, key_map: pd.DataFrame) -> pd.DataFrame:
    """One row per (response, codebook item). `judges` has blind_id, judge, item_id, grades."""
    prompts = {p.id: p.tier for p in study.prompts}
    styles = {p.id: p.style for p in study.prompts}
    items = {c.id: c for c in study.controversies}
    cands = {c.id: c for r in study.races for c in r.candidates}
    models = {m.key: m for m in study.models}
    b2k = dict(zip(key_map["blind_id"], key_map["key"]))

    j = judges.assign(key=judges["blind_id"].map(b2k))
    jwide = j.pivot_table(index=["key", "item_id"], columns="judge",
                          values=["mentioned", "framing", "exculpatory", "factual_error"])
    judge_keys = sorted(set(j["judge"]))

    rows = []
    for r in raw:
        for kw in score_record(study, r):
            item = items[kw["item_id"]]
            cand = cands[item.candidate_id]
            idx = (r["key"], item.id)
            g = {f: [jwide.get((f, jk), pd.Series(dtype=float)).get(idx, math.nan) for jk in judge_keys]
                 for f in ["mentioned", "framing", "exculpatory", "factual_error"]}
            m = g["mentioned"] + [math.nan] * (2 - len(g["mentioned"]))
            shown = bool(r.get("shown", True))
            if shown:
                mentioned, disputed = consensus_mentioned(m[0], m[1], kw["kw_mentioned"])
            else:
                mentioned, disputed = 0, False  # counts only in the "effective" (unconditional) rate
            framing_vals = [v for v in g["framing"] if not pd.isna(v)]
            rows.append({
                "key": r["key"], "model": r["model_key"], "arm": r["arm"], "temp_mode": r["temp_mode"],
                "race_id": r["race_id"], "prompt_id": r["prompt_id"], "tier": prompts[r["prompt_id"]],
                "style": styles[r["prompt_id"]], "shown": int(shown),
                "rep": r["rep"], "item_id": item.id, "candidate_id": cand.id, "party": cand.party,
                "incumbent": int(cand.incumbent), "severity": item.severity,
                "in_scope": in_scope(item.date, models[r["model_key"]].training_cutoff, r["arm"]),
                "kw_mentioned": kw["kw_mentioned"], "mentioned": mentioned, "disputed": disputed,
                "framing": sum(framing_vals) / len(framing_vals) if framing_vals else math.nan,
                "exculpatory": max((v for v in g["exculpatory"] if not pd.isna(v)), default=math.nan),
                "factual_error": max((v for v in g["factual_error"] if not pd.isna(v)), default=math.nan),
            })
    return pd.DataFrame(rows)


def shown_frame(study: Study, raw: list[dict]) -> pd.DataFrame:
    """One row per response: did the product show an AI answer at all?"""
    prompts = {p.id: p for p in study.prompts}
    return pd.DataFrame([{
        "key": r["key"], "model": r["model_key"], "arm": r["arm"], "race_id": r["race_id"],
        "candidate_id": r["candidate_id"], "tier": prompts[r["prompt_id"]].tier,
        "style": prompts[r["prompt_id"]].style, "shown": int(bool(r.get("shown", True))),
    } for r in raw])


def share_frame(study: Study, raw: list[dict]) -> pd.DataFrame:
    """One row per race-tier response: incumbent word share and coverage diagnostics."""
    prompts = {p.id: p.tier for p in study.prompts}
    styles = {p.id: p.style for p in study.prompts}
    rows = []
    for r in raw:
        if prompts[r["prompt_id"]] != "race" or not r.get("shown", True):
            continue
        race = study.race(r["race_id"])
        att = attribute(r["text"], race)
        inc, chal = race.incumbent, race.challenger
        prom = (math.log(inc.prominence / chal.prominence)
                if inc.prominence and chal.prominence else math.nan)
        rows.append({
            "key": r["key"], "model": r["model_key"], "arm": r["arm"], "temp_mode": r["temp_mode"],
            "style": styles[r["prompt_id"]], "race_id": race.id, "group": race.group, "incumbent_party": inc.party,
            "share": att.share(inc.id, chal.id), "total_words": att.total,
            "unattributed_frac": (att.words["both"] + att.words["neutral"]) / att.total if att.total else math.nan,
            "incumbent_listed_first": int(r["candidate_order"][0] == inc.id),
            "log_prominence_ratio": prom,
        })
    return pd.DataFrame(rows)
