"""Human grading sample and agreement statistics (Cohen's kappa)."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from biasstudy.config import Study
from biasstudy.score.keywords import relevant_items
from biasstudy.store import Store

HUMAN_FIELDS = ["mentioned", "framing", "exculpatory", "factual_error"]


def export_sample(study: Study, blinded: Store, key_map: pd.DataFrame, out_csv: Path,
                  frac: float = 0.15, seed: int = 0) -> int:
    """Stratify by model x arm (using the key map internally) but write only blind IDs."""
    records = {r["key"]: r for r in blinded}
    km = key_map[key_map["blind_id"].isin(records)]
    sampled = (km.groupby(["model_key", "arm"], group_keys=False)
                 .apply(lambda g: g.sample(frac=frac, random_state=seed)))
    ids = list(sampled["blind_id"])
    random.Random(seed).shuffle(ids)
    n = 0
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["blind_id", "item_id", "candidate", "item_description", "response_text", *HUMAN_FIELDS])
        for bid in ids:
            rec = records[bid]
            race = study.race(rec["race_id"])
            names = {c.id: c.name for c in race.candidates}
            for item in relevant_items(study, rec["race_id"], rec["candidate_id"]):
                w.writerow([bid, item.id, names[item.candidate_id], item.description.strip(),
                            rec["text"], "", "", "", ""])
                n += 1
    return n


def judge_frame(judged: Store) -> pd.DataFrame:
    rows = []
    for r in judged:
        for g in r["items"]:
            rows.append({"blind_id": r["key"], "judge": r["judge"], **{k: g[k] for k in ["item_id", *HUMAN_FIELDS]}})
    return pd.DataFrame(rows)


def agreement(human_csv: Path, judges: pd.DataFrame) -> pd.DataFrame:
    """Kappa between the human and each judge, and between judges, per field."""
    human = pd.read_csv(human_csv).dropna(subset=["mentioned"])
    human = human.assign(judge="human")[["blind_id", "judge", "item_id", *HUMAN_FIELDS]]
    wide = pd.concat([human, judges]).pivot_table(index=["blind_id", "item_id"], columns="judge",
                                                  values=HUMAN_FIELDS)
    raters = sorted(set(wide.columns.get_level_values(1)))
    out = []
    for field in HUMAN_FIELDS:
        weights = "quadratic" if field == "framing" else None
        for i, a in enumerate(raters):
            for b in raters[i + 1:]:
                pair = wide[field][[a, b]].dropna()
                if len(pair) < 2:
                    continue
                k = cohen_kappa_score(pair[a].astype(int), pair[b].astype(int), weights=weights)
                out.append({"field": field, "rater_a": a, "rater_b": b, "n": len(pair), "kappa": k})
    return pd.DataFrame(out)
