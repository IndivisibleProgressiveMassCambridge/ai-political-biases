"""Synthetic data with known effects: the pre-registered models must recover them."""

import numpy as np
import pandas as pd

from biasstudy.analysis import models as M
from biasstudy.analysis.frames import consensus_mentioned, in_scope

MODELS = ["m_a", "m_b", "m_c", "m_d"]


def synthetic_items(seed=0, n_cand=40, reps=6):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cand):
        cand_effect = rng.normal(0, 0.5)
        party = "D" if c % 2 else "R"
        for m in MODELS:
            for tier in ["race", "profile", "controversy"]:
                for _ in range(reps):
                    logit = 0.5 + cand_effect + (-1.2 if m == "m_b" else 0) + (0.8 if tier == "controversy" else 0)
                    mentioned = int(rng.random() < 1 / (1 + np.exp(-logit)))
                    framing = (np.clip(round(2 + (-0.6 if m == "m_c" else 0) + rng.normal(0, 0.5)), 1, 3)
                               if mentioned else np.nan)
                    rows.append({"arm": "nosearch", "style": "search", "shown": 1, "in_scope": True,
                                 "model": m, "tier": tier,
                                 "candidate_id": f"c{c}", "party": party, "incumbent": c % 3 == 0,
                                 "severity": 1 + c % 3, "mentioned": mentioned, "framing": framing})
    return pd.DataFrame(rows)


def synthetic_shares(seed=0, n_race=30, reps=6):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_race):
        race_effect = rng.normal(0, 0.05)
        for m in MODELS:
            for _ in range(reps):
                share = 0.55 + race_effect + (0.08 if m == "m_d" else 0) + rng.normal(0, 0.04)
                rows.append({"arm": "nosearch", "style": "search", "model": m, "race_id": f"r{r}", "share": share,
                             "incumbent_listed_first": int(rng.random() < 0.5),
                             "log_prominence_ratio": np.nan})
    return pd.DataFrame(rows)


def test_h2_recovers_omission_effect():
    res = M.fit_h2(synthetic_items(), "nosearch").set_index("level")
    assert res.loc["m_b", "estimate"] < -0.5 and res.loc["m_b", "p"] < 0.01
    assert all(res.loc[m, "p"] > 0.001 or res.loc[m, "estimate"] > 0 for m in ["m_a", "m_c", "m_d"])
    assert abs(res["estimate"].sum()) < 1e-8  # deviations sum to zero


def test_h2_primary_excludes_unshown_but_effective_counts_them():
    items = synthetic_items()
    # m_a's product shows no answer for half its responses: primary ignores those rows,
    # effective analysis counts them as not mentioned (so m_a looks worse there).
    hide = (items["model"] == "m_a") & (items.index % 2 == 0)
    items.loc[hide, ["shown", "mentioned"]] = [0, 0]
    primary = M.fit_h2(items, "nosearch").set_index("level")
    effective = M.fit_h2(items, "nosearch", effective=True).set_index("level")
    assert effective.loc["m_a", "estimate"] < primary.loc["m_a", "estimate"] - 0.3


def test_h3_recovers_framing_effect():
    res = M.fit_h3(synthetic_items(), "nosearch").set_index("level")
    assert res.loc["m_c", "estimate"] < -0.2 and res.loc["m_c", "p"] < 0.01


def test_h1_recovers_share_effect():
    res = M.fit_h1(synthetic_shares(), "nosearch").set_index("level")
    assert res.loc["m_d", "estimate"] > 0.04 and res.loc["m_d", "p"] < 0.01


def test_h4_null_when_no_party_effect():
    res = M.fit_h4(synthetic_items(seed=1), "nosearch")
    assert res["p"].iloc[0] > 0.01


def test_holm_adds_adjusted_p():
    df = M.holm(pd.DataFrame({"p": [0.01, 0.02, 0.5]}))
    assert list(df["p_holm"].round(3)) == [0.03, 0.04, 0.5]


def test_consensus_and_scope():
    from datetime import date

    assert consensus_mentioned(1, 1, 0) == (1, False)
    assert consensus_mentioned(1, 0, 0) == (0, True)
    assert in_scope(date(2024, 1, 1), date(2025, 1, 1), "nosearch")
    assert not in_scope(date(2025, 6, 1), date(2025, 1, 1), "nosearch")
    assert not in_scope(None, date(2025, 1, 1), "nosearch")
    assert in_scope(date(2026, 1, 1), date(2025, 1, 1), "search")
