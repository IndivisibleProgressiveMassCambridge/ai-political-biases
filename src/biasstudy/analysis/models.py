"""Pre-registered models (see PREREGISTRATION.md).

Primary fits are per search arm, because Llama has no search arm (an empty model x arm cell).
Models use sum-to-zero ("deviation") coding, so each model's effect is its difference from the
average of all models. That is symmetric and names no model as the reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests


def deviation_effects(result, var: str, levels: list[str]) -> pd.DataFrame:
    """Recover all levels' deviations from a Sum-coded fit, including the omitted last level."""
    names = list(result.params.index)
    coded = [f"C({var}, Sum)[S.{lvl}]" for lvl in levels[:-1]]
    idx = [names.index(c) for c in coded]
    cov = np.asarray(result.cov_params())
    params = np.asarray(result.params)
    rows = []
    for lvl in levels:
        L = np.zeros(len(names))
        if lvl == levels[-1]:
            L[idx] = -1.0
        else:
            L[names.index(f"C({var}, Sum)[S.{lvl}]")] = 1.0
        est = float(L @ params)
        se = float(np.sqrt(L @ cov @ L))
        z = est / se if se > 0 else np.nan
        rows.append({"level": lvl, "estimate": est, "se": se,
                     "ci_low": est - 1.96 * se, "ci_high": est + 1.96 * se,
                     "p": float(2 * stats.norm.sf(abs(z))) if se > 0 else np.nan})
    return pd.DataFrame(rows)


def _levels(df: pd.DataFrame, col: str) -> list[str]:
    return sorted(df[col].unique())


def _subset(df: pd.DataFrame, arm: str, style: str) -> pd.DataFrame:
    return df[(df["arm"] == arm) & (df["style"] == style)]


def fit_h1(shares: pd.DataFrame, arm: str, style: str = "search") -> pd.DataFrame:
    """Incumbent word share ~ model (+ prominence, + order), random intercept per race."""
    d = _subset(shares, arm, style)
    d = d[d["share"].notna()].copy()
    terms = ["C(model, Sum)", "incumbent_listed_first"]
    if d["log_prominence_ratio"].notna().all():
        terms.append("log_prominence_ratio")
    res = smf.mixedlm("share ~ " + " + ".join(terms), d, groups=d["race_id"]).fit(reml=True)
    return deviation_effects(res, "model", _levels(d, "model")).assign(hypothesis="H1_share", arm=arm, style=style)


def _gee(formula: str, d: pd.DataFrame, family) -> object:
    return smf.gee(formula, groups="candidate_id", data=d, family=family,
                   cov_struct=sm.cov_struct.Exchangeable()).fit()


def fit_h2(items: pd.DataFrame, arm: str, style: str = "search", effective: bool = False) -> pd.DataFrame:
    """Mentioned (0/1) ~ model + tier + severity + party + incumbency; GEE clustered by candidate.
    Estimates are log-odds deviations from the all-model average.
    Primary: among answers actually shown. effective=True also counts "no AI answer" as not mentioned."""
    d = _subset(items, arm, style)
    d = d[d["in_scope"] & ((d["shown"] == 1) | effective)].copy()
    formula = "mentioned ~ C(model, Sum) + C(tier) + severity + C(party) + incumbent"
    res = _gee(formula, d, sm.families.Binomial())
    hyp = "H2_effective" if effective else "H2_omission"
    return deviation_effects(res, "model", _levels(d, "model")).assign(hypothesis=hyp, arm=arm, style=style)


def fit_h3(items: pd.DataFrame, arm: str, style: str = "search") -> pd.DataFrame:
    """Framing score (1-3) among mentioned items ~ model + covariates; GEE clustered by candidate."""
    d = _subset(items, arm, style)
    d = d[d["in_scope"] & (d["shown"] == 1) & (d["mentioned"] == 1) & d["framing"].notna()].copy()
    formula = "framing ~ C(model, Sum) + C(tier) + severity + C(party) + incumbent"
    res = _gee(formula, d, sm.families.Gaussian())
    return deviation_effects(res, "model", _levels(d, "model")).assign(hypothesis="H3_framing", arm=arm, style=style)


def fit_h4(items: pd.DataFrame, arm: str, style: str = "search") -> pd.DataFrame:
    """Party symmetry: does any model's mention rate depend on the candidate's party?
    Joint Wald test of the model x party interaction terms."""
    d = _subset(items, arm, style)
    d = d[d["in_scope"] & (d["shown"] == 1) & d["party"].isin(["D", "R"])].copy()
    formula = "mentioned ~ C(model, Sum) * C(party) + C(tier) + severity + incumbent"
    res = _gee(formula, d, sm.families.Binomial())
    names = list(res.params.index)
    inter = [i for i, n in enumerate(names) if "C(model, Sum)" in n and ":" in n]
    R = np.zeros((len(inter), len(names)))
    for row, i in enumerate(inter):
        R[row, i] = 1.0
    w = res.wald_test(R, scalar=True)
    return pd.DataFrame([{"hypothesis": "H4_party", "arm": arm, "style": style, "level": "model x party",
                          "estimate": float(w.statistic), "df": len(inter), "p": float(w.pvalue)}])


def holm(results: pd.DataFrame) -> pd.DataFrame:
    """Holm correction across all primary contrasts passed in."""
    out = results.copy()
    mask = out["p"].notna()
    out.loc[mask, "p_holm"] = multipletests(out.loc[mask, "p"], method="holm")[1]
    return out


def shown_rates(shown: pd.DataFrame) -> pd.DataFrame:
    """How often each product showed an AI answer at all, by query tier."""
    return (shown.groupby(["arm", "style", "model", "tier"])
                 .agg(n=("shown", "size"), shown_rate=("shown", "mean")).reset_index())


def descriptive_tables(items: pd.DataFrame) -> pd.DataFrame:
    """Plain mention rates (among shown answers) by model x arm x style x tier."""
    d = items[items["in_scope"] & (items["shown"] == 1)]
    return (d.groupby(["arm", "style", "model", "tier"])
              .agg(n=("mentioned", "size"), mention_rate=("mentioned", "mean"),
                   kw_rate=("kw_mentioned", "mean"), disputed_rate=("disputed", "mean"),
                   mean_framing=("framing", "mean"))
              .reset_index())


def chi_square_by_model(items: pd.DataFrame, arm: str) -> dict:
    """Descriptive chi-square of mention counts by model (ignores clustering; not the primary test)."""
    d = items[(items["arm"] == arm) & items["in_scope"]]
    table = pd.crosstab(d["model"], d["mentioned"])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return {"arm": arm, "chi2": chi2, "dof": dof, "p": p, "table": table}
