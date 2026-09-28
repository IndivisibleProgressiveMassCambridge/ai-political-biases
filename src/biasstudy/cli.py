"""Command-line entry point: `uv run biasstudy <command>`."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

import pandas as pd
from dotenv import load_dotenv

from biasstudy.config import DATA_DIR, ROOT, load_study
from biasstudy.store import Store

RAW = DATA_DIR / "raw" / "responses.jsonl"
RAW_ERR = DATA_DIR / "raw" / "errors.jsonl"
BLINDED = DATA_DIR / "blinded" / "responses.jsonl"
KEY_MAP = DATA_DIR / "blinded" / "key_map.csv"
SCORED = DATA_DIR / "scored"
HUMAN = DATA_DIR / "human"
REPORTS = ROOT / "reports"


def cmd_validate(args) -> int:
    study = load_study()
    enabled = [m.key for m in study.models if m.enabled]
    print(f"enabled products: {', '.join(enabled)}")
    print(f"{len(study.models)} models, {len(study.races)} races, "
          f"{len(study.controversies)} codebook items, {len(study.prompts)} prompts")
    groups = pd.Series([r.group for r in study.races]).value_counts().to_dict()
    print(f"race groups: {groups}")
    if study.unverified:
        print(f"UNVERIFIED codebook items ({len(study.unverified)}): "
              + ", ".join(c.id for c in study.unverified))
    # Cutoffs only matter for the search-off arm.
    missing = [m.key for m in study.models
               if m.enabled and m.supports_nosearch and m.training_cutoff is None]
    if missing:
        print(f"models missing training_cutoff: {', '.join(missing)}")
    return 0


def cmd_list_models(args) -> int:
    """Print OpenRouter model IDs and prices for the vendors in this study (public endpoint)."""
    import httpx

    prefixes = tuple(args.prefix or ["openai/", "anthropic/", "google/gemini", "x-ai/", "meta-llama/",
                                     "mistralai/mistral-large", "deepseek/"])
    data = httpx.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]
    for m in sorted(data, key=lambda m: m["id"]):
        if not m["id"].startswith(prefixes) or m["id"].endswith(":batch"):
            continue
        p = m.get("pricing", {})
        print(f"{m['id']:<45} in ${float(p.get('prompt', 0)) * 1e6:>6.2f}/M  "
              f"out ${float(p.get('completion', 0)) * 1e6:>6.2f}/M  web ${p.get('web_search') or '-'}")
    return 0


def cmd_collect(args) -> int:
    from biasstudy.collect import build_jobs, run_jobs
    from biasstudy.providers import make_provider

    study = load_study()
    if study.unverified and not args.allow_unverified:
        print(f"Refusing: {len(study.unverified)} codebook items are unverified. Verify and freeze the "
              "codebook first, or pass --allow-unverified for a pilot run.", file=sys.stderr)
        return 2
    jobs = build_jobs(study, models=args.models, races=args.races, reps=args.reps,
                      arms=tuple(args.arms), temp_modes=tuple(args.temp_modes))
    done = Store(RAW).keys()
    todo = [j for j in jobs if j.key not in done]
    by_model = pd.Series([j.model_key for j in todo]).value_counts().to_dict() if todo else {}
    print(f"{len(jobs)} jobs in matrix, {len(todo)} to run: {by_model}")
    if args.dry_run:
        for j in todo[:5]:
            print(f"  e.g. [{j.model_key}/{j.arm}] {j.prompt_text}")
        return 0
    providers = {k: make_provider(study.model(k)) for k in {j.model_key for j in todo}}
    locations = {r.id: r.location for r in study.races}
    stats = asyncio.run(run_jobs(jobs, providers, Store(RAW), Store(RAW_ERR),
                                 concurrency=args.concurrency, study_locations=locations))
    print(stats)
    return 0 if stats["error"] == 0 else 1


def cmd_blind(args) -> int:
    from biasstudy.blind import blind

    n = blind(Store(RAW), Store(BLINDED), KEY_MAP, salt=os.environ.get("BLIND_SALT", ""))
    print(f"blinded {n} new responses -> {BLINDED}")
    return 0


def cmd_judge(args) -> int:
    from biasstudy.score.judge import Judge, run_judge

    study = load_study()
    code = 0
    for spec in study.judges:
        if args.judges and spec.key not in args.judges:
            continue
        stats = asyncio.run(run_judge(study, Judge(spec), Store(BLINDED),
                                      Store(SCORED / f"judge_{spec.key}.jsonl"),
                                      Store(SCORED / f"judge_{spec.key}_errors.jsonl"),
                                      concurrency=args.concurrency))
        print(f"{spec.key}: {stats}")
        code |= int(stats["error"] > 0)
    return code


def _judges_df(study):
    from biasstudy.score.human import judge_frame

    frames = [judge_frame(Store(SCORED / f"judge_{j.key}.jsonl")) for j in study.judges]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames) if frames else pd.DataFrame(
        columns=["blind_id", "judge", "item_id", "mentioned", "framing", "exculpatory", "factual_error"])


def cmd_human_export(args) -> int:
    from biasstudy.score.human import export_sample

    study = load_study()
    out = HUMAN / "sample_to_grade.csv"
    n = export_sample(study, Store(BLINDED), pd.read_csv(KEY_MAP), out, frac=args.frac)
    print(f"wrote {n} rows -> {out}. Fill in the grade columns, save as {HUMAN / 'graded.csv'}.")
    return 0


def cmd_agreement(args) -> int:
    from biasstudy.score.human import agreement

    study = load_study()
    df = agreement(HUMAN / "graded.csv", _judges_df(study))
    print(df.to_string(index=False))
    REPORTS.mkdir(exist_ok=True)
    df.to_csv(REPORTS / "agreement.csv", index=False)
    return 0


def cmd_analyze(args) -> int:
    from biasstudy.analysis import models as M
    from biasstudy.analysis.frames import item_frame, share_frame, shown_frame

    study = load_study()
    raw = [r for r in Store(RAW) if r["temp_mode"] == args.temp_mode]
    items = item_frame(study, raw, _judges_df(study), pd.read_csv(KEY_MAP))
    shares = share_frame(study, raw)
    REPORTS.mkdir(exist_ok=True)
    items.to_csv(REPORTS / "item_frame.csv", index=False)
    shares.to_csv(REPORTS / "share_frame.csv", index=False)
    M.descriptive_tables(items).to_csv(REPORTS / "descriptive.csv", index=False)
    shown = M.shown_rates(shown_frame(study, raw))
    shown.to_csv(REPORTS / "shown_rates.csv", index=False)
    print("AI answer shown rate:\n" + shown.to_string(index=False))

    results = []
    combos = items[["arm", "style"]].drop_duplicates().itertuples(index=False)
    for arm, style in sorted(combos):
        for name, fit in (("H1", lambda: M.fit_h1(shares, arm, style)),
                          ("H2", lambda: M.fit_h2(items, arm, style)),
                          ("H2_effective", lambda: M.fit_h2(items, arm, style, effective=True)),
                          ("H3", lambda: M.fit_h3(items, arm, style)),
                          ("H4", lambda: M.fit_h4(items, arm, style))):
            try:
                results.append(fit())
            except Exception as exc:  # e.g. too little data in a pilot
                print(f"[{arm}/{style}] {name} failed: {exc!r}")
    if results:
        primary = pd.concat(results)
        is_primary = primary["hypothesis"].isin(["H1_share", "H2_omission", "H3_framing"])
        out = pd.concat([M.holm(primary[is_primary]), primary[~is_primary]])
        out.to_csv(REPORTS / "results.csv", index=False)
        print(out.to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="biasstudy")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("validate", help="load and sanity-check config").set_defaults(fn=cmd_validate)
    lm = sub.add_parser("list-models", help="list OpenRouter model IDs + prices for pinning")
    lm.add_argument("--prefix", nargs="*", help="e.g. openai/ anthropic/")
    lm.set_defaults(fn=cmd_list_models)

    c = sub.add_parser("collect", help="query models under test")
    c.add_argument("--models", nargs="*")
    c.add_argument("--races", nargs="*")
    c.add_argument("--reps", type=int, default=2)
    c.add_argument("--arms", nargs="*", default=["nosearch", "search"])
    c.add_argument("--temp-modes", nargs="*", default=["default"])
    c.add_argument("--concurrency", type=int, default=4)
    c.add_argument("--allow-unverified", action="store_true", help="pilot only")
    c.add_argument("--dry-run", action="store_true")
    c.set_defaults(fn=cmd_collect)

    sub.add_parser("blind", help="scrub and blind responses for grading").set_defaults(fn=cmd_blind)

    j = sub.add_parser("judge", help="run blinded LLM judges")
    j.add_argument("--judges", nargs="*")
    j.add_argument("--concurrency", type=int, default=8)
    j.set_defaults(fn=cmd_judge)

    h = sub.add_parser("human-export", help="export stratified sample for human grading")
    h.add_argument("--frac", type=float, default=0.15)
    h.set_defaults(fn=cmd_human_export)
    sub.add_parser("agreement", help="kappa: human vs judges").set_defaults(fn=cmd_agreement)

    a = sub.add_parser("analyze", help="fit pre-registered models")
    a.add_argument("--temp-mode", default="default", choices=["default", "zero"])
    a.set_defaults(fn=cmd_analyze)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
