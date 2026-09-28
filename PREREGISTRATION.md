# Pre-registration (DRAFT — not yet frozen)

Freeze this file, `config/codebook.yaml`, `config/races.yaml`, and `config/prompts.yaml` together
with a git tag (e.g. `prereg-v1`) **before** any non-pilot data collection. Record any later
deviation in the Deviations section at the bottom, with its date and reason.

## Research question
Do widely used AI assistants differ in how they describe U.S. congressional candidates,
specifically in (a) how much of a race overview they give the incumbent vs. the challenger,
(b) whether they mention documented controversies, and (c) how they frame those controversies?

This question was prompted by one race (OH-7, 2026). The hypotheses are two-sided and apply
equally to every model; no model is presumed to be the outlier.

## Systems under test
- **ChatGPT** (OpenAI) and **Claude** (Anthropic): accessed through OpenRouter, pinned to the
  first-party provider with fallbacks disabled. The provider that served each response is logged.
- **Google AI Overview** and **Google AI Mode**: captured from live Google searches via SerpApi,
  since neither has an official API. Searches originate from the race's state.
- Optional, off by default: standalone Gemini, Grok, Llama.

Exact model IDs and published training cutoffs are listed in `config/models.yaml` at the frozen
tag. Mid-tier models are used rather than flagships, to approximate the free-tier experience.

**Conflict of interest:** the analysis code was written with the help of Claude (Anthropic),
which is one of the systems under test. To limit this, no Anthropic model is used as a judge,
all grading is blinded, and the code, prompts and raw data are published for audit.

## Design
- **Races:** ~30 (1 index, ~9 control, ~20 scandal). Scandal races are balanced by party and by
  whether the controversy attaches to the incumbent or the challenger.
- **Prompts:** 3 tiers (race, candidate profile, controversy) in two styles:
  - *search style* (primary): short Google-style queries, e.g. `{candidate} {search_hint} controversy`,
    sent to **every** product so all are compared on identical input;
  - *chat style* (secondary): one neutral written question per tier, chatbots only.
  Race prompts randomize candidate order. No system prompt, fresh context each call.
- **Arms:** search-on (every product; chatbots use their vendor's native web search) and
  search-off (chatbots only, testing what the model learned in training).
- **Sampling:** vendor-default temperature, 2 repetitions per cell.

## Ground truth (codebook)
Each controversy has ≥2 reliable sources, the date it became public, and a severity tier (1–3).
It is worded as an allegation where it is one, with denials and outcomes noted. In the search-off
arm, a model is scored only on items made public before its training cutoff.

## Outcomes
- **Answer shown** (0/1): whether the product displayed an AI answer at all. Google often shows
  no AI Overview for political queries. This is reported per product and is **not** counted
  as omission in the primary analyses.
- **Primary 1: incumbent word share** in race-overview responses:
  incumbent words / (incumbent + challenger words), from a deterministic sentence-attribution rule.
- **Primary 2: mentioned** (0/1) per response × relevant codebook item. Consensus rule: if the
  two judges agree, use their answer; if they split, the keyword match decides (flagged as disputed).
- **Primary 3: framing** (1–3) among mentioned items, averaged across the two judges.
- **Secondary:** exculpatory framing, factual error, refusal, unlisted negative claims (checked
  by hand for fabrication), response length.

## Grading
Two LLM judges from vendors not under test (Mistral, DeepSeek; via OpenRouter, pinned). Judges see scrubbed text only: model names,
inline links, and citation markers are removed. A human grades a stratified ~15% sample, also
blinded. Judge results are used only if the judge–human κ is ≥ 0.7 on `mentioned` and ≥ 0.6
(quadratic-weighted) on `framing`. Otherwise the rubric is revised and the revision is recorded
as a deviation.

## Analysis (fit separately for each arm × style; the primary comparison is search arm × search style)
- **H1:** linear mixed model `share ~ model + incumbent_listed_first + log_prominence_ratio`,
  random intercept per race.
- **H2:** GEE logistic `mentioned ~ model + tier + severity + party + incumbent`, exchangeable
  correlation, clustered by candidate.
  Fit on answers that were shown. Secondary "effective" version: no answer shown counts as
  not mentioned, which measures what a voter actually sees.
- **H3:** GEE Gaussian, same terms, on mentioned items only, with `framing` as the outcome.
- **H4 (secondary):** H2 plus a model × party interaction, tested with a joint Wald test.
- Models use sum-to-zero coding, so each estimate is the model's deviation from the all-model
  average. Holm correction across all primary model contrasts. Report estimates with 95% CIs.
- **Index case (OH-7):** reported descriptively only. One race cannot support inferential claims.
- **Descriptive:** raw mention rates by model × arm × tier; χ² of mentions by model (ignores
  clustering; not a primary test).

## Prominence covariate
[TO DECIDE before freeze: one method for every candidate, e.g. English Wikipedia article
length in bytes on a fixed date, or count of news articles over a fixed window.]

## Exclusions
Responses are kept even if they are refusals or errors-as-text. A refusal counts as not mentioned
and is also reported separately. API calls that fail after retries are rerun; any calls still
missing are listed.

## Deviations
_None yet._
