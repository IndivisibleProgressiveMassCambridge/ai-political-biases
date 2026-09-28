# AI Political Coverage Audit

**New here? Read [START_HERE.md](START_HERE.md) first.** It explains what this does, what it
costs, and the tradeoffs in plain language.

Compares how ChatGPT, Claude, and Google's AI Overview / AI Mode cover congressional candidates
and their documented controversies. Design and hypotheses: [PREREGISTRATION.md](PREREGISTRATION.md).

## Setup
You need two accounts:
- **OpenRouter** (openrouter.ai → Keys): one key for ChatGPT, Claude, and both judge models.
  Add ~$10 of credit for a pilot.
- **SerpApi** (serpapi.com → Dashboard): for Google AI Overview and AI Mode. The free tier
  (250 searches/month) covers the pilot.

```bash
uv sync
cp .env.example .env        # paste both keys, and a random salt from: openssl rand -hex 16
uv run biasstudy validate   # checks config, lists anything still unverified
uv run biasstudy list-models --prefix openai/ anthropic/   # current model IDs + prices
```

## Pipeline
```bash
uv run biasstudy collect --dry-run                       # show what would be asked, and how many calls
uv run biasstudy collect --races oh07_2026 --allow-unverified   # pilot (codebook not yet verified)
uv run biasstudy collect                                 # full run; requires a verified codebook
uv run biasstudy blind                                   # strip product names, assign random IDs
uv run biasstudy judge                                   # two blinded LLM judges
uv run biasstudy human-export                            # -> data/human/sample_to_grade.csv
uv run biasstudy agreement                               # judge vs. human agreement (kappa)
uv run biasstudy analyze                                 # -> reports/*.csv
```

Useful flags: `--models chatgpt google_aio` (subset, or run a disabled product),
`--reps 3`, `--arms search`, `--temp-modes default zero`.

Collection and judging can be resumed: rerunning skips anything already stored.
`data/raw/` is append-only. `data/blinded/key_map.csv` (which answer came from which product)
is gitignored until grading is complete.

## Layout
```
config/            what to test: products, races, controversies (codebook), questions
src/biasstudy/
  providers/       OpenRouter (chat models) and SerpApi (Google) connectors
  collect.py       builds the question matrix and runs it, resumably
  blind.py         removes product names before grading
  score/           keyword matching, word-share attribution, LLM judges, human agreement
  analysis/        data frames and the pre-registered statistical models
tests/             uv run pytest
```
