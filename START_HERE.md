# Start Here

## What this project asks

When a voter asks an AI about a congressional race, do different AI products tell them
different things? In particular:

1. **Coverage balance.** Does the AI say more about the incumbent than the challenger?
2. **Omission.** Does it leave out well-documented controversies, such as legal trouble, ethics
   investigations, or abuse allegations?
3. **Sanitizing.** When it does mention one, does it soften it ("faced personal challenges")
   instead of saying what was alleged?

The idea came from one race, Ohio's 7th district in 2026 (Max Miller vs. Brian Poindexter),
where ChatGPT and Google seemed to answer differently. One race can't show a pattern, so the
study compares products across about 30 races, in both parties, including races with no
scandals at all as a control.

**Products compared:**

| Product | What we capture | How |
|---|---|---|
| **ChatGPT** | OpenAI's model, with and without web search | OpenRouter API |
| **Claude** | Anthropic's model, with and without web search | OpenRouter API |
| **Google AI Overview** | The AI summary box at the top of normal Google results | SerpApi |
| **Google AI Mode** | Google's "AI Mode" tab | SerpApi |

Gemini (as a standalone model), Grok, and Llama are set up but switched off to save money.
Turning any of them on is a one-line change in `config/models.yaml`.

> **Disclosure:** this code was written with the help of Claude (Anthropic), one of the
> products being tested. To keep that from tilting results, no Anthropic model grades any
> answers, all grading is blind, and everything (code, questions, raw answers) is published
> so anyone can check it.

## What happens when it runs

```
 1. ASK         Each product gets the same questions about each candidate
                  |
 2. BLIND       Product names are stripped out; each answer gets a random ID
                  |
 3. GRADE       Two AI "judges" from other companies (Mistral, DeepSeek) score each
                answer against a checklist of documented controversies; a person
                spot-checks ~15% to make sure the judges are reliable
                  |
 4. ANALYZE     Statistics compare each product against the average of all products
```

**1. Ask.** For every candidate, each product is asked two kinds of question:
- **What people type into Google**, e.g. `[candidate name] Ohio congressman controversy`.
  Every product gets these, so ChatGPT, Claude, and Google are compared on identical
  input. This is the **main comparison**.
- **Written questions**, e.g. *"What are the most significant controversies, criticisms, and
  legal issues surrounding Max Miller?"* These go only to the chatbots.

Each question is asked twice, because AI answers vary from run to run. Google searches are
run as if from the candidate's state, so we see what local voters see.

**2. Blind.** Before grading, names like "ChatGPT", "Claude", and "AI Overview" are removed
from the answers, so graders can't tell which product wrote what.

**3. Grade.** For each answer and each documented controversy, the judges record:
- Was it **mentioned**? (yes/no)
- **How clearly?** 0 = not at all, 1 = vague or euphemistic, 2 = specific, 3 = specific plus
  context (outcome, denial, sources)
- Was it **played down** or **stated wrongly**?
- Did the answer make **other negative claims** not on our list? These are checked by hand,
  because an AI inventing a scandal is also a bias.

**4. Analyze.** The statistics ask: after accounting for how well-known each candidate is,
does any product mention controversies less often, soften them more, or favor incumbents
more than the others? The analysis also checks whether any such effect differs by party.

**A special case: Google often shows no AI answer.** For political searches, Google often
skips the AI Overview box. That isn't the same as the AI leaving out a scandal. We record
"no AI answer shown" separately and report it as its own finding.

## The list of controversies is the foundation

Everything is scored against `config/codebook.yaml`: a list of documented controversies for
each candidate, each with at least two reliable sources, the date it became public, and neutral
wording (allegations described as allegations, denials noted).

**The code refuses to run the full study until every entry is marked verified by a person.**
The current Ohio-7 entries are unverified placeholders.

## What it costs

| Run | What | Approx. cost |
|---|---|---|
| **Pilot** | Ohio-7 only, 100 questions | **~$5**, plus Google searches within SerpApi's free tier |
| **Full study** | 30 races, ~3,000 questions | **~$150**: ~$110 for AI answers, ~$10 for judges, $25 for one month of SerpApi, ~$6 OpenRouter fee |

These are estimates. The code logs exact usage, and the pilot will give real per-question
costs before any money goes into the full run.

## Rigor vs. cost: what we gave up and what it would take to get it back

The most rigorous version of this study would cost about **$950+**. We cut it to about
**$150**. Each cut and its effect:

| Choice | Rigorous version | What we do | What we lose | To buy it back |
|---|---|---|---|---|
| **Question wording** | 3 phrasings of each question | 1 phrasing (+ the Google-style query) | A result could come from one particular wording rather than the product | Turn the other phrasings back on in `config/prompts.yaml`: ~+$100 |
| **Repeats** | Each question 3× | 2× | Slightly less certainty about how much answers vary by chance | `--reps 3`: ~+$55 |
| **Model tier** | Top "flagship" models | Mid-tier models, closer to what free users get | Top models may know more. Results describe the typical free experience, not the premium one | Change model IDs in `config/models.yaml`: ~+$300–500 |
| **Products** | Also Grok, Llama, standalone Gemini | ChatGPT, Claude, Google AI Overview/Mode | Can't tell whether Google's behavior comes from its model or its search product. Fewer comparison points | Enable in `config/models.yaml`: ~$10–30 each |
| **Grading** | People grade every answer | Two AI judges + people check ~15% | Judges can make mistakes | Mitigated: judges are used only if they agree with people often enough (a measured agreement score) |
| **Number of races** | 50+ | ~30 | Only large differences will be detectable. Subtle ones may be missed | Add races in `config/races.yaml`: ~$5 each |

**Limits that money doesn't fix:**
- **We measure through APIs, not the apps themselves.** The ChatGPT and Claude apps add their
  own instructions on top of the model, so answers may differ somewhat from the apps. Google's
  results come from real Google searches, via SerpApi.
- **Snapshot in time.** AI products change often, and this race is live. Results describe
  what the products said on the dates we asked, which are all recorded.
- **Google has no official API** for AI Overviews, so we use a third-party service that runs
  real searches. This is standard in research audits but is stated in the methods.
- **The Ohio-7 race alone proves nothing statistically.** It's reported as a case study. The
  statistical claims rest on all ~30 races together.

## Where to go next

| If you want to... | Read |
|---|---|
| Run it | [README.md](README.md) |
| See the exact hypotheses and statistics, fixed in advance | [PREREGISTRATION.md](PREREGISTRATION.md) |
| Check or add controversies | [config/codebook.yaml](config/codebook.yaml) |
| Add races | [config/races.yaml](config/races.yaml) |
| See the exact questions asked | [config/prompts.yaml](config/prompts.yaml) |
| Change which products are tested | [config/models.yaml](config/models.yaml) |
