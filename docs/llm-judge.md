# LLM-as-a-judge for lead replies

`eval/live_run.py` checks the **priority** of each lead against a known label. It can't check the **reply**, because there is no single right wording. `eval/judge.py` asks a judge model to grade every GPT-written `suggested_response`, using the full knowledge base (`knowledge/*.md`) as the source of truth.

## Rubric

| Criterion (1–5) | What it checks |
|---|---|
| **groundedness** | Every specific fact (price, number, discount, policy, timing, feature, SLA) is in the knowledge base. Invented or contradicted facts score 2 or lower |
| **relevance** | The reply addresses what the lead asked |
| **tone** | Professional, friendly and concise |
| **next_step** | The reply moves the lead forward in a way that fits its priority: High gets a concrete step, Medium gets info and an invitation, Low gets a polite, brief answer |

The judge also lists every **unsupported claim** verbatim. A reply **passes** when groundedness, relevance and tone are all ≥ 4 and it has no unsupported claims. This rule is applied in code, not left to the judge. `next_step` is reported but does not decide the verdict.

The judge runs with `temperature=0` and a strict JSON schema, so every grade has the same shape.

## Calibration: can the judge be trusted?

Every run first grades **9 planted replies** whose verdict is known. 5 should fail:
- an invented price;
- a wrong refund promise;
- an invented 99.999% SLA;
- an off-topic answer;
- a rude answer.

4 should pass: a correct pricing answer, a good High-lead reply, a polite spam decline, and an "I'll confirm with the team" answer to a question the knowledge base doesn't cover. The report shows how many it got right. A judge that waves planted defects through is flagged with "read its scores with care".

## Run it

```bash
.venv/bin/python -m eval.live_run --yes          # 1. get fresh replies from the live API (~$0.01)
.venv/bin/python -m eval.judge                   # 2. dry run: what will be judged, estimated cost
.venv/bin/python -m eval.judge --yes             # 3. calibration + judge every GPT reply (~$0.02)
.venv/bin/python -m eval.judge --yes --calibrate-only           # just the planted cases (~$0.003)
.venv/bin/python -m eval.judge --yes --judge-model gpt-4o       # stronger second opinion (~$0.4)
```

- **Input:** the newest `eval/results/live-*.json` that has replies in it. `live_run.py` now saves each reply, rationale and knowledge source. Leads answered by the classifier's fixed reply are skipped.
- **Output:** `eval/results/judge-<time>.md` and `.json` (gitignored). These contain:
  - the calibration table;
  - the pass rate;
  - the average score for each criterion;
  - every failed reply, with the judge's reason and the unsupported claims.
- **Keys:** uses `OPENAI_API_KEY` from `.env`.
- **Cost:** with `gpt-4o-mini` as the judge, about $0.0003 per reply (the knowledge base is ~1.1k tokens). Nothing is sent without `--yes`.

## First run and what it changed (2026-11-25)

The judge caught 5 of 5 planted defects, but failed the good spam decline. Of 70 live GPT replies, 74% passed. Reading the 18 failures against `knowledge/`:

- **About 5 invented facts.** For example "we offer trial accounts", "mid-month seats are prorated", "yes, we have customers in healthcare" and "$149 per seat". None of these is in the knowledge base.
- **About 8 vague but safe replies,** to questions the knowledge base doesn't cover (Zendesk, Teams, accuracy, nonprofit pricing).
- **4 judge errors on Low leads** (`ok`, `asdf`, insults), and **1 arithmetic error** (it called $18k too little for 15 Growth seats).

Two fixes followed:
1. The answer prompt in `main.py` now states a company fact only when the knowledge says it, and otherwise says it will confirm with the team.
2. The judge prompt now treats a polite Low reply, and "I'll confirm", as grounded. It also checks arithmetic against plan limits.

Rerun both steps to compare.

## Limits

- **Self-preference:** a `gpt-4o-mini` judge grading `gpt-4o-mini` replies may be lenient towards its own style. `--judge-model gpt-4o` gives an independent, stricter read.
- **A guide, not a measurement:** the scores are for finding bad replies and tracking changes, not a precise quality number. Read the failed replies; they are the useful part.
- **Not run in CI:** it calls OpenAI, so it stays a manual step, like `eval/run_ragas.py`. CI tests the script with a fake judge (`tests/test_judge.py`).
