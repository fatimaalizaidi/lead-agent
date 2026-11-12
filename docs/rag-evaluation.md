# RAG evaluation (Ragas)

`eval/run_ragas.py` measures how well the RAG pipeline answers questions, using [Ragas](https://docs.ragas.io/) 0.2 with an OpenAI model as the judge.

## What it measures

| Metric | Question it answers | Grades |
|---|---|---|
| **Faithfulness** | Is every claim in the answer supported by the retrieved context? (catches hallucination) | answer vs. context |
| **Answer Relevancy** | Does the answer actually address the question? Hedged "I don't know / contact sales" answers score 0 | answer vs. question |
| **Context Recall** | Did retrieval find everything needed to give the reference answer? | context vs. reference |
| **Context Precision** | Are the relevant chunks ranked above the irrelevant ones? | context vs. reference |

All scores are 0–1, higher is better.

## How it works

1. `eval/golden.jsonl` holds 26 questions written like real lead messages, each with a reference answer taken from `knowledge/` and the source article.
2. Each question runs through the real Knowledge Agent (`multiagent.knowledge_node`): the same `retrieve_context()`, prompt and `OPENAI_MODEL` the API uses. Every question gets a fresh `thread_id`, so conversation memory never leaks between questions.
3. Ragas grades each (question, retrieved contexts, answer, reference) row.
4. Results are written to `eval/results/` (git-ignored): one CSV per mode with per-question scores, plus `<timestamp>-summary.json`.

## Running it

```bash
pip install -r requirements-dev.txt        # adds ragas (not in the Docker image)
python -m eval.run_ragas                   # current RAG_* settings from .env
python -m eval.run_ragas --compare         # vector vs hybrid vs hybrid + re-rank
python -m eval.run_ragas --mode vector     # one mode: configured | vector | hybrid | hybrid-rerank
python -m eval.run_ragas --limit 5         # first 5 questions only (quick check)
python -m eval.run_ragas --judge-model gpt-4o
python -m eval.run_ragas --fail-under 0.8  # exit 1 if any metric mean is below 0.8
python -m eval.run_ragas --gate --per-source 2   # the CI gate: per-metric floors, 10 questions
```

It needs a real `OPENAI_API_KEY`. A full `--compare` run makes a few hundred `gpt-4o-mini` calls plus embeddings, takes about 5 minutes, and costs cents. You may see `RuntimeError: Event loop is closed` messages while it runs; they come from OpenAI clients being cleaned up between runs and don't affect the scores.

## Retrieval gate in CI (free)

`eval/retrieval_gate.py` is a second, LLM-free evaluation that runs in CI (`.github/workflows/eval.yml`). It runs the real `retrieve_context()` on all 26 golden questions and scores:

| Metric | What it checks |
|---|---|
| **top1_accuracy** | The question's source article is ranked first |
| **hit_rate** | The source article is retrieved at all (top `RAG_K`) |
| **mrr** | Mean reciprocal rank of the source article |
| **fact_recall** | The reference answer's key facts (prices, percentages, hours, day counts) appear in the retrieved text, a cheap stand-in for context recall |

It fails if any metric is under its floor in `eval/retrieval_thresholds.json`. The only difference from production is the embedding model: vector search uses Chroma's bundled all-MiniLM-L6-v2 (local, CPU) in a temporary index, so it needs no API key and never touches `data/chroma`.

```bash
python -m eval.retrieval_gate              # production mode (hybrid, no re-ranker); exit 1 if under a floor
python -m eval.retrieval_gate --compare    # vector vs hybrid vs hybrid + re-rank
python -m eval.retrieval_gate --no-gate    # scores only
```

**Baseline (2026-11-12, 26 questions, local embeddings):**

| Mode | Top-1 accuracy | Hit rate | MRR | Fact recall |
|---|---|---|---|---|
| vector | 0.846 | 1.000 | 0.917 | 1.000 |
| hybrid (production) | 0.885 | 1.000 | 0.942 | 1.000 |
| hybrid + re-rank | 0.962 | 1.000 | 0.981 | 1.000 |

**Does it catch real bugs?** Re-introducing the old "one chunk per article" bug drops fact recall to 0.964 (the Starter price `$49` is no longer retrieved), and the gate fails. Switching production to vector-only search fails on MRR (0.917 against a 0.92 floor).

Retrieval is deterministic, so each floor sits just under the baseline with about one question of slack. Raise a floor in the same PR as a change that improves it.

The Ragas run above is still the full evaluation: faithfulness and answer relevancy need an LLM judge, so it stays a manual step (`--gate` uses the floors in `eval/thresholds.json`).

`pytest` covers the harness offline (`tests/test_ragas_eval.py`) and never calls OpenAI or Ragas.

## Adding questions

Add one JSON object per line to `eval/golden.jsonl`:

```json
{"id": "unique-id", "question": "How a lead would ask it", "reference": "The correct answer, from knowledge/", "source": "file.md"}
```

`source` must name a file in `knowledge/` (the tests check it). When you add or change a knowledge article, add questions for it.

## Results (2026-11-09, 26 questions, judge gpt-4o-mini)

| Mode | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|---|---|---|---|---|
| vector | 0.739 | 0.620 | 0.891 | 0.917 |
| hybrid | 0.772 | 0.694 | 0.910 | 0.962 |
| hybrid + re-rank | 0.737 | 0.653 | **0.929** | **0.987** |

Before the fix below, the hybrid + re-rank scores were 0.585 / 0.616 / 0.795 / 0.776.

What the evaluation found:

- **Fixed: retrieval kept only one chunk per article.** `retrieve_context()` used to keep only the longest chunk of each article, so the second half of a split article never reached the LLM. "When do volume discounts kick in?" and "What does Enterprise include?" were answered "I don't have that information", although `pricing.md` covers both. Chunks of the same article are now joined (`rag._group_by_article`). Every question now gets a real answer, and context recall rose from 0.80 to 0.93.
- **Hybrid search and re-ranking help retrieval:** recall goes 0.89 → 0.91 → 0.93 and precision 0.92 → 0.96 → 0.99 from vector to hybrid to hybrid + re-rank.
- **The "contact sales" ending lowers answer relevancy.** The Knowledge Agent prompt asks it to suggest sales or escalation, so 22 of 26 answers end with "reach out to our sales team". Ragas counts this as non-committal: 8 correct answers scored 0 for answer relevancy, and the line also lowers faithfulness. This comes from the prompt, not from retrieval.

Scores vary by a few points between runs, because the judge is an LLM. Compare modes from the same run, and look at the per-question CSV before drawing conclusions from small differences.
