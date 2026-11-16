# Fine-tuning: a local priority classifier (LoRA)

`POST /analyze-lead` asks gpt-4o-mini for three things at once: a priority, a drafted reply and a rationale. The priority alone is a plain 3-way text classification, and a small fine-tuned model can do it without OpenAI. `POST /classify-priority` serves that model.

| | `/analyze-lead` (gpt-4o-mini + RAG) | `/classify-priority` (fine-tuned MiniLM) |
|---|---|---|
| Returns | priority, drafted reply, rationale, sources | priority, confidence, per-label scores |
| Latency | 2–5 s | **~2 ms** model time on CPU (p95 3.5 ms) |
| Cost per lead | OpenAI tokens | **$0** |
| Runs where | OpenAI | Inside the API container (onnxruntime) |

**Use in n8n:** call `/classify-priority` first, archive or park confident `Low` leads (spam, job seekers, wrong numbers), and send only `High`/`Medium` leads to `/analyze-lead` for a drafted reply. LLM spend then scales with real leads, not with spam.

## Results

These scores are on **84 hand-written leads** (`finetune/data/test.jsonl`). They were never generated, never trained on and never used to tune anything. All runs use the same data, 5 epochs, CPU/Apple GPU.

| Model | Trainable params | Accuracy | Macro-F1 | High recall | Low recall | Train time |
|---|---|---|---|---|---|---|
| **LoRA r=8 (shipped, seed 42)** | 74,883 (0.33%) | **0.857** | **0.855** | **1.00** | 0.93 | 17 s |
| LoRA r=8, seeds 1–3 | 74,883 | 0.83 / 0.88 / 0.94 | 0.83 / 0.88 / 0.94 | 0.96–1.00 | 0.62–0.93 | ~17 s |
| Head only (frozen encoder, "linear probe") | 1,155 (0.01%) | 0.71–0.73 | 0.72–0.73 | 0.72–0.76 | 0.59–0.62 | ~10 s |

The full report, with confusion matrices and every misclassified lead, is in [finetune/results/report.md](../finetune/results/report.md).

What the numbers show:

- **Fine-tuning the encoder matters.** Training only a head on frozen sentence embeddings tops out around 72%. LoRA adapters on the attention layers add ~15 points while training a third of a percent of the weights.
- **No hot lead was missed.** High recall is 1.00 for the shipped model (0.96–1.00 across seeds), which is the metric that costs money when it slips. It has its own gate floor.
- **The weak spot is Medium.** Exploratory questions that mention pricing words ("seat", "billing", "refund") get pulled toward High, and short curious messages get pulled toward Low. That's a training-data limit: the templates tie pricing vocabulary to buying intent. Teacher labels on real traffic fix that better than more templates (see below).
- **Seed variance is real** (0.83–0.94 on 84 examples, about ±4 leads). The shipped seed was fixed before looking at test scores. Picking the best-scoring seed would have meant tuning on the test set.

## How it works

```
generate_dataset.py ─▶ train.jsonl (1,200) + val.jsonl (180)   seeded templates, rubric labels
export_langsmith.py ─▶ private/train_mix.jsonl + production_test.jsonl   gpt-4o-mini's labels on real traffic
        │                         [optional] label_with_teacher.py ─▶ gpt-4o-mini labels on synthetic leads
        ▼
train_lora.py   all-MiniLM-L6-v2 + LoRA(r=8, q/v) + 3-way head ─▶ merge ─▶ ONNX ─▶ int8 (23 MB)
        ▼
evaluate.py     hand-written test.jsonl, through classifier.py (the serving code) ─▶ gate
        ▼
finetune/model/ ─▶ Docker image ─▶ POST /classify-priority
```

1. **Data** (`finetune/generate_dataset.py`): seeded templates with random slots (company, seats, budget, timeline, feature, competitor) and light noise (casing, punctuation, typos). Labels follow the same rubric as the production prompt (`main.SYSTEM_PROMPT`): High = clear buying intent, Medium = interested but exploratory, Low = vague, spam or poor fit. Company names, seat counts and features also appear in Low templates (agencies, list sellers), so the model can't use them as shortcuts. The script refuses to generate any lead that is in the test set.
2. **Training** (`finetune/train_lora.py`): `sentence-transformers/all-MiniLM-L6-v2` (22M parameters) with a new classification head. PEFT LoRA puts rank-8 adapters on the attention `query`/`value` projections and trains them plus the head. The best epoch on validation is kept.
3. **Export:** the adapters are merged into the base weights (`merge_and_unload`), so serving has no LoRA code. Then the model is exported to ONNX and dynamically quantised to int8: 90 MB → 23 MB.
4. **Serving** (`classifier.py`): onnxruntime + Hugging Face `tokenizers`, both already installed for Chroma. No torch in the image. The model loads lazily on the first request, single-threaded so it doesn't compete with uvicorn.
5. **Gate:** `tests/test_classifier.py` runs `finetune/evaluate.py` on the committed model in CI. A retrained model under the floors in [finetune/thresholds.json](../finetune/thresholds.json) (accuracy 0.80, macro-F1 0.80, High recall 0.95) fails the build, so Render never deploys it.

### Distillation from production logs (LangSmith)

The shipped model learns from rubric labels on synthetic leads. The real goal is to learn from **what gpt-4o-mini actually decided on real traffic**. Every `/analyze-lead` call (synchronous or background job) already leaves an LLM run in LangSmith, with the prompt and the model's JSON answer. `finetune/export_langsmith.py` turns those runs into a training set:

```bash
python -m finetune.export_langsmith --days 30          # needs LANGSMITH_API_KEY in .env
.venv-train/bin/python -m finetune.train_lora --train-file finetune/data/private/train_mix.jsonl --out models/priority-prod
python -m finetune.evaluate --model models/priority-prod --model finetune/model \
    --test-file finetune/data/private/production_test.jsonl --no-gate     # new vs shipped, on real traffic
```

| Step | What it does |
|---|---|
| Find the runs | LLM runs from the last `--days`, kept only when the user message has the `/analyze-lead` shape (`email: …\nmessage: …\n\nCOMPANY KNOWLEDGE`). `/multi-agent` and `/agent-action` calls are skipped. |
| Label | The model's own `priority` from its JSON answer: the **teacher label**. The app's later overrides (forcing High on low RAG confidence) aren't in it, so the student learns the model's judgement of the text. |
| Clean | Skips failed runs, unparseable answers and follow-up turns (their priority depends on earlier messages the student never sees; `--include-followups` keeps them). Drops duplicates and any lead in the hand-written test set. |
| Privacy | Drops the sender's email. Masks emails, phone numbers and URLs in the text as `[EMAIL]`, `[PHONE]` and `[URL]`. Writes only to `finetune/data/private/`, which is gitignored: **real customer text never goes into the public repo or the Docker image.** |
| Split by time | The newest 20% (`--test-share`) becomes `production_test.jsonl` and is never trained on, so the score reflects future traffic, not memorised past traffic. The rest is added to the synthetic set in `train_mix.jsonl`. |

It prints how many runs it read, why each skipped run was skipped, and the label mix. Tracing must be on in production with inputs visible: with `LANGSMITH_HIDE_INPUTS=true` the traces have no lead text, and the script says so. Under ~100 production leads, keep mixing with the synthetic set and re-export as traffic grows. Promote a new model by copying it to `finetune/model` only when it beats the shipped one on `production_test.jsonl` **and** passes the hand-written gate (`python -m finetune.evaluate`).

### Teacher labels for the synthetic set (optional)

Without production traffic, the production LLM can still label the synthetic leads:

```bash
python -m finetune.label_with_teacher finetune/data/train.jsonl          # dry run: prints the cost
python -m finetune.label_with_teacher finetune/data/train.jsonl --yes    # ~ $0.04 with gpt-4o-mini
.venv-train/bin/python -m finetune.train_lora --train-file finetune/data/train.teacher.jsonl
```

It uses your `OPENAI_API_KEY` and the app's own system prompt. For the test set it adds a `teacher_label` field, and `evaluate.py` then reports student–teacher agreement as well as accuracy. Running it on anonymised real leads (with consent) is the best next step for the Medium class.

## Retrain

```bash
python -m venv .venv-train && .venv-train/bin/pip install -r requirements-train.txt
python -m finetune.generate_dataset                                   # only if the templates changed
.venv-train/bin/python -m finetune.train_lora --out finetune/model    # ~20 s on a laptop
python -m finetune.evaluate --report finetune/results/report.md       # gate + report (app venv, no torch)
```

On **Google Colab (free T4)**: clone the repo, `!pip install -r requirements-train.txt`, then `!python -m finetune.train_lora --out finetune/model`, and download the `finetune/model` folder. Baseline for comparison: `--mode head-only --out models/priority-head-only`, then `python -m finetune.evaluate --model finetune/model --model models/priority-head-only`.

## API

```bash
curl -s -X POST "$URL/classify-priority" -H "X-API-Key: la_..." -H "Content-Type: application/json" \
  -d '{"message":"Budget approved for 40 seats, please send the contract today."}'
# {"priority":"High","confidence":0.97,"scores":{"High":0.97,"Medium":0.02,"Low":0.01},
#  "model":"lora-all-MiniLM-L6-v2","latency_ms":3.1}
```

The same API keys, rate limits and input guardrails apply as on `/analyze-lead`. If the model files are missing (`PRIORITY_MODEL_DIR`, default `finetune/model`), it returns `503 classifier_unavailable` and the rest of the API keeps working.

## Design notes

- **Why a small encoder, not a fine-tuned GPT:** the task is classification, not generation. A 22M-parameter encoder fits in the free Render container, answers in milliseconds and costs nothing per call. Fine-tuning gpt-4o-mini would still cost per token and add a network round trip.
- **Why LoRA:** it trains 0.33% of the weights, so it trains in seconds on a laptop or free Colab, and it works much better than a frozen-encoder head (table above). After merging, the served model is identical in size and speed to a fully fine-tuned one.
- **Why commit the model (23 MB) to git:** it keeps the deployed artifact, the gate and the code in one reviewed commit, with no model registry account needed. If retrains get frequent, move it to a GitHub Release or the Hugging Face Hub and pin it by SHA-256 in the Dockerfile.
- **Limits:** until a LangSmith export is trained on, the training data is synthetic and English-only, and 84 test leads give a confidence interval of a few points. Treat the classifier as a cheap first pass, not the final say. `/analyze-lead` remains the source of truth for routing.
