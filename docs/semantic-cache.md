# Answer cache (exact + semantic)

`POST /analyze-lead` checks a cache before it runs RAG and GPT-4o-mini. A hit returns the earlier answer in milliseconds, at no OpenAI cost. Code: `semantic_cache.py`.

```
lead → guardrails → classifier (confident spam stops here)
                  → cache: 1. exact  2. near-duplicate (semantic)   ← hit: earlier answer, no GPT
                  → hybrid RAG + GPT-4o-mini → answer is stored in the cache
```

## Two levels

| Level | How it matches | Cost | Catches |
|---|---|---|---|
| **Exact** | Lower-cased text with whitespace collapsed, then hashed | $0 | Demo example buttons, double form submits, n8n retries |
| **Semantic** | Cosine similarity of OpenAI `text-embedding-3-small` vectors (256 dims) | ~$0.00000002 per lead | Paraphrases ("How much is Growth?" / "What's the price of the Growth plan?") |

## Guards: when a near-duplicate is *not* reused

Two leads can read alike and still need different answers. "Budget approved for 40 seats" and "budget approved for 4 seats" are close in embedding space. A semantic hit is used only when **all three** of these hold:

1. similarity ≥ `SEMANTIC_CACHE_THRESHOLD` (default **0.92**);
2. the **numbers** in both messages are identical. Seats, budgets and dates are compared after dropping commas, so "30,000" and "30000" count as the same;
3. the fine-tuned classifier gives both messages the **same priority**.

Some answers are **never stored or served**:
- follow-up turns, because their answer depends on the conversation;
- answers flagged for human review, such as a weak knowledge-base match;
- leads the classifier answered alone, which are free already.

Entries expire after 24 hours (`SEMANTIC_CACHE_TTL_SECONDS`), so a pricing change in `knowledge/` can't keep serving an old answer for long. At most 500 near-duplicate vectors are kept (`SEMANTIC_CACHE_MAX_ENTRIES`), and the oldest are dropped first.

## Choosing the threshold

`python -m eval.cache_threshold --yes` embeds the 264 labelled leads (test and validation sets, ~$0.0001). It then looks for **dangerous pairs**: pairs that pass guards 2 and 3 but have different known priorities. It prints the most similar of these and recommends a threshold 0.02 above the highest. Set it on Render with `SEMANTIC_CACHE_THRESHOLD`.

**Measured on 2026-11-22** (264 leads): 393 pairs pass the number and label guards but have different known priorities. The most similar of them reaches only **0.575** cosine: *"Following up on the demo: we're in. Please set up 12 seats…"* (High) vs *"We're a 12-person startup. Maybe next quarter…"* (Medium). So the 0.92 default sits about 0.35 above any dangerous pair, and stays. The script's 0.85 recommendation is its built-in floor, not a measured edge.

Caveat: this data has no labelled paraphrase pairs, so it shows the threshold is **safe**, not how many real near-duplicates it catches. Lowering it towards 0.85 would catch more rewordings and is still far above 0.575. Keep 0.92 until real traffic shows misses worth catching.

## Where it lives

- **Storage:** Redis on Render, with answers under `cache:exact:*` (TTL) and the vector index in the `cache:semantic` hash. The index is loaded into process memory once and then compared with numpy, which is fast at this size (500 × 256 floats). Without Redis it falls back to process memory.
- **Response:** `cache` is `"exact"`, `"semantic"` or `null`, plus `cache_similarity`.
- **Stats:** `GET /routing/stats` reports `cache_hits` next to `llm_calls_saved`, and `saved_share` counts both.
- **Demo page:** a repeated example click is served from the cache and **doesn't use the daily GPT budget**. The card shows "⚡ cached answer, no GPT call".
- **Off switch:** `SEMANTIC_CACHE_ENABLED=false`.

## Cost and latency

On a miss, the cache adds one embedding call, roughly 0.1–0.3 s and $0.00000002. On a hit, it saves a GPT call (~$0.00015, ~2 s) and the retrieval. With real, unique leads the hit rate will be low. The clear win is repeated text: demo clicks and retries.
