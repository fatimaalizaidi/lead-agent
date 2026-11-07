# Hybrid search + re-ranking

`rag.retrieve_context()` (used by `/analyze-lead`, `/agent-action` and `/multi-agent`) now runs:

1. **Vector search** — Chroma similarity over OpenAI embeddings (meaning: "money back" ≈ "refund").
2. **Keyword search** — BM25 (`rank-bm25`) over the same chunks (exact terms: emails, URLs, plan names, "Sev-2", "MSA").
3. **Fusion** — Reciprocal Rank Fusion: each chunk scores `Σ 1 / (60 + rank)` across both lists.
4. **Re-ranking** — a cross-encoder reads the question and each candidate together and re-orders them.
5. Top `RAG_K` chunks go into the LLM prompt.

Chunks are scored with their article title prepended, so a chunk from the middle of an article keeps its topic.

## Re-rankers

| `RAG_RERANKER` | What runs |
|---|---|
| `auto` (default) | Cohere if `COHERE_API_KEY` is set, otherwise the local cross-encoder |
| `cross-encoder` | FlashRank `ms-marco-MiniLM-L-12-v2` (ONNX, ~33 MB, CPU, no torch). Baked into the Docker image; downloaded to `./models/flashrank` on first use when run locally |
| `cohere` | Cohere Rerank API (`COHERE_RERANK_MODEL`, default `rerank-v3.5`) |
| `none` | Fused order only |

Any re-ranker failure (no network, bad key, timeout) logs a warning and falls back to the fused order — requests never fail because of it.

## Settings

| Variable | Default | Notes |
|---|---|---|
| `RAG_HYBRID` | `true` | `false` = vector search only |
| `RAG_CANDIDATE_K` | `10` | Candidates per retriever before fusion/re-rank (capped at chunk count) |
| `RAG_RERANKER` | `auto` | See table above |
| `RAG_RERANKER_MODEL` | `ms-marco-MiniLM-L-12-v2` | Must match the model baked in `Dockerfile` |
| `RAG_RERANKER_CACHE_DIR` | `./models/flashrank` | |
| `COHERE_API_KEY` | empty | Optional |

`best_score` (and therefore `RAG_SCORE_THRESHOLD` / low-confidence review) is still the best **vector** L2 distance, so its meaning is unchanged.

## Measured on the current `knowledge/` (5 articles, 9 chunks)

| Queries | Vector only | + BM25 | + BM25 + re-rank |
|---|---|---|---|
| 8 exact-term queries (emails, URLs, "Sev-2", "MSA"…), correct top article | 6/8 | 7/8 | **8/8** |
| 10 natural-language questions, correct article in the top 3 sent to the LLM | 10/10 | 10/10 | 10/10 |

With such a small KB the top 3 almost always contains the right article; the gains grow with the KB. The local re-ranker adds roughly 150–200 ms per query on CPU.

Every retrieval logs `rag retrieve vector_hits=… keyword_hits=… fused=… reranker=… k=…`.
