"""Application settings loaded from environment / `.env` (never log secrets)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime config. `openai_api_key` is read from OPENAI_API_KEY; never log or return it."""

    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    rag_k: int = 3
    # Chroma L2 distance via similarity_search_with_score: **lower is better**.
    # Scores worse than (greater than) this threshold → low RAG confidence → force High.
    rag_score_threshold: float = 1.2
    # Hybrid retrieval: fuse Chroma vector hits with BM25 keyword hits (Reciprocal Rank Fusion).
    rag_hybrid: bool = True
    # Candidates pulled from each retriever before fusion + re-ranking (final count is rag_k).
    rag_candidate_k: int = 10
    # Re-ranker: auto (Cohere when COHERE_API_KEY is set, else local cross-encoder),
    # cohere, cross-encoder, or none. Failures fall back to fused order.
    rag_reranker: str = "auto"
    # Local cross-encoder (FlashRank ONNX, no torch). Empty cache dir → ./models/flashrank.
    rag_reranker_model: str = "ms-marco-MiniLM-L-12-v2"
    rag_reranker_cache_dir: str = ""
    # Cohere Rerank (optional). Never log or return the key.
    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-v3.5"
    # Per-request OpenAI chat timeout (seconds). Hard failure → HTTP 502 for n8n retry.
    openai_timeout_seconds: float = 30.0
    # Conversation memory (Redis). Empty REDIS_URL → memory soft-disabled (stateless).
    redis_url: str = ""
    # TTL for each thread's message list (seconds). Default 7 days.
    memory_ttl_seconds: int = 604800
    # Max role+content messages retained per thread (oldest dropped).
    memory_max_messages: int = 20
    # Classifier-first routing (routing.py): a confident Low from the fine-tuned classifier answers
    # /analyze-lead without calling RAG or OpenAI. 0.7 = 0 wrong routes on the hand-written test set.
    classifier_route_enabled: bool = True
    classifier_route_threshold: float = 0.7
    # Answer cache for /analyze-lead (semantic_cache.py): exact matches, then near-duplicates whose
    # numbers and classifier label also match. Threshold: see eval/cache_threshold.py.
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.92
    semantic_cache_ttl_seconds: int = 86400
    semantic_cache_max_entries: int = 500
    # Multi-agent Router webhooks (optional). Empty → stub notifications only.
    slack_webhook_url: str = ""
    email_webhook_url: str = ""
    # LangSmith tracing (optional). Off unless LANGSMITH_TRACING=true AND LANGSMITH_API_KEY is set.
    # Never log or return the key. Traces include prompts and user messages — see docs/tracing.md.
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "lead-agent"
    # Empty = LangSmith US cloud. EU: https://eu.api.smith.langchain.com
    langsmith_endpoint: str = ""
    # true = send traces without prompts / user messages (structure, timing, tokens and cost only).
    langsmith_hide_inputs: bool = False
    # Cost estimates in logs (USD per 1M tokens). Empty → built-in price table for OPENAI_MODEL.
    llm_price_input_per_1m: float | None = None
    llm_price_output_per_1m: float | None = None

    # API authentication (security.py, docs/security.md). Comma-separated "key_id:secret" pairs,
    # e.g. "n8n:la_...,demo:la_...". Several keys at once = rotation without downtime.
    # Empty = auth OFF (a warning is logged at startup). Generate one: python -m security new-key n8n
    api_keys: str = ""
    # HS256 secret for POST /auth/token (OAuth2 client credentials). Empty = JWT disabled,
    # API keys still work. Never log or return it.
    jwt_secret: str = ""
    jwt_ttl_seconds: int = 3600
    # Rate limits (slowapi / limits syntax). AI endpoints are limited per API key (per IP when
    # auth is off); the token endpoint per IP. Storage: memory:// (one instance) or redis://...
    rate_limit_enabled: bool = True
    rate_limit: str = "20/minute;200/day"
    rate_limit_token: str = "10/minute"
    rate_limit_storage_uri: str = "memory://"

    # Background jobs (jobs.py, docs/job-queue.md): arq on REDIS_URL. Without Redis, /jobs/* → 503.
    # In-process = the API also runs the worker (one free Render service). Set false when a
    # separate `arq jobs.WorkerSettings` process runs the jobs.
    job_worker_in_process: bool = True
    job_max_concurrency: int = 2
    job_max_tries: int = 3
    job_timeout_seconds: int = 120
    job_result_ttl_seconds: int = 3600
    # Polling GET /jobs/{id} has its own, looser per-key limit.
    rate_limit_poll: str = "120/minute"

    # Fine-tuned priority classifier (classifier.py, docs/fine-tuning.md): int8 ONNX model +
    # tokenizer from finetune/train_lora.py. Missing files → POST /classify-priority returns 503.
    priority_model_dir: str = "finetune/model"

    # Public demo page (demo.py, docs/demo.md): GET /demo, no API key. Per-IP rate limits plus a
    # global daily cap on OpenAI calls keep a public link cheap; the fine-tuned classifier is free.
    demo_enabled: bool = True
    demo_daily_llm_limit: int = 200
    demo_rate_limit: str = "6/minute;40/day"
    demo_classify_rate_limit: str = "30/minute"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Cached Settings instance (loads `.env` once per process)."""
    return Settings()
