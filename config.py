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
    # Per-request OpenAI chat timeout (seconds). Hard failure → HTTP 502 for n8n retry.
    openai_timeout_seconds: float = 30.0
    # Conversation memory (Redis). Empty REDIS_URL → memory soft-disabled (stateless).
    redis_url: str = ""
    # TTL for each thread's message list (seconds). Default 7 days.
    memory_ttl_seconds: int = 604800
    # Max role+content messages retained per thread (oldest dropped).
    memory_max_messages: int = 20
    # Multi-agent Router webhooks (optional). Empty → stub notifications only.
    slack_webhook_url: str = ""
    email_webhook_url: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Cached Settings instance (loads `.env` once per process)."""
    return Settings()
