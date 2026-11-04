"""Shared fixtures: env isolation, fakeredis memory, httpx ASGI client, OpenAI patches."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import patch

import fakeredis
import pytest
from httpx import ASGITransport, AsyncClient

# Force offline-safe settings before any app import that calls get_settings().
# Never put real secrets here — placeholders only.
os.environ["OPENAI_API_KEY"] = "sk-test-offline-key-not-real"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"  # never contacted; fakeredis used
os.environ["SLACK_WEBHOOK_URL"] = ""
os.environ["EMAIL_WEBHOOK_URL"] = ""
os.environ.setdefault("OPENAI_MODEL", "gpt-4o-mini")


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    """Reload Settings between tests so env monkeypatches take effect."""
    from config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fake_redis() -> Iterator[fakeredis.FakeRedis]:
    """In-memory Redis stand-in so conversation memory works without a live server."""
    import memory as memory_mod

    client = fakeredis.FakeRedis(decode_responses=True)
    # Reset module-level client cache and inject FakeRedis.
    prev_client = memory_mod._redis_client
    prev_attempted = memory_mod._redis_init_attempted
    memory_mod._redis_client = client
    memory_mod._redis_init_attempted = True
    try:
        yield client
    finally:
        memory_mod._redis_client = prev_client
        memory_mod._redis_init_attempted = prev_attempted
        client.flushall()


@pytest.fixture
def rag_chunks() -> list[dict[str, str]]:
    return [
        {
            "title": "Pricing Plans",
            "source": "pricing.md",
            "content": "Growth plan is $149/month per seat. Annual billing is 15% off.",
        }
    ]


@pytest.fixture
def mock_rag(rag_chunks: list[dict[str, str]]) -> Iterator[Any]:
    """Skip Chroma / embedding calls; return deterministic KB chunks + score."""
    with patch("main.retrieve_context", return_value=(rag_chunks, 0.4)) as p:
        yield p


@pytest.fixture
def mock_rag_multiagent(rag_chunks: list[dict[str, str]]) -> Iterator[Any]:
    """RAG used by multi-agent Support node (imports retrieve_context from rag)."""
    with patch("multiagent.retrieve_context", return_value=(rag_chunks, 0.4)) as p:
        yield p


@pytest.fixture
async def client(fake_redis: fakeredis.FakeRedis) -> AsyncIterator[AsyncClient]:
    """httpx AsyncClient against the FastAPI app (no real network)."""
    # Import after env is set so Settings sees the test key.
    from main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


SAMPLE_LEAD = {
    "message": "We need a demo this week for a 200-seat rollout. Budget is approved.",
    "email": "alex@acme.com",
}

SAMPLE_AGENT = {
    "message": "Can I get a Growth quote for 25 seats?",
    "email": "buyer@acme.com",
}

SAMPLE_MULTI = {
    "message": "What is your refund policy for annual plans?",
    "email": "support-lead@acme.com",
}
