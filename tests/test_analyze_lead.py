"""POST /analyze-lead — OpenAI + RAG mocked; Redis via fakeredis."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import AsyncClient

from tests.conftest import SAMPLE_LEAD
from tests.openai_fakes import completion_with_json, make_async_openai_client


@pytest.mark.asyncio
async def test_analyze_lead_success(
    client: AsyncClient,
    mock_rag,
    fake_redis,
) -> None:
    fake_result = {
        "priority": "High",
        "suggested_response": "Happy to schedule a demo this week for your 200-seat rollout.",
        "rationale": "Clear budget and urgency for a large seat count.",
    }
    openai_client = make_async_openai_client(
        [completion_with_json(fake_result)]
    )

    with patch("main._get_async_client", return_value=openai_client):
        resp = await client.post("/analyze-lead", json=SAMPLE_LEAD)

    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == SAMPLE_LEAD["email"]
    assert body["priority"] == "High"
    assert body["suggested_response"] == fake_result["suggested_response"]
    assert body["rationale"] == fake_result["rationale"]
    assert body["knowledge_sources"] == ["Pricing Plans"]
    assert body["needs_human_review"] is False
    assert body["fallback_reason"] is None
    assert isinstance(body["thread_id"], str) and body["thread_id"]
    openai_client.chat.completions.create.assert_awaited()
    mock_rag.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_lead_persists_memory(
    client: AsyncClient,
    mock_rag,
    fake_redis,
) -> None:
    fake_result = {
        "priority": "Medium",
        "suggested_response": "Thanks for reaching out — happy to share next steps.",
        "rationale": "Exploratory interest with some commercial signal.",
    }
    openai_client = make_async_openai_client(
        [completion_with_json(fake_result)]
    )
    thread_id = "test-thread-analyze-1"

    with patch("main._get_async_client", return_value=openai_client):
        resp = await client.post(
            "/analyze-lead",
            json={**SAMPLE_LEAD, "thread_id": thread_id},
        )

    assert resp.status_code == 200
    assert resp.json()["thread_id"] == thread_id

    import memory

    history = memory.get_messages(thread_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == SAMPLE_LEAD["message"]
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == fake_result["suggested_response"]


@pytest.mark.asyncio
async def test_analyze_lead_rag_unavailable_flags_review(
    client: AsyncClient,
    fake_redis,
) -> None:
    """When RAG raises (soft path), endpoint still returns 200 with needs_human_review.

    Note: ``RuntimeError`` from retrieve is treated as missing API key (hard 500/AppError),
    so tests use ``OSError`` to exercise the soft RAG fallback.
    """
    fake_result = {
        "priority": "Low",
        "suggested_response": "Thanks for writing in — how can we help?",
        "rationale": "Vague inbound with no buying signal.",
    }
    openai_client = make_async_openai_client(
        [completion_with_json(fake_result)]
    )

    with (
        patch("main.retrieve_context", side_effect=OSError("chroma down")),
        patch("main._get_async_client", return_value=openai_client),
    ):
        resp = await client.post("/analyze-lead", json=SAMPLE_LEAD)

    assert resp.status_code == 200
    body = resp.json()
    assert body["needs_human_review"] is True
    assert body["fallback_reason"] == "rag_unavailable"
    assert body["priority"] == "High"
    assert body["knowledge_sources"] == []


@pytest.mark.asyncio
async def test_analyze_lead_validation_error(client: AsyncClient) -> None:
    """Empty message / bad email → AppError-style 400 (not FastAPI's default 422)."""
    resp = await client.post(
        "/analyze-lead",
        json={"message": "", "email": "not-an-email"},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert "error_code" in body
