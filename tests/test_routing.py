"""Classifier-first routing on /analyze-lead (routing.py): confident Low skips RAG + OpenAI.

Runs the real shipped ONNX classifier; OpenAI and RAG are fakes that fail the test if a skipped
lead reaches them.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import AsyncClient

import classifier
import routing
from finetune import evaluate, generate_dataset
from tests.conftest import SAMPLE_LEAD
from tests.openai_fakes import completion_with_json, make_async_openai_client

SPAM = {"message": "Cheap SEO backlinks and guest posts! Reply now for our price list.", "email": "seo@spam.example"}
LLM_ANSWER = {"priority": "High", "suggested_response": "Happy to help.", "rationale": "Clear intent."}
KEY = "la_routing-test-secret-0123456789abcdef"


@pytest.fixture(autouse=True)
def _fresh() -> Iterator[None]:
    routing.reset()
    classifier.reset()
    yield
    routing.reset()
    classifier.reset()


def _llm(*answers: dict):
    return make_async_openai_client([completion_with_json(a) for a in answers])


def _must_not_call(*_a, **_k):
    raise AssertionError("OpenAI must not be called for a confident Low")


async def test_confident_low_skips_rag_and_openai(client: AsyncClient, mock_rag, fake_redis) -> None:
    with patch("main._get_async_client", side_effect=_must_not_call):
        resp = await client.post("/analyze-lead", json=SPAM)
    assert resp.status_code == 200
    body = resp.json()
    assert (body["priority"], body["priority_source"], body["classifier_priority"]) == ("Low", "classifier", "Low")
    assert body["classifier_confidence"] >= 0.7
    assert body["suggested_response"] == routing.LOW_INTENT_REPLY
    assert body["needs_human_review"] is False and body["knowledge_sources"] == []
    mock_rag.assert_not_called()
    # The turn is still remembered, so a follow-up has context.
    import memory

    assert [m["role"] for m in memory.get_messages(body["thread_id"])] == ["user", "assistant"]


async def test_buying_lead_still_goes_to_the_llm(client: AsyncClient, mock_rag, fake_redis) -> None:
    openai_client = _llm(LLM_ANSWER)
    with patch("main._get_async_client", return_value=openai_client):
        resp = await client.post("/analyze-lead", json=SAMPLE_LEAD)
    body = resp.json()
    assert (body["priority"], body["priority_source"]) == ("High", "llm")
    assert body["classifier_priority"] == "High", "classifier label is reported on the LLM path too"
    openai_client.chat.completions.create.assert_awaited_once()
    mock_rag.assert_called_once()


async def test_follow_up_turns_always_use_the_llm(client: AsyncClient, mock_rag, fake_redis) -> None:
    openai_client = _llm(LLM_ANSWER, LLM_ANSWER)
    with patch("main._get_async_client", return_value=openai_client):
        first = (await client.post("/analyze-lead", json=SAMPLE_LEAD)).json()
        resp = await client.post("/analyze-lead", json={**SPAM, "thread_id": first["thread_id"]})
    assert resp.json()["priority_source"] == "llm"
    assert openai_client.chat.completions.create.await_count == 2


async def test_routing_can_be_switched_off(
    client: AsyncClient, mock_rag, fake_redis, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLASSIFIER_ROUTE_ENABLED", "false")
    with patch("main._get_async_client", return_value=_llm(LLM_ANSWER)):
        body = (await client.post("/analyze-lead", json=SPAM)).json()
    assert (body["priority_source"], body["classifier_priority"]) == ("llm", None)


async def test_missing_model_falls_back_to_the_llm(
    client: AsyncClient, mock_rag, fake_redis, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PRIORITY_MODEL_DIR", str(tmp_path))
    with patch("main._get_async_client", return_value=_llm(LLM_ANSWER)):
        resp = await client.post("/analyze-lead", json=SPAM)
    assert resp.status_code == 200 and resp.json()["priority_source"] == "llm"


async def test_demo_page_still_runs_gpt_on_spam(client: AsyncClient, mock_rag) -> None:
    import demo

    demo.reset_budget()
    with patch("main._get_async_client", return_value=_llm({**LLM_ANSWER, "priority": "Low"})):
        body = (await client.post("/demo/api/analyze", json={"message": SPAM["message"]})).json()
    assert body["llm"] is not None and body["llm"]["priority"] == "Low"
    assert routing.stats().analyzed == 0, "demo traffic is not counted as API routing"


async def test_stats_count_saved_calls(
    client: AsyncClient, mock_rag, fake_redis, monkeypatch: pytest.MonkeyPatch
) -> None:
    with patch("main._get_async_client", return_value=_llm(LLM_ANSWER)):
        await client.post("/analyze-lead", json=SAMPLE_LEAD)
        await client.post("/analyze-lead", json=SPAM)
        await client.post("/analyze-lead", json={**SPAM, "email": "seo2@spam.example"})

    monkeypatch.setenv("API_KEYS", f"ops:{KEY}")
    from config import get_settings

    get_settings.cache_clear()
    assert (await client.get("/routing/stats")).status_code == 401
    stats = (await client.get("/routing/stats", headers={"X-API-Key": KEY})).json()
    assert (stats["analyzed"], stats["llm_calls_saved"], stats["saved_share"]) == (3, 2, 0.6667)
    assert stats["since"] == "redis" and stats["threshold"] == 0.7
    assert (await client.get("/demo/api/info")).json()["routing"]["llm_calls_saved"] == 2


def test_threshold_never_routes_a_buyer_on_the_test_set() -> None:
    """The safety claim in routing.py: at the configured threshold, every lead the classifier
    would answer alone is really Low on the hand-written test set, and it still covers some."""
    from config import get_settings

    rows = generate_dataset.read_jsonl(evaluate.DATA_DIR / "test.jsonl")
    threshold = get_settings().classifier_route_threshold
    preds = classifier.get_classifier().predict_batch([r["text"] for r in rows])
    routed = [r for r, p in zip(rows, preds, strict=True) if p.label == "Low" and p.confidence >= threshold]
    assert routed, "threshold too high: nothing would be routed"
    assert [r["label"] for r in routed if r["label"] != "Low"] == [], json.dumps(routed, indent=1)
