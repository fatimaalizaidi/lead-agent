"""Hybrid retrieval (Chroma + BM25 + RRF) and re-ranking — offline, fake embeddings."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

import rag

KB = {
    "pricing.md": "# Pricing Plans\n\nGrowth costs $149 per seat per month. Annual billing is 15% off.",
    "refund-policy.md": "# Refund Policy\n\nAnnual subscriptions get a full refund within 14 days.",
    "sla.md": "# Service Level Agreement (SLA)\n\nTarget uptime is 99.9% monthly for the API.",
    "support-hours.md": "# Support Hours\n\nSupport is staffed 09:00-17:00 CET on weekdays.",
}


@pytest.fixture
def kb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Isolated knowledge dir + Chroma store with deterministic fake embeddings."""
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    for name, text in KB.items():
        (knowledge / name).write_text(text, encoding="utf-8")
    chroma = tmp_path / "chroma"
    monkeypatch.setattr(rag, "KNOWLEDGE_DIR", knowledge)
    monkeypatch.setattr(rag, "CHROMA_DIR", chroma)
    monkeypatch.setattr(rag, "FINGERPRINT_PATH", chroma / ".kb_fingerprint")
    monkeypatch.setattr(rag, "_embeddings", lambda: DeterministicFakeEmbedding(size=32))
    monkeypatch.setattr(rag, "_vectorstore", None)
    monkeypatch.setattr(rag, "_bm25_cache", None)
    monkeypatch.setenv("RAG_RERANKER", "none")
    yield
    if rag._vectorstore is not None:
        rag._vectorstore.delete_collection()


def _doc(text: str, source: str = "a.md") -> Document:
    return Document(page_content=text, metadata={"source": source, "title": source})


def test_rrf_rewards_docs_ranked_by_both_lists() -> None:
    a, b, c = _doc("a"), _doc("b"), _doc("c")
    fused = rag._reciprocal_rank_fusion([a, b, c], [c, b])
    # b and c appear in both lists; c is first in the second list.
    assert [d.page_content for d in fused] == ["c", "b", "a"]


def test_keyword_search_finds_exact_terms(kb: None) -> None:
    store = rag.get_vectorstore()
    hits = rag._keyword_search(store, "refund within 14 days", k=4)
    assert hits and hits[0].metadata["title"] == "Refund Policy"


def test_keyword_search_ignores_no_overlap(kb: None) -> None:
    store = rag.get_vectorstore()
    assert rag._keyword_search(store, "zebra quantum", k=4) == []


def test_hybrid_puts_keyword_match_first(kb: None, monkeypatch: pytest.MonkeyPatch) -> None:
    # Fake embeddings rank arbitrarily; BM25 must pull the keyword match into the top result.
    monkeypatch.setenv("RAG_CANDIDATE_K", "4")
    chunks, best = rag.retrieve_context("uptime 99.9% SLA", k=1)
    assert [c["title"] for c in chunks] == ["Service Level Agreement (SLA)"]
    assert best is not None  # still the vector L2 distance


def test_hybrid_off_is_vector_only(kb: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_HYBRID", "false")
    with patch.object(rag, "_keyword_search") as keyword:
        chunks, _ = rag.retrieve_context("uptime", k=2)
    keyword.assert_not_called()
    assert len(chunks) == 2


def test_cross_encoder_reorders(kb: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_RERANKER", "cross-encoder")

    class FakeRanker:
        def rerank(self, request: Any) -> list[dict[str, Any]]:
            # Put the Support Hours passage first regardless of fused order.
            order = sorted(request.passages, key=lambda p: "Support Hours" not in p["text"])
            return [{"id": p["id"], "score": 1.0} for p in order]

    monkeypatch.setattr(rag, "_cross_encoder", FakeRanker())
    chunks, _ = rag.retrieve_context("refund within 14 days", k=1)
    assert chunks[0]["title"] == "Support Hours"


def test_cohere_reorders(kb: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_RERANKER", "auto")
    monkeypatch.setenv("COHERE_API_KEY", "test-not-real")
    docs = [_doc("first"), _doc("second")]
    response = MagicMock()
    response.json.return_value = {"results": [{"index": 1, "relevance_score": 0.9}]}
    with patch.object(rag.httpx, "post", return_value=response) as post:
        ranked, used = rag._rerank("q", docs)
    assert used == "cohere"
    assert [d.page_content for d in ranked] == ["second"]
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-not-real"


def test_reranker_failure_falls_back_to_fused_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_RERANKER", "cohere")
    monkeypatch.setenv("COHERE_API_KEY", "test-not-real")
    docs = [_doc("first"), _doc("second")]
    with patch.object(rag.httpx, "post", side_effect=rag.httpx.ConnectError("down")):
        ranked, used = rag._rerank("q", docs)
    assert used == "none"
    assert ranked == docs


def test_group_by_article_keeps_every_chunk_of_an_article() -> None:
    # Regression: chunks of one article used to collapse to the longest, so the second
    # half of e.g. pricing.md ("volume discounts start at 50 seats") never reached the LLM.
    docs = [
        _doc("Starter $49, Growth $149 and a long intro paragraph", "Pricing Plans"),
        _doc("Uptime 99.9%", "SLA"),
        _doc("Volume discounts start at 50 seats", "Pricing Plans"),
        _doc("Uptime 99.9%", "SLA"),  # duplicate text is not repeated
    ]
    assert rag._group_by_article(docs) == [
        {
            "title": "Pricing Plans",
            "content": "Starter $49, Growth $149 and a long intro paragraph\n\n"
            "Volume discounts start at 50 seats",
        },
        {"title": "SLA", "content": "Uptime 99.9%"},
    ]


def test_retrieve_context_returns_later_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A multi-chunk article: a question about its second chunk must get that chunk's text."""
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    intro = "# Pricing Plans\n\n" + ("Starter plan details and seat limits. " * 30)
    (knowledge / "pricing.md").write_text(
        intro + "\n## Volume\nVolume discounts start at 50 seats on Growth.", encoding="utf-8"
    )
    chroma = tmp_path / "chroma"
    monkeypatch.setattr(rag, "KNOWLEDGE_DIR", knowledge)
    monkeypatch.setattr(rag, "CHROMA_DIR", chroma)
    monkeypatch.setattr(rag, "FINGERPRINT_PATH", chroma / ".kb_fingerprint")
    monkeypatch.setattr(rag, "_embeddings", lambda: DeterministicFakeEmbedding(size=32))
    monkeypatch.setattr(rag, "_vectorstore", None)
    monkeypatch.setattr(rag, "_bm25_cache", None)
    monkeypatch.setenv("RAG_RERANKER", "none")
    try:
        count = rag._collection_count(rag.get_vectorstore())
        assert count >= 2, "article should be split into several chunks"
        chunks, _ = rag.retrieve_context("When do volume discounts start?", k=count)
        assert [c["title"] for c in chunks] == ["Pricing Plans"]
        assert "50 seats" in chunks[0]["content"]
        assert "Starter plan details" in chunks[0]["content"]
    finally:
        rag._vectorstore.delete_collection()
