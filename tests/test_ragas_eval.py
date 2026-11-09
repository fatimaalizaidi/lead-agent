"""Offline tests for the Ragas evaluation harness (no OpenAI calls, no Ragas scoring)."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval import run_ragas

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"


def test_golden_set_is_well_formed() -> None:
    golden = run_ragas.load_golden()
    assert len(golden) >= 20
    ids = [row["id"] for row in golden]
    assert len(ids) == len(set(ids)), "duplicate ids in golden.jsonl"
    for row in golden:
        assert (KNOWLEDGE_DIR / row["source"]).is_file(), row["id"]


def test_load_golden_limit_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    path.write_text('{"question": "q1", "reference": "r1"}\n\n{"question": "q2", "reference": "r2"}\n')
    assert [r["question"] for r in run_ragas.load_golden(path, limit=1)] == ["q1"]

    path.write_text('{"question": "q1"}\n')
    with pytest.raises(ValueError, match="reference"):
        run_ragas.load_golden(path)


@pytest.mark.parametrize(
    ("mode", "hybrid", "reranker"),
    [("vector", False, "none"), ("hybrid", True, "none"), ("hybrid-rerank", True, "auto")],
)
def test_configure_mode(mode: str, hybrid: bool, reranker: str) -> None:
    from config import get_settings

    run_ragas.configure_mode(mode)
    assert get_settings().rag_hybrid is hybrid
    assert get_settings().rag_reranker == reranker


def test_configured_mode_leaves_settings_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    from config import get_settings

    monkeypatch.setenv("RAG_HYBRID", "false")
    monkeypatch.setenv("RAG_RERANKER", "cohere")
    get_settings.cache_clear()
    run_ragas.configure_mode("configured")
    assert get_settings().rag_hybrid is False
    assert get_settings().rag_reranker == "cohere"


async def test_knowledge_agent_answer_captures_contexts(
    monkeypatch: pytest.MonkeyPatch, rag_chunks: list[dict[str, str]]
) -> None:
    import multiagent

    original_retrieve = lambda query, k=None: (rag_chunks, 0.4)  # noqa: E731
    monkeypatch.setattr(multiagent, "retrieve_context", original_retrieve)
    seen_threads: list[str] = []

    async def fake_node(state):
        seen_threads.append(state["thread_id"])
        chunks, _ = multiagent.retrieve_context(state["message"])
        return {"agent_reply": f"Growth is $149 ({len(chunks)} chunk)"}

    monkeypatch.setattr(multiagent, "knowledge_node", fake_node)

    answer, contexts = await run_ragas.knowledge_agent_answer("How much is Growth?")
    await run_ragas.knowledge_agent_answer("again")

    assert answer == "Growth is $149 (1 chunk)"
    assert contexts == ["Pricing Plans\nGrowth plan is $149/month per seat. Annual billing is 15% off."]
    assert multiagent.retrieve_context is original_retrieve, "retriever patch must be undone"
    assert len(set(seen_threads)) == 2, "each question needs a fresh memory thread"


async def test_build_samples_shape() -> None:
    async def answer_fn(question: str):
        return f"answer to {question}", ["ctx"]

    samples = await run_ragas.build_samples(
        [{"id": "a", "question": "q", "reference": "r"}], answer_fn
    )
    assert samples == [
        {"user_input": "q", "retrieved_contexts": ["ctx"], "response": "answer to q", "reference": "r"}
    ]


def test_summarize_and_threshold() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "faithfulness": [1.0, 0.5, None],
            "answer_relevancy": [0.9, 0.7, 0.8],
            "context_recall": [None, None, None],
        }
    )
    summary = run_ragas.summarize(df)
    assert summary == {"faithfulness": 0.75, "answer_relevancy": 0.8}
    assert run_ragas.failing_metrics(summary, 0.8) == ["faithfulness"]
    assert run_ragas.failing_metrics(summary, None) == []
