"""Answer cache on /analyze-lead (semantic_cache.py): exact and near-duplicate hits, and the guards
that keep similar-but-different leads apart. OpenAI chat and embeddings are fakes."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from httpx import AsyncClient

import routing
import semantic_cache
from tests.openai_fakes import completion_with_json, make_async_openai_client

ANSWER = {"priority": "High", "suggested_response": "Happy to send the order form today.", "rationale": "Budget + seats."}
LEAD = "Budget approved for 40 seats, please send the contract today."
PARAPHRASE = "We have budget sign-off for 40 seats. Can you send the contract today?"
FOUR_SEATS = "We have budget sign-off for 4 seats. Can you send the contract today?"


def _vec(seed: int, noise: float = 0.0, base: int = 7) -> list[float]:
    """Unit vectors: same ``base`` + small ``noise`` -> cosine close to 1."""
    rng = np.random.default_rng(base)
    v = rng.normal(size=semantic_cache.EMBED_DIMENSIONS)
    v = v + noise * np.random.default_rng(seed).normal(size=v.shape)
    return list(v / np.linalg.norm(v))


@pytest.fixture
def embeddings() -> Iterator[dict[str, list[float]]]:
    """Map message -> vector; unknown messages get an unrelated vector."""
    table: dict[str, list[float]] = {}

    async def fake(text: str) -> list[float]:
        return table.get(text, _vec(0, base=hash(text) % 1000 + 100))

    with patch("semantic_cache.embed", new=AsyncMock(side_effect=fake)):
        yield table


@pytest.fixture(autouse=True)
def _fresh_routing() -> Iterator[None]:
    routing.reset()
    yield
    routing.reset()


def _llm(n: int = 5):
    return make_async_openai_client([completion_with_json(ANSWER) for _ in range(n)])


async def _post(client: AsyncClient, message: str, **extra) -> dict:
    resp = await client.post("/analyze-lead", json={"message": message, "email": "a@b.example", **extra})
    assert resp.status_code == 200, resp.text
    return resp.json()


async def test_exact_repeat_skips_gpt(client: AsyncClient, mock_rag, fake_redis, embeddings) -> None:
    llm = _llm()
    with patch("main._get_async_client", return_value=llm):
        first = await _post(client, LEAD)
        second = await _post(client, "  budget approved for 40 seats, please send the CONTRACT today.  ")
    assert first["cache"] is None and second["cache"] == "exact"
    assert second["suggested_response"] == ANSWER["suggested_response"]
    assert llm.chat.completions.create.await_count == 1
    assert mock_rag.call_count == 1, "a cache hit skips retrieval too"
    assert any(k.startswith("cache:exact:") for k in fake_redis.keys()), "stored in Redis"


async def test_a_new_prompt_does_not_reuse_old_answers(
    client: AsyncClient, mock_rag, fake_redis, embeddings
) -> None:
    """Changing the answer prompt (or the knowledge files) must not serve answers from the old one."""
    embeddings[LEAD] = _vec(1, noise=0.1)
    embeddings[PARAPHRASE] = _vec(2, noise=0.1)
    llm = _llm()
    with patch("main._get_async_client", return_value=llm):
        await _post(client, LEAD)
        old_version = semantic_cache.answer_version()
        semantic_cache.answer_version.cache_clear()
        with patch("main.SYSTEM_PROMPT", "a new prompt"):
            assert semantic_cache.answer_version() != old_version
            exact = await _post(client, LEAD)
            near = await _post(client, PARAPHRASE)
        semantic_cache.answer_version.cache_clear()
    assert exact["cache"] is None, "the old exact answer is not served"
    assert near["cache"] == "semantic", "answers under the new prompt are cached as usual"
    assert llm.chat.completions.create.await_count == 2


async def test_near_duplicate_is_served(client: AsyncClient, mock_rag, fake_redis, embeddings) -> None:
    embeddings[LEAD] = _vec(1, noise=0.1)
    embeddings[PARAPHRASE] = _vec(2, noise=0.1)
    llm = _llm()
    with patch("main._get_async_client", return_value=llm):
        await _post(client, LEAD)
        body = await _post(client, PARAPHRASE)
    assert body["cache"] == "semantic" and body["cache_similarity"] >= 0.92
    assert llm.chat.completions.create.await_count == 1
    assert routing.stats().cache_hits == 1


async def test_different_numbers_never_share_an_answer(
    client: AsyncClient, mock_rag, fake_redis, embeddings
) -> None:
    embeddings[PARAPHRASE] = _vec(1)
    embeddings[FOUR_SEATS] = _vec(1)  # identical vectors on purpose: only the number differs
    llm = _llm()
    with patch("main._get_async_client", return_value=llm):
        await _post(client, PARAPHRASE)
        body = await _post(client, FOUR_SEATS)
    assert body["cache"] is None
    assert llm.chat.completions.create.await_count == 2


async def test_different_classifier_label_never_shares_an_answer(
    client: AsyncClient, mock_rag, fake_redis, embeddings
) -> None:
    embeddings[LEAD] = _vec(1)
    embeddings[PARAPHRASE] = _vec(1)
    labels = iter(["High", "Medium"])  # one classifier call per lookup
    llm = _llm()
    with patch("main._get_async_client", return_value=llm), patch("semantic_cache._label", lambda t: next(labels)):
        await _post(client, LEAD)
        body = await _post(client, PARAPHRASE)
    assert body["cache"] is None


async def test_below_threshold_is_a_miss(client: AsyncClient, mock_rag, fake_redis, embeddings) -> None:
    embeddings[LEAD] = _vec(1, noise=1.5)
    embeddings[PARAPHRASE] = _vec(2, noise=1.5)
    with patch("main._get_async_client", return_value=_llm()):
        await _post(client, LEAD)
        body = await _post(client, PARAPHRASE)
    assert body["cache"] is None


async def test_follow_ups_and_flagged_answers_are_not_cached(
    client: AsyncClient, fake_redis, embeddings, rag_chunks
) -> None:
    llm = _llm()
    with patch("main._get_async_client", return_value=llm), patch(
        "main.retrieve_context", return_value=(rag_chunks, 9.9)  # weak match -> human review
    ):
        flagged = await _post(client, LEAD)
        assert flagged["needs_human_review"] is True
        again = await _post(client, LEAD)
    assert again["cache"] is None, "flagged answers are never stored"

    with patch("main._get_async_client", return_value=llm), patch("main.retrieve_context", return_value=(rag_chunks, 0.4)):
        first = await _post(client, PARAPHRASE)
        follow = await _post(client, PARAPHRASE, thread_id=first["thread_id"])
    assert follow["cache"] is None, "a follow-up turn always goes to GPT"


async def test_cache_can_be_switched_off(
    client: AsyncClient, mock_rag, fake_redis, embeddings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "false")
    llm = _llm()
    with patch("main._get_async_client", return_value=llm):
        await _post(client, LEAD)
        body = await _post(client, LEAD)
    assert body["cache"] is None and llm.chat.completions.create.await_count == 2


async def test_index_survives_a_restart(client: AsyncClient, mock_rag, fake_redis, embeddings) -> None:
    embeddings[LEAD] = _vec(1, noise=0.1)
    embeddings[PARAPHRASE] = _vec(2, noise=0.1)
    with patch("main._get_async_client", return_value=_llm()):
        await _post(client, LEAD)
    semantic_cache.reset()  # new process: in-memory index gone, Redis still has it
    with patch("main._get_async_client", return_value=_llm()):
        body = await _post(client, PARAPHRASE)
    assert body["cache"] == "semantic"


async def test_demo_repeat_click_uses_no_budget(client: AsyncClient, mock_rag, fake_redis, embeddings) -> None:
    import demo

    demo.reset_budget()
    with patch("main._get_async_client", return_value=_llm()):
        first = (await client.post("/demo/api/analyze", json={"message": LEAD})).json()
        second = (await client.post("/demo/api/analyze", json={"message": LEAD})).json()
    assert first["llm"]["cache"] is None and second["llm"]["cache"] == "exact"
    assert second["budget_remaining"] == first["budget_remaining"], "the cached click is free"


def test_numbers_are_normalised() -> None:
    assert semantic_cache.numbers("$30,000 for 40 seats") == semantic_cache.numbers("40 seats, 30000 budget")
    assert semantic_cache.numbers("40 seats") != semantic_cache.numbers("4 seats")


def test_threshold_tool_finds_pairs_the_guards_miss() -> None:
    from eval import cache_threshold

    rows = [
        {"text": "need 40 seats", "label": "High"},
        {"text": "curious about 40 seats", "label": "Medium"},
        {"text": "need 10 seats", "label": "High"},
    ]
    vecs = np.array([[1.0, 0.0], [0.9, 0.1], [1.0, 0.0]], dtype=np.float32)
    pairs = cache_threshold.unsafe_pairs(rows, vecs, ["High", "High", "High"])
    assert [(i, j) for _, i, j in pairs] == [(0, 1)], "same numbers + same classifier label, different truth"
    assert cache_threshold.recommend(pairs) == round(pairs[0][0] + 0.02, 2)
    assert cache_threshold.recommend([]) == 0.85
