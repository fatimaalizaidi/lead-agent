"""Local ChromaDB + LangChain helpers for company knowledge retrieval."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any

import httpx

# Chroma's PostHog telemetry is broken in this version (logs capture() errors per query); opt out
# before chromadb is imported. Set ANONYMIZED_TELEMETRY=True explicitly to re-enable.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain_chroma import Chroma  # noqa: E402
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from config import get_settings

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
KNOWLEDGE_DIR = ROOT / "knowledge"
CHROMA_DIR = ROOT / "data" / "chroma"
FINGERPRINT_PATH = CHROMA_DIR / ".kb_fingerprint"
COLLECTION_NAME = "company_kb"

# Reciprocal Rank Fusion constant (standard value from the original RRF paper).
RRF_K = 60
COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"

_vectorstore: Chroma | None = None
# (store id, KB fingerprint, BM25 index, chunk docs) — rebuilt when the KB changes.
_bm25_cache: tuple[int, str, BM25Okapi, list[Document]] | None = None
_cross_encoder: Any = None
_cross_encoder_failed = False


def _require_api_key() -> str:
    key = get_settings().openai_api_key
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Put it in `.env` or export it before using the knowledge retriever."
        )
    return key


def _default_k() -> int:
    return get_settings().rag_k


def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(api_key=_require_api_key())


def _kb_fingerprint() -> str:
    """Stable hash of knowledge/*.md paths + mtimes + sizes for cheap change detection."""
    h = hashlib.sha256()
    if not KNOWLEDGE_DIR.is_dir():
        return h.hexdigest()
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        stat = path.stat()
        h.update(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}".encode())
    return h.hexdigest()


def _title_from_markdown(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or fallback
    return fallback


def _load_knowledge_documents() -> list[Document]:
    docs: list[Document] = []
    if not KNOWLEDGE_DIR.is_dir():
        logger.warning("Knowledge directory missing: %s", KNOWLEDGE_DIR)
        return docs

    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = _title_from_markdown(text, fallback=path.stem.replace("-", " ").title())
        docs.append(
            Document(
                page_content=text,
                metadata={"title": title, "source": path.name},
            )
        )
    return docs


def _split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(docs)
    # Preserve article title on every chunk
    for chunk in chunks:
        if "title" not in chunk.metadata and docs:
            chunk.metadata["title"] = chunk.metadata.get("source", "Unknown")
    return chunks


def _collection_count(store: Chroma) -> int:
    try:
        return int(store._collection.count())  # noqa: SLF001 — Chroma public count via collection
    except Exception:
        try:
            return len(store.get().get("ids") or [])
        except Exception:
            return 0


def _needs_reindex(store: Chroma) -> bool:
    if _collection_count(store) == 0:
        return True
    current = _kb_fingerprint()
    if not FINGERPRINT_PATH.exists():
        return True
    try:
        return FINGERPRINT_PATH.read_text(encoding="utf-8").strip() != current
    except OSError:
        return True


def _write_fingerprint() -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    FINGERPRINT_PATH.write_text(_kb_fingerprint(), encoding="utf-8")


def _build_index(embeddings: OpenAIEmbeddings) -> Chroma:
    """(Re)build the Chroma collection from knowledge/*.md."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # Drop existing collection if present so rebuild is clean
    try:
        existing = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )
        existing.delete_collection()
    except Exception:
        pass

    docs = _load_knowledge_documents()
    if not docs:
        logger.warning("No knowledge/*.md files found; creating empty collection.")
        store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )
        _write_fingerprint()
        return store

    chunks = _split_documents(docs)
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    _write_fingerprint()
    logger.info("Indexed %d chunks from %d knowledge articles.", len(chunks), len(docs))
    return store


def get_vectorstore(*, force_rebuild: bool = False) -> Chroma:
    """Return a persisted Chroma store, indexing knowledge/*.md when empty or stale."""
    global _vectorstore
    embeddings = _embeddings()

    if force_rebuild:
        _vectorstore = _build_index(embeddings)
        return _vectorstore

    if _vectorstore is not None and not _needs_reindex(_vectorstore):
        return _vectorstore

    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    if _needs_reindex(store):
        store = _build_index(embeddings)

    _vectorstore = store
    return store


def get_retriever(k: int | None = None):
    """LangChain retriever over the company knowledge base."""
    return get_vectorstore().as_retriever(search_kwargs={"k": k if k is not None else _default_k()})


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _passage(doc: Document) -> str:
    """Chunk text with its article title, so later chunks (no "# Title" line) keep their topic."""
    title = str(doc.metadata.get("title") or "")
    if title and title not in doc.page_content:
        return f"{title}\n{doc.page_content}"
    return doc.page_content


def _chunk_key(doc: Document) -> tuple[str, str]:
    return (str(doc.metadata.get("source") or ""), doc.page_content)


def _bm25_index(store: Chroma) -> tuple[BM25Okapi, list[Document]] | None:
    """BM25 index over every chunk in the Chroma collection (cached per KB fingerprint)."""
    global _bm25_cache
    fingerprint = _kb_fingerprint()
    if _bm25_cache is not None and _bm25_cache[0] == id(store) and _bm25_cache[1] == fingerprint:
        return _bm25_cache[2], _bm25_cache[3]

    raw = store.get(include=["documents", "metadatas"])
    texts = raw.get("documents") or []
    metas = raw.get("metadatas") or [{}] * len(texts)
    docs = [Document(page_content=t, metadata=m or {}) for t, m in zip(texts, metas, strict=False) if t]
    if not docs:
        return None
    index = BM25Okapi([_tokenize(_passage(d)) for d in docs])
    _bm25_cache = (id(store), fingerprint, index, docs)
    return index, docs


def _keyword_search(store: Chroma, query: str, k: int) -> list[Document]:
    built = _bm25_index(store)
    tokens = _tokenize(query)
    if built is None or not tokens:
        return []
    index, docs = built
    scores = index.get_scores(tokens)
    ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    # Zero/negative BM25 = no useful term overlap; don't let it pad the fusion.
    return [docs[i] for i in ranked[:k] if scores[i] > 0]


def _reciprocal_rank_fusion(*rankings: list[Document]) -> list[Document]:
    """Merge ranked lists: each doc scores sum(1 / (RRF_K + rank)) across lists."""
    scores: dict[tuple[str, str], float] = {}
    docs: dict[tuple[str, str], Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            key = _chunk_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            docs.setdefault(key, doc)
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)]


def _get_cross_encoder() -> Any:
    """Lazy FlashRank cross-encoder (ONNX). None when unavailable — logged once."""
    global _cross_encoder, _cross_encoder_failed
    if _cross_encoder is not None or _cross_encoder_failed:
        return _cross_encoder
    settings = get_settings()
    cache_dir = settings.rag_reranker_cache_dir or str(ROOT / "models" / "flashrank")
    try:
        from flashrank import Ranker

        _cross_encoder = Ranker(model_name=settings.rag_reranker_model, cache_dir=cache_dir)
    except Exception:
        _cross_encoder_failed = True
        logger.warning("Cross-encoder re-ranker unavailable; using fused order.", exc_info=True)
    return _cross_encoder


def _rerank_cross_encoder(query: str, docs: list[Document]) -> list[Document] | None:
    ranker = _get_cross_encoder()
    if ranker is None:
        return None
    from flashrank import RerankRequest

    passages = [{"id": i, "text": _passage(d)} for i, d in enumerate(docs)]
    results = ranker.rerank(RerankRequest(query=query, passages=passages))
    return [docs[int(r["id"])] for r in results]


def _rerank_cohere(query: str, docs: list[Document]) -> list[Document] | None:
    settings = get_settings()
    if not settings.cohere_api_key:
        return None
    response = httpx.post(
        COHERE_RERANK_URL,
        headers={"Authorization": f"Bearer {settings.cohere_api_key}"},
        json={
            "model": settings.cohere_rerank_model,
            "query": query,
            "documents": [_passage(d) for d in docs],
        },
        timeout=10.0,
    )
    response.raise_for_status()
    results = response.json().get("results") or []
    return [docs[int(r["index"])] for r in results]


def _rerank(query: str, docs: list[Document]) -> tuple[list[Document], str]:
    """Re-order candidates by relevance. Returns (docs, reranker used); never raises."""
    mode = get_settings().rag_reranker.strip().lower()
    if mode == "none" or len(docs) < 2:
        return docs, "none"
    if mode == "auto":
        mode = "cohere" if get_settings().cohere_api_key else "cross-encoder"

    rerankers = {"cohere": _rerank_cohere, "cross-encoder": _rerank_cross_encoder}
    reranker = rerankers.get(mode)
    if reranker is None:
        logger.warning("Unknown RAG_RERANKER=%r; using fused order.", mode)
        return docs, "none"
    try:
        ranked = reranker(query, docs)
    except Exception as exc:
        logger.warning("%s re-rank failed (%s); using fused order.", mode, type(exc).__name__)
        return docs, "none"
    return (ranked, mode) if ranked else (docs, "none")


def retrieve_context(
    query: str, k: int | None = None
) -> tuple[list[dict[str, str]], float | None]:
    """
    Retrieve top-k knowledge chunks for a query with best distance score.

    Pipeline: Chroma vector search + BM25 keyword search (``RAG_HYBRID``), merged with
    Reciprocal Rank Fusion, then re-ranked (``RAG_RERANKER``) and cut to ``k``.

    ``best_score`` is the lowest Chroma L2 distance among vector hits (**lower is
    better**), or ``None`` when nothing was retrieved. It stays vector-only so the
    ``RAG_SCORE_THRESHOLD`` low-confidence check keeps its meaning.

    Chunks are ``{"title": ..., "content": ...}`` dicts, one per article: the top-k chunks
    of the same article are joined, never dropped (see ``_group_by_article``).
    """
    query = (query or "").strip()
    if not query:
        return [], None

    settings = get_settings()
    top_k = k if k is not None else _default_k()
    store = get_vectorstore()
    count = _collection_count(store)
    if count == 0:
        return [], None
    # Never ask Chroma for more results than it holds (it warns on every query otherwise).
    candidate_k = min(max(top_k, settings.rag_candidate_k), count)

    # Chroma default space is L2: smaller score = closer match.
    scored = store.similarity_search_with_score(query, k=candidate_k)
    best_score = min((float(score) for _, score in scored), default=None)
    vector_docs = [doc for doc, _ in scored]

    keyword_docs = _keyword_search(store, query, candidate_k) if settings.rag_hybrid else []
    candidates = _reciprocal_rank_fusion(vector_docs, keyword_docs)
    if not candidates:
        return [], None

    ranked, reranker = _rerank(query, candidates)
    logger.info(
        "rag retrieve vector_hits=%d keyword_hits=%d fused=%d reranker=%s k=%d",
        len(vector_docs),
        len(keyword_docs),
        len(candidates),
        reranker,
        top_k,
    )

    return _group_by_article(ranked[:top_k]), best_score


def _group_by_article(docs: list[Document]) -> list[dict[str, str]]:
    """
    One ``{"title", "content"}`` entry per article, in rank order of its best chunk.

    Every selected chunk is kept: chunks from the same article are joined (in rank order)
    rather than collapsed to one, so e.g. the second half of a long article still reaches
    the LLM when it was retrieved.
    """
    parts: dict[str, list[str]] = {}
    for doc in docs:
        title = str(doc.metadata.get("title") or doc.metadata.get("source") or "Untitled")
        content = doc.page_content.strip()
        article = parts.setdefault(title, [])
        if content and content not in article:
            article.append(content)
    return [{"title": title, "content": "\n\n".join(texts)} for title, texts in parts.items()]


def format_context_for_prompt(chunks: list[dict[str, str]]) -> str:
    """Render retrieved chunks into a compact prompt block."""
    if not chunks:
        return "(no matching knowledge-base articles)"
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[{i}] {chunk['title']}\n{chunk['content']}")
    return "\n\n".join(parts)


def knowledge_source_titles(chunks: list[dict[str, Any]]) -> list[str]:
    """Unique article titles in retrieval order."""
    seen: set[str] = set()
    titles: list[str] = []
    for chunk in chunks:
        title = str(chunk.get("title") or "").strip()
        if title and title not in seen:
            seen.add(title)
            titles.append(title)
    return titles
