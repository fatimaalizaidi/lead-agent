"""Local ChromaDB + LangChain helpers for company knowledge retrieval."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
KNOWLEDGE_DIR = ROOT / "knowledge"
CHROMA_DIR = ROOT / "data" / "chroma"
FINGERPRINT_PATH = CHROMA_DIR / ".kb_fingerprint"
COLLECTION_NAME = "company_kb"

_vectorstore: Chroma | None = None


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


def retrieve_context(
    query: str, k: int | None = None
) -> tuple[list[dict[str, str]], float | None]:
    """
    Retrieve top-k knowledge chunks for a query with best distance score.

    Uses Chroma ``similarity_search_with_score`` (L2 distance: **lower is better**).
    Returns ``(chunks, best_score)`` where ``best_score`` is the lowest distance
    among hits, or ``None`` when nothing was retrieved.

    Chunks are ``{"title": ..., "content": ...}`` dicts, deduplicated by article
    title (preferring longer content snippets when merging).
    """
    query = (query or "").strip()
    if not query:
        return [], None

    top_k = k if k is not None else _default_k()
    store = get_vectorstore()
    if _collection_count(store) == 0:
        return [], None

    # Chroma default space is L2: smaller score = closer match.
    scored = store.similarity_search_with_score(query, k=top_k)
    if not scored:
        return [], None

    best_score = min(float(score) for _, score in scored)

    by_title: dict[str, str] = {}
    order: list[str] = []
    for doc, _score in scored:
        title = str(doc.metadata.get("title") or doc.metadata.get("source") or "Untitled")
        content = doc.page_content.strip()
        if title not in by_title:
            by_title[title] = content
            order.append(title)
        elif len(content) > len(by_title[title]):
            by_title[title] = content

    chunks = [{"title": t, "content": by_title[t]} for t in order]
    return chunks, best_score


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
