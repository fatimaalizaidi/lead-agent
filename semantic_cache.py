"""Answer cache for /analyze-lead: exact matches first, then near-duplicates (semantic), in Redis.

Two levels, checked before RAG and GPT:

1. **Exact:** the message, lower-cased with whitespace collapsed, hashed. Free and risk-free.
   Catches the demo page's example buttons, double form submissions and n8n retries.
2. **Semantic:** the message is embedded (OpenAI ``text-embedding-3-small`` at 256 dimensions,
   ~$0.00000002 per lead) and compared by cosine similarity with earlier answered leads.

A semantic hit is only used when **all three** hold, because two leads can read alike and still
need different answers ("40 seats by Friday" vs "4 seats next year"):

- similarity >= ``SEMANTIC_CACHE_THRESHOLD`` (default 0.92, see ``eval/cache_threshold.py``);
- the **numbers** in both messages are identical (seats, budgets, dates);
- the fine-tuned classifier gives both messages the **same priority**.

Never cached: follow-up turns (their answer depends on the conversation), answers flagged for
human review, and leads the classifier answered alone (already free). Entries expire after
``SEMANTIC_CACHE_TTL_SECONDS`` (default 24 h), so a pricing change in the knowledge base can't
keep serving an old answer for long. At most ``SEMANTIC_CACHE_MAX_ENTRIES`` near-duplicate
vectors are kept (oldest dropped). Without Redis the cache lives in process memory.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np
from loguru import logger

import classifier
import memory
from config import get_settings

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMENSIONS = 256
_EXACT_PREFIX = "cache:exact:"
_INDEX_KEY = "cache:semantic"
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")

_lock = threading.Lock()
_local_answers: dict[str, tuple[float, str]] = {}  # key -> (expires_at, json) when Redis is off
_index: list[dict] | None = None  # in-process copy of the semantic index (loaded from Redis once)


@dataclass
class CachedAnswer:
    priority: str
    suggested_response: str
    rationale: str
    knowledge_sources: list[str]


@dataclass
class Lookup:
    answer: CachedAnswer | None
    kind: str | None = None  # "exact" | "semantic"
    similarity: float | None = None
    vector: list[float] | None = None  # reused by store() on a miss
    label: str | None = None


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def numbers(text: str) -> list[str]:
    """Every number in the message, commas dropped: "30,000 for 40 seats" -> ["30000", "40"]."""
    return sorted(n.replace(",", "") for n in _NUMBER_RE.findall(text))


def exact_key(text: str) -> str:
    return _EXACT_PREFIX + hashlib.sha256(normalize(text).encode()).hexdigest()[:32]


def _label(text: str) -> str | None:
    try:
        return classifier.get_classifier().predict(text).label
    except Exception:
        return None


# ---- storage (Redis, or process memory) ---------------------------------------------------------


def _get(key: str) -> str | None:
    client = memory._get_redis_sync()
    if client is not None:
        try:
            return client.get(key)
        except Exception:
            return None
    with _lock:
        expires, value = _local_answers.get(key, (0.0, ""))
        return value if expires > time.time() else None


def _set(key: str, value: str, ttl: int) -> None:
    client = memory._get_redis_sync()
    if client is not None:
        try:
            client.set(key, value, ex=ttl)
            return
        except Exception:
            pass
    with _lock:
        _local_answers[key] = (time.time() + ttl, value)


def _load_index() -> list[dict]:
    global _index
    if _index is not None:
        return _index
    entries: list[dict] = []
    client = memory._get_redis_sync()
    if client is not None:
        try:
            for raw in client.hvals(_INDEX_KEY):
                entries.append(json.loads(raw))
        except Exception:
            entries = []
    for entry in entries:
        entry["vec"] = np.frombuffer(base64.b64decode(entry["v"]), dtype=np.float32)
    _index = sorted(entries, key=lambda e: e["t"])
    return _index


def _add_to_index(key: str, vector: list[float], nums: list[str], label: str | None) -> None:
    settings = get_settings()
    vec = np.asarray(vector, dtype=np.float32)
    vec = vec / (np.linalg.norm(vec) or 1.0)
    entry = {"k": key, "n": nums, "l": label, "t": time.time(), "v": base64.b64encode(vec.tobytes()).decode()}
    with _lock:
        index = _load_index()
        index[:] = [e for e in index if e["k"] != key]
        index.append({**entry, "vec": vec})
        dropped = index[: max(0, len(index) - settings.semantic_cache_max_entries)]
        del index[: len(dropped)]
    client = memory._get_redis_sync()
    if client is not None:
        try:
            client.hset(_INDEX_KEY, key, json.dumps(entry))
            if dropped:
                client.hdel(_INDEX_KEY, *[e["k"] for e in dropped])
        except Exception:
            pass


# ---- embedding ------------------------------------------------------------------------------------


async def embed(text: str) -> list[float] | None:
    """OpenAI embedding of the message; None on any failure (the cache then just misses)."""
    try:
        from main import _get_async_client

        resp = await _get_async_client().embeddings.create(
            model=EMBED_MODEL, input=text, dimensions=EMBED_DIMENSIONS
        )
        vector = list(resp.data[0].embedding)
        return vector if len(vector) == EMBED_DIMENSIONS else None
    except Exception as exc:
        logger.warning("semantic cache: embedding failed ({})", type(exc).__name__)
        return None


# ---- public API -----------------------------------------------------------------------------------


def peek_exact(text: str) -> bool:
    """Is there an exact cached answer? (The demo checks this before spending its daily budget.)"""
    return get_settings().semantic_cache_enabled and _get(exact_key(text)) is not None


def _best_match(vector: list[float], nums: list[str], label: str | None) -> tuple[str, float] | None:
    vec = np.asarray(vector, dtype=np.float32)
    vec = vec / (np.linalg.norm(vec) or 1.0)
    ttl = get_settings().semantic_cache_ttl_seconds
    now = time.time()
    with _lock:
        candidates = [
            e for e in _load_index()
            if e["n"] == nums and e["l"] == label and now - e["t"] < ttl
        ]
    if not candidates:
        return None
    sims = np.stack([e["vec"] for e in candidates]) @ vec
    best = int(sims.argmax())
    return candidates[best]["k"], float(sims[best])


async def lookup(text: str) -> Lookup:
    settings = get_settings()
    if not settings.semantic_cache_enabled:
        return Lookup(None)
    raw = await asyncio.to_thread(_get, exact_key(text))
    if raw:
        return Lookup(CachedAnswer(**json.loads(raw)), "exact", 1.0)

    label = await asyncio.to_thread(_label, text)
    vector = await embed(text)
    if vector is None or label is None:
        return Lookup(None, vector=vector, label=label)
    match = await asyncio.to_thread(_best_match, vector, numbers(text), label)
    if match and match[1] >= settings.semantic_cache_threshold:
        cached = await asyncio.to_thread(_get, match[0])
        if cached:
            return Lookup(CachedAnswer(**json.loads(cached)), "semantic", round(match[1], 4), vector, label)
    return Lookup(None, vector=vector, label=label)


async def store(text: str, answer: CachedAnswer, miss: Lookup) -> None:
    """Remember an answer for this message (exact) and, with a vector, for near-duplicates."""
    settings = get_settings()
    if not settings.semantic_cache_enabled:
        return
    key = exact_key(text)
    await asyncio.to_thread(_set, key, json.dumps(asdict(answer)), settings.semantic_cache_ttl_seconds)
    if miss.vector is not None and miss.label is not None:
        await asyncio.to_thread(_add_to_index, key, miss.vector, numbers(text), miss.label)


def reset() -> None:
    """Tests only."""
    global _index
    with _lock:
        _local_answers.clear()
        _index = None
