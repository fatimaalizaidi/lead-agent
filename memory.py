"""Redis-backed multi-turn conversation memory (thread_id → message list)."""

from __future__ import annotations

import json
import uuid
from typing import Any

from loguru import logger

from config import get_settings

# Redis list key: thread:{id}:messages — JSON objects {"role","content"}
_KEY_PREFIX = "thread:"
_KEY_SUFFIX = ":messages"


def new_thread_id() -> str:
    """Generate a new conversation thread id (UUID4)."""
    return str(uuid.uuid4())


def resolve_thread_id(thread_id: str | None) -> str:
    """Return provided thread_id or a fresh UUID when missing/blank."""
    if thread_id and thread_id.strip():
        return thread_id.strip()
    return new_thread_id()


def _redis_key(thread_id: str) -> str:
    return f"{_KEY_PREFIX}{thread_id}{_KEY_SUFFIX}"


_redis_client: Any = None
_redis_init_attempted = False


def _get_redis_sync():
    """Lazy sync Redis client (None when REDIS_URL unset or connect fails).

    Successful clients are cached. Failed attempts are retried on the next call
    so a brief startup blip does not permanently disable memory.
    """
    global _redis_client, _redis_init_attempted
    if _redis_client is not None:
        return _redis_client

    settings = get_settings()
    url = (settings.redis_url or "").strip()
    if not url:
        return None

    try:
        import redis  # deferred so apps can import this module without redis installed

        client = redis.Redis.from_url(url, decode_responses=True)
        client.ping()
        _redis_client = client
        return _redis_client
    except Exception:
        if not _redis_init_attempted:
            logger.warning("Redis unavailable; conversation memory soft-disabled until reconnect")
            _redis_init_attempted = True
        return None


def memory_enabled() -> bool:
    """True when REDIS_URL is set and a client can be obtained."""
    return _get_redis_sync() is not None


def get_messages(thread_id: str) -> list[dict[str, str]]:
    """
    Load prior turns for ``thread_id`` (role + content only).

    Soft-fails to ``[]`` when Redis is down or the key is missing.
    """
    client = _get_redis_sync()
    if client is None:
        return []
    settings = get_settings()
    try:
        raw_items = client.lrange(_redis_key(thread_id), 0, -1)
    except Exception:
        logger.exception("Redis get_messages failed thread_id={}", thread_id[:8])
        return []

    messages: list[dict[str, str]] = []
    for raw in raw_items:
        try:
            item = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant", "system") and isinstance(content, str):
            messages.append({"role": role, "content": content})

    # Cap from the left (drop oldest) if somehow over limit.
    max_n = max(1, settings.memory_max_messages)
    if len(messages) > max_n:
        messages = messages[-max_n:]
    return messages


def append_messages(thread_id: str, messages: list[dict[str, str]]) -> None:
    """
    Append turn(s) and refresh TTL. Trims the list to ``memory_max_messages``.

    Soft-fails (logs) when Redis is down — request still succeeds without persistence.
    """
    if not messages:
        return
    client = _get_redis_sync()
    if client is None:
        return
    settings = get_settings()
    key = _redis_key(thread_id)
    max_n = max(1, settings.memory_max_messages)
    ttl = max(60, settings.memory_ttl_seconds)

    payload: list[str] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("user", "assistant", "system") or not isinstance(content, str):
            continue
        payload.append(json.dumps({"role": role, "content": content}, ensure_ascii=False))

    if not payload:
        return

    try:
        pipe = client.pipeline()
        pipe.rpush(key, *payload)
        # Keep only the newest max_n entries (LTRIM inclusive indices).
        pipe.ltrim(key, -max_n, -1)
        pipe.expire(key, ttl)
        pipe.execute()
    except Exception:
        logger.exception("Redis append_messages failed thread_id={}", thread_id[:8])


def clear_thread(thread_id: str) -> None:
    """Delete a thread's history (best-effort)."""
    client = _get_redis_sync()
    if client is None:
        return
    try:
        client.delete(_redis_key(thread_id))
    except Exception:
        logger.exception("Redis clear_thread failed thread_id={}", thread_id[:8])


def history_for_openai(thread_id: str) -> list[dict[str, Any]]:
    """Prior turns shaped for OpenAI ``messages`` (excludes system; callers add that)."""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in get_messages(thread_id)
        if m["role"] in ("user", "assistant")
    ]
