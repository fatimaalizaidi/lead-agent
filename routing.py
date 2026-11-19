"""Classifier-first routing for /analyze-lead: skip the LLM for clear low-intent leads.

The fine-tuned classifier (classifier.py) labels a lead in a few ms on CPU for free. When it says
**Low** with confidence >= ``CLASSIFIER_ROUTE_THRESHOLD`` (spam, "price?", wrong fit), the API
answers with a fixed polite reply and never calls RAG or OpenAI. Every other lead (High, Medium,
or an unsure Low) goes to GPT-4o-mini + RAG exactly as before.

Why only Low: a wrong Low on a real buyer loses a sale, and the classifier never confuses High
with a confident Low on the hand-written test set (``docs/fine-tuning.md``). At 0.7 it routes 15 of
84 test leads (18%) with 0 mistakes; at 0.6 one Medium lead slips through. Follow-up turns in a
thread always go to the LLM, since the classifier can't see the earlier conversation.

Counters (Redis when ``REDIS_URL`` is set, else in-process) feed ``GET /routing/stats``.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass

from loguru import logger
from pydantic import BaseModel, Field

import classifier
import memory
from config import get_settings

LOW_INTENT_REPLY = (
    "Thanks for reaching out! If you're looking at pricing or a demo, reply with your team size "
    "and timeline and we'll get back to you."
)

_KEY_TOTAL = "routing:analyze_total"
_KEY_SKIPPED = "routing:llm_skipped"
_local: dict[str, int] = {}
_lock = threading.Lock()


@dataclass(frozen=True)
class Decision:
    use_llm: bool
    label: str | None = None
    confidence: float | None = None
    reason: str = ""


def decide(message: str, *, has_history: bool) -> Decision:
    """Whether /analyze-lead needs the LLM for this message (sync; a few ms)."""
    settings = get_settings()
    if not settings.classifier_route_enabled:
        return Decision(True, reason="routing_disabled")
    if has_history:
        return Decision(True, reason="followup_turn")
    try:
        prediction = classifier.get_classifier().predict(message)
    except Exception as exc:  # missing model / onnx error: never block the LLM path
        logger.warning("classifier routing unavailable: {}", type(exc).__name__)
        return Decision(True, reason="classifier_unavailable")
    confident_low = prediction.label == "Low" and prediction.confidence >= settings.classifier_route_threshold
    return Decision(
        not confident_low,
        prediction.label,
        round(prediction.confidence, 4),
        "confident_low" if confident_low else "needs_llm",
    )


async def decide_async(message: str, *, has_history: bool) -> Decision:
    return await asyncio.to_thread(decide, message, has_history=has_history)


def record(skipped: bool) -> None:
    """Count one /analyze-lead request, and whether it skipped the LLM."""
    keys = [_KEY_TOTAL] + ([_KEY_SKIPPED] if skipped else [])
    client = memory._get_redis_sync()
    if client is not None:
        try:
            for key in keys:
                client.incr(key)
            return
        except Exception:
            pass
    with _lock:
        for key in keys:
            _local[key] = _local.get(key, 0) + 1


class RoutingStats(BaseModel):
    analyzed: int = Field(description="/analyze-lead requests answered (sync and background jobs)")
    llm_calls_saved: int = Field(description="Requests the classifier answered without calling OpenAI")
    saved_share: float = Field(description="llm_calls_saved / analyzed (0 when nothing analyzed yet)")
    threshold: float
    enabled: bool
    since: str = Field(description="Where the counts live: redis (survives restarts) or memory")


def stats() -> RoutingStats:
    settings = get_settings()
    client = memory._get_redis_sync()
    total = skipped = 0
    source = "memory"
    if client is not None:
        try:
            total = int(client.get(_KEY_TOTAL) or 0)
            skipped = int(client.get(_KEY_SKIPPED) or 0)
            source = "redis"
        except Exception:
            client = None
    if client is None:
        total, skipped = _local.get(_KEY_TOTAL, 0), _local.get(_KEY_SKIPPED, 0)
    return RoutingStats(
        analyzed=total,
        llm_calls_saved=skipped,
        saved_share=round(skipped / total, 4) if total else 0.0,
        threshold=settings.classifier_route_threshold,
        enabled=settings.classifier_route_enabled,
        since=source,
    )


def reset() -> None:
    """Tests only."""
    _local.clear()
