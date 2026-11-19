"""Public demo: a one-page UI at ``GET /demo`` plus the small, key-free API it calls.

A portfolio link can't ask visitors for an API key, so the demo has its own endpoints that never
expose one. They run the same code as the real API (``main.run_analyze_lead`` and the fine-tuned
classifier) under tighter limits:

- **Per IP:** ``DEMO_RATE_LIMIT`` for the OpenAI-backed analysis, ``DEMO_CLASSIFY_RATE_LIMIT`` for
  the free local classifier.
- **Global daily budget:** at most ``DEMO_DAILY_LLM_LIMIT`` OpenAI analyses per UTC day (counted
  in Redis, or in memory without it). Past the cap the page still answers with the classifier and
  says the LLM budget is used up, so a viral post costs cents, not dollars.
- **Same guardrails:** prompt-injection and length checks from guardrails.py.
- ``DEMO_ENABLED=false`` turns all of it off (404).

See docs/demo.md.
"""

# No `from __future__ import annotations`: FastAPI resolves the Request annotation at runtime.
import asyncio
import json
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, RedirectResponse
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from slowapi.util import get_remote_address

import classifier
import jobs
import memory
import routing
from config import get_settings
from errors import AppError
from guardrails import sanitize_message
from security import _rate_limits_off, limiter

PAGE = Path(__file__).resolve().parent / "static" / "demo.html"
DEMO_EMAIL = "visitor@demo.example"
MAX_DEMO_CHARS = 1000


class DemoDisabledError(AppError):
    status_code = 404
    error_code = "demo_disabled"


def _require_enabled() -> None:
    if not get_settings().demo_enabled:
        raise DemoDisabledError("The public demo is turned off.")


limit_demo = limiter.limit(
    lambda: get_settings().demo_rate_limit, key_func=get_remote_address, exempt_when=_rate_limits_off
)
limit_demo_classify = limiter.limit(
    lambda: get_settings().demo_classify_rate_limit, key_func=get_remote_address, exempt_when=_rate_limits_off
)


# ---- daily OpenAI budget ---------------------------------------------------------------------------

_local_counts: dict[str, int] = {}
_local_lock = threading.Lock()


def _budget_key() -> str:
    return f"demo:llm:{datetime.now(UTC):%Y-%m-%d}"


def _take_budget_sync() -> tuple[bool, int]:
    """Reserve one OpenAI analysis for today: (allowed, remaining after this one)."""
    limit = get_settings().demo_daily_llm_limit
    key = _budget_key()
    client = memory._get_redis_sync()
    used: int
    if client is not None:
        try:
            used = int(client.incr(key))
            if used == 1:
                client.expire(key, 2 * 86400)
        except Exception:
            client = None
    if client is None:
        with _local_lock:
            _local_counts[key] = used = _local_counts.get(key, 0) + 1
    return used <= limit, max(0, limit - used)


def _remaining_sync() -> int:
    limit = get_settings().demo_daily_llm_limit
    client = memory._get_redis_sync()
    used = 0
    if client is not None:
        try:
            used = int(client.get(_budget_key()) or 0)
        except Exception:
            client = None
    if client is None:
        used = _local_counts.get(_budget_key(), 0)
    return max(0, limit - used)


def reset_budget() -> None:
    """Tests only."""
    _local_counts.clear()


# ---- models ----------------------------------------------------------------------------------------


class DemoRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_DEMO_CHARS)
    thread_id: str | None = Field(default=None, max_length=64)

    @field_validator("message")
    @classmethod
    def _sanitize(cls, value: str) -> str:
        return sanitize_message(value)

    @field_validator("thread_id")
    @classmethod
    def _demo_threads_only(cls, value: str | None) -> str | None:
        """Visitors can only continue demo threads, never read memory from real (n8n) threads."""
        return value if value and value.startswith("demo-") else None


class ClassifierResult(BaseModel):
    priority: str
    confidence: float
    scores: dict[str, float]
    latency_ms: float
    model: str


class LLMResult(BaseModel):
    priority: str
    suggested_response: str
    rationale: str
    knowledge_sources: list[str]
    needs_human_review: bool
    latency_ms: float
    model: str


class DemoResponse(BaseModel):
    classifier: ClassifierResult | None
    llm: LLMResult | None
    llm_skipped_reason: str | None = None  # daily_budget_reached | llm_error
    thread_id: str | None = None
    budget_remaining: int


# ---- helpers ---------------------------------------------------------------------------------------


async def _classify(message: str) -> ClassifierResult | None:
    try:
        model = classifier.get_classifier()
    except AppError:
        return None
    started = time.perf_counter()
    prediction = await asyncio.to_thread(model.predict, message)
    return ClassifierResult(
        priority=prediction.label,
        confidence=round(prediction.confidence, 4),
        scores={k: round(v, 4) for k, v in prediction.scores.items()},
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        model=model.version,
    )


async def _analyze(message: str, thread_id: str | None) -> tuple[LLMResult | None, str | None, str | None]:
    from main import AnalyzeLeadRequest, run_analyze_lead

    thread_id = thread_id or f"demo-{uuid.uuid4().hex}"
    started = time.perf_counter()
    try:
        result = await run_analyze_lead(
            AnalyzeLeadRequest(message=message, email=DEMO_EMAIL, thread_id=thread_id),
            classifier_route=False,  # the page compares both models, so GPT always runs here
        )
    except AppError as exc:
        logger.warning("demo analysis failed error_code={}", exc.error_code)
        return None, "llm_error", thread_id
    return (
        LLMResult(
            priority=result.priority.value if hasattr(result.priority, "value") else str(result.priority),
            suggested_response=result.suggested_response,
            rationale=result.rationale,
            knowledge_sources=result.knowledge_sources,
            needs_human_review=result.needs_human_review,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            model=get_settings().openai_model,
        ),
        None,
        result.thread_id,
    )


def _model_card() -> dict[str, Any]:
    try:
        meta = json.loads((classifier.model_dir() / "meta.json").read_text())
    except (OSError, ValueError):
        return {}
    keys = ("base_model", "lora_rank", "trainable_params", "total_params", "train_examples", "size_mb")
    return {k: meta.get(k) for k in keys}


# ---- routes ----------------------------------------------------------------------------------------

router = APIRouter(include_in_schema=False)


@router.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse("/demo" if get_settings().demo_enabled else "/docs")


@router.get("/demo")
async def demo_page() -> FileResponse:
    _require_enabled()
    return FileResponse(PAGE, media_type="text/html", headers={"Cache-Control": "no-cache"})


@router.post("/demo/api/analyze", response_model=DemoResponse)
@limit_demo
async def demo_analyze(request: Request, body: DemoRequest) -> DemoResponse:
    """Both models on one lead: GPT + RAG (budgeted) and the fine-tuned classifier (free)."""
    _require_enabled()
    allowed, remaining = await asyncio.to_thread(_take_budget_sync)
    if allowed:
        (llm, reason, thread_id), clf = await asyncio.gather(
            _analyze(body.message, body.thread_id), _classify(body.message)
        )
    else:
        llm, reason, thread_id = None, "daily_budget_reached", body.thread_id
        clf = await _classify(body.message)
    return DemoResponse(
        classifier=clf, llm=llm, llm_skipped_reason=reason, thread_id=thread_id, budget_remaining=remaining
    )


@router.post("/demo/api/classify", response_model=DemoResponse)
@limit_demo_classify
async def demo_classify(request: Request, body: DemoRequest) -> DemoResponse:
    """Fine-tuned classifier only: no OpenAI call, so it doesn't use the daily budget."""
    _require_enabled()
    return DemoResponse(
        classifier=await _classify(body.message),
        llm=None,
        budget_remaining=await asyncio.to_thread(_remaining_sync),
    )


@router.get("/demo/api/info")
async def demo_info() -> dict[str, Any]:
    """Model card, today's remaining OpenAI budget and queue health (aggregate numbers only)."""
    _require_enabled()
    try:
        stats: dict[str, Any] | None = (await jobs.queue_stats()).model_dump(mode="json")
    except AppError:
        stats = None
    settings = get_settings()
    return {
        "llm_model": settings.openai_model,
        "classifier": _model_card(),
        "budget_remaining": await asyncio.to_thread(_remaining_sync),
        "budget_daily": settings.demo_daily_llm_limit,
        "queue": stats,
        "memory": memory.memory_enabled(),
        "routing": (await asyncio.to_thread(routing.stats)).model_dump(),
    }
