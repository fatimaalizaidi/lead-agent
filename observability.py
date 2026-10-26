"""Structured logging and request observability (loguru).

Never log API keys, full prompts with secrets, or raw PII dumps.
Request correlation uses ``X-Request-ID`` (incoming or generated UUID).
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"
_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"


def get_request_id() -> str | None:
    """Current request id from context (None outside a request)."""
    return _request_id_var.get()


def email_domain_only(email: str | None) -> str:
    """Return only the domain part of an email (no local-part / PII dump)."""
    if not email or "@" not in email:
        return "-"
    return email.rsplit("@", 1)[-1].strip().lower() or "-"


def _patch_record(record: dict[str, Any]) -> None:
    record["extra"].setdefault("request_id", _request_id_var.get() or "-")


class _InterceptHandler(logging.Handler):
    """Route stdlib logging into loguru (rag.py and third-party)."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(*, log_to_file: bool = True) -> None:
    """Configure loguru: stderr + optional rotating ``logs/app.log``."""
    logger.remove()
    logger.configure(patcher=_patch_record)

    fmt = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
        "request_id={extra[request_id]} | {name}:{function}:{line} - {message}"
    )
    logger.add(sys.stderr, format=fmt, level="INFO", enqueue=False)

    if log_to_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.add(
            LOG_DIR / "app.log",
            format=fmt,
            level="INFO",
            rotation="10 MB",
            retention="7 days",
            enqueue=False,
        )

    # Capture stdlib loggers used elsewhere in the app.
    logging.basicConfig(handlers=[_InterceptHandler()], level=logging.INFO, force=True)


def elapsed_ms(started: float) -> float:
    """Milliseconds since ``time.perf_counter()`` start, rounded to 1 decimal."""
    return round((time.perf_counter() - started) * 1000.0, 1)


def log_request_start(*, path: str, email_domain: str | None = None) -> None:
    logger.info(
        "request_start path={} email_domain={}",
        path,
        email_domain or "-",
    )


def log_rag(
    *,
    chroma_latency_ms: float,
    hit_count: int,
    best_score: float | None,
    low_confidence: bool,
    rag_unavailable: bool = False,
) -> None:
    score_str = f"{best_score:.4f}" if best_score is not None else "none"
    logger.info(
        "rag chroma_latency_ms={} hit_count={} best_score={} low_confidence={} "
        "rag_unavailable={}",
        chroma_latency_ms,
        hit_count,
        score_str,
        low_confidence,
        rag_unavailable,
    )


def log_llm(
    *,
    model: str,
    llm_latency_ms: float,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
) -> None:
    logger.info(
        "llm model={} llm_latency_ms={} prompt_tokens={} completion_tokens={} "
        "total_tokens={}",
        model,
        llm_latency_ms,
        prompt_tokens if prompt_tokens is not None else "-",
        completion_tokens if completion_tokens is not None else "-",
        total_tokens if total_tokens is not None else "-",
    )


def log_request_end(
    *,
    status: int,
    total_ms: float,
    needs_human_review: bool | None = None,
) -> None:
    review = needs_human_review if needs_human_review is not None else "-"
    logger.info(
        "request_end status={} total_ms={} needs_human_review={}",
        status,
        total_ms,
        review,
    )


def log_error(*, error_code: str, status: int | None = None) -> None:
    """Structured error line — error_code only, no stack secrets in the message."""
    if status is not None:
        logger.warning("error error_code={} status={}", error_code, status)
    else:
        logger.warning("error error_code={}", error_code)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Read or generate ``X-Request-ID``, bind log context, echo on response."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        incoming = (request.headers.get(REQUEST_ID_HEADER) or "").strip()
        request_id = incoming or str(uuid.uuid4())
        token = _request_id_var.set(request_id)
        request.state.request_id = request_id
        try:
            response = await call_next(request)
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            _request_id_var.reset(token)
