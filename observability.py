"""Structured logging and request observability (loguru).

Never log API keys, full prompts with secrets, or raw PII dumps.
Request correlation uses ``X-Request-ID`` (incoming or generated UUID).

Formats:
- Human (default): ``time | LEVEL | request_id=... | module:func:line - message``
- JSON (``LOG_JSON=true`` / ``1`` / ``yes``): one JSON object per line for collectors
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"
_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
_metrics_var: ContextVar[RequestMetrics | None] = ContextVar(
    "request_metrics", default=None
)

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"

# Extra keys copied into JSON log objects when present (collector-friendly).
_JSON_EXTRA_KEYS = (
    "event",
    "path",
    "status",
    "total_ms",
    "llm_latency_ms",
    "llm_calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "error_code",
    "needs_human_review",
    "email_domain",
    "model",
    "chroma_latency_ms",
    "hit_count",
    "best_score",
    "low_confidence",
    "rag_unavailable",
    "cost_usd",
    "tool",
    "tool_ok",
    "tool_latency_ms",
    "tool_error",
    "tool_calls",
    "tool_failures",
)

# OpenAI list prices, USD per 1M tokens (input, output). Override with LLM_PRICE_*_PER_1M.
MODEL_PRICES_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
}


def estimate_cost_usd(
    model: str, prompt_tokens: int | None, completion_tokens: int | None
) -> float | None:
    """USD cost of one call from token usage; ``None`` when usage or the price is unknown."""
    if prompt_tokens is None and completion_tokens is None:
        return None
    from config import get_settings

    settings = get_settings()
    prices = MODEL_PRICES_PER_1M.get(model)
    # Dated snapshots ("gpt-4o-mini-2024-07-18") use their base model's price.
    if prices is None:
        base = max((m for m in MODEL_PRICES_PER_1M if model.startswith(m + "-")), key=len, default=None)
        prices = MODEL_PRICES_PER_1M.get(base) if base else None
    price_in = settings.llm_price_input_per_1m
    price_out = settings.llm_price_output_per_1m
    if price_in is None or price_out is None:
        if prices is None:
            return None
        price_in = prices[0] if price_in is None else price_in
        price_out = prices[1] if price_out is None else price_out
    cost = ((prompt_tokens or 0) * price_in + (completion_tokens or 0) * price_out) / 1_000_000
    return round(cost, 6)


@dataclass
class RequestMetrics:
    """Per-request accumulators for LLM latency and token usage."""

    path: str | None = None
    llm_latency_ms: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    has_token_usage: bool = False
    cost_usd: float = 0.0
    has_cost: bool = False
    tool_calls: int = 0
    tool_failures: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def get_request_id() -> str | None:
    """Current request id from context (None outside a request)."""
    return _request_id_var.get()


def get_request_metrics() -> RequestMetrics | None:
    """Current request metrics accumulator (None outside a request)."""
    return _metrics_var.get()


def begin_request_metrics(*, path: str | None = None) -> RequestMetrics:
    """Reset / start metrics for the active request (middleware or endpoint)."""
    metrics = RequestMetrics(path=path)
    _metrics_var.set(metrics)
    return metrics


def email_domain_only(email: str | None) -> str:
    """Return only the domain part of an email (no local-part / PII dump)."""
    if not email or "@" not in email:
        return "-"
    return email.rsplit("@", 1)[-1].strip().lower() or "-"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _patch_record(record: dict[str, Any]) -> None:
    record["extra"].setdefault("request_id", _request_id_var.get() or "-")


def _build_json_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Collector-friendly flat JSON fields (request_id + event metrics)."""
    extra = record["extra"]
    payload: dict[str, Any] = {
        "timestamp": record["time"].strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "level": record["level"].name,
        "request_id": extra.get("request_id", "-"),
        "logger": record["name"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
    }
    for key in _JSON_EXTRA_KEYS:
        if key in extra:
            payload[key] = extra[key]
    if record["exception"] is not None:
        payload["exception"] = str(record["exception"])
    return payload


def _json_sink(message: Any) -> None:
    """Emit one JSON object per log record to stderr."""
    sys.stderr.write(json.dumps(_build_json_payload(message.record), default=str) + "\n")


def _json_format(record: dict[str, Any]) -> str:
    """loguru ``format=`` callable for rotating file sinks (must end with newline)."""
    # loguru evaluates the returned template against the record; embed prebuilt JSON.
    record["extra"]["_json"] = json.dumps(_build_json_payload(record), default=str)
    return "{extra[_json]}\n"


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


def _record_llm_into_metrics(
    *,
    llm_latency_ms: float,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    cost_usd: float | None = None,
) -> None:
    metrics = _metrics_var.get()
    if metrics is None:
        return
    metrics.llm_latency_ms = round(metrics.llm_latency_ms + llm_latency_ms, 1)
    metrics.llm_calls += 1
    if cost_usd is not None:
        metrics.cost_usd = round(metrics.cost_usd + cost_usd, 6)
        metrics.has_cost = True
    if prompt_tokens is not None:
        metrics.prompt_tokens += int(prompt_tokens)
        metrics.has_token_usage = True
    if completion_tokens is not None:
        metrics.completion_tokens += int(completion_tokens)
        metrics.has_token_usage = True
    if total_tokens is not None:
        metrics.total_tokens += int(total_tokens)
        metrics.has_token_usage = True


def setup_logging(*, log_to_file: bool = True, log_json: bool | None = None) -> None:
    """Configure loguru: stderr + optional rotating ``logs/app.log``.

    Set ``LOG_JSON=true`` (or pass ``log_json=True``) for one JSON object per line.
    Default remains human-readable key=value lines.
    """
    use_json = _env_flag("LOG_JSON") if log_json is None else log_json
    level = (os.getenv("LOG_LEVEL") or "INFO").strip().upper() or "INFO"

    logger.remove()
    logger.configure(patcher=_patch_record)

    if use_json:
        logger.add(_json_sink, level=level, enqueue=False)
        if log_to_file:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            logger.add(
                LOG_DIR / "app.log",
                format=_json_format,
                level=level,
                rotation="10 MB",
                retention="7 days",
                enqueue=False,
            )
    else:
        fmt = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "request_id={extra[request_id]} | {name}:{function}:{line} - {message}"
        )
        logger.add(sys.stderr, format=fmt, level=level, enqueue=False)

        if log_to_file:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            logger.add(
                LOG_DIR / "app.log",
                format=fmt,
                level=level,
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
    metrics = _metrics_var.get()
    if metrics is None:
        metrics = begin_request_metrics(path=path)
    else:
        metrics.path = path

    domain = email_domain or "-"
    logger.bind(event="request_start", path=path, email_domain=domain).info(
        "request_start path={} email_domain={}",
        path,
        domain,
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
    logger.bind(
        event="rag",
        chroma_latency_ms=chroma_latency_ms,
        hit_count=hit_count,
        best_score=score_str,
        low_confidence=low_confidence,
        rag_unavailable=rag_unavailable,
    ).info(
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
    cost_usd = estimate_cost_usd(model, prompt_tokens, completion_tokens)
    _record_llm_into_metrics(
        llm_latency_ms=llm_latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
    )
    cost = cost_usd if cost_usd is not None else "-"
    cost_text = f"{cost_usd:.6f}" if cost_usd is not None else "-"
    logger.bind(
        event="llm",
        model=model,
        llm_latency_ms=llm_latency_ms,
        prompt_tokens=prompt_tokens if prompt_tokens is not None else "-",
        completion_tokens=completion_tokens if completion_tokens is not None else "-",
        total_tokens=total_tokens if total_tokens is not None else "-",
        cost_usd=cost,
    ).info(
        "llm model={} llm_latency_ms={} prompt_tokens={} completion_tokens={} "
        "total_tokens={} cost_usd={}",
        model,
        llm_latency_ms,
        prompt_tokens if prompt_tokens is not None else "-",
        completion_tokens if completion_tokens is not None else "-",
        total_tokens if total_tokens is not None else "-",
        cost_text,
    )


def log_tool(*, name: str, ok: bool, latency_ms: float, error: str | None = None) -> None:
    """One line per tool call; failures (``ok=False``) log at WARNING and count per request."""
    metrics = _metrics_var.get()
    if metrics is not None:
        metrics.tool_calls += 1
        if not ok:
            metrics.tool_failures += 1
    err = error or "-"
    bound = logger.bind(event="tool", tool=name, tool_ok=ok, tool_latency_ms=latency_ms, tool_error=err)
    log = bound.info if ok else bound.warning
    log("tool name={} ok={} tool_latency_ms={} error={}", name, ok, latency_ms, err)


def log_request_end(
    *,
    status: int,
    total_ms: float,
    needs_human_review: bool | None = None,
    error_code: str | None = None,
    path: str | None = None,
) -> None:
    """Log request completion with latency, aggregated LLM metrics, and optional error."""
    metrics = _metrics_var.get()
    resolved_path = path or (metrics.path if metrics else None) or "-"
    llm_latency = round(metrics.llm_latency_ms, 1) if metrics else 0.0
    llm_calls = metrics.llm_calls if metrics else 0
    prompt_tokens: int | str
    completion_tokens: int | str
    total_tokens: int | str
    if metrics and metrics.has_token_usage:
        prompt_tokens = metrics.prompt_tokens
        completion_tokens = metrics.completion_tokens
        total_tokens = metrics.total_tokens
    else:
        prompt_tokens = "-"
        completion_tokens = "-"
        total_tokens = "-"

    review = needs_human_review if needs_human_review is not None else "-"
    code = error_code if error_code is not None else "-"
    cost: float | str = round(metrics.cost_usd, 6) if metrics and metrics.has_cost else "-"
    cost_text = f"{cost:.6f}" if isinstance(cost, float) else cost
    tool_calls = metrics.tool_calls if metrics else 0
    tool_failures = metrics.tool_failures if metrics else 0

    bound_fields: dict[str, Any] = {
        "event": "request_end",
        "path": resolved_path,
        "status": status,
        "total_ms": total_ms,
        "llm_latency_ms": llm_latency,
        "llm_calls": llm_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost,
        "tool_calls": tool_calls,
        "tool_failures": tool_failures,
        "needs_human_review": review,
        "error_code": code,
    }
    logger.bind(**bound_fields).info(
        "request_end path={} status={} total_ms={} llm_latency_ms={} llm_calls={} "
        "prompt_tokens={} completion_tokens={} total_tokens={} cost_usd={} "
        "tool_calls={} tool_failures={} needs_human_review={} error_code={}",
        resolved_path,
        status,
        total_ms,
        llm_latency,
        llm_calls,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        cost_text,
        tool_calls,
        tool_failures,
        review,
        code,
    )


def log_error(*, error_code: str, status: int | None = None) -> None:
    """Structured error line — error_code only, no stack secrets in the message."""
    if status is not None:
        logger.bind(event="error", error_code=error_code, status=status).warning(
            "error error_code={} status={}", error_code, status
        )
    else:
        logger.bind(event="error", error_code=error_code).warning(
            "error error_code={}", error_code
        )


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Read or generate ``X-Request-ID``, bind log context, echo on response."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        incoming = (request.headers.get(REQUEST_ID_HEADER) or "").strip()
        request_id = incoming or str(uuid.uuid4())
        token = _request_id_var.set(request_id)
        metrics_token = _metrics_var.set(RequestMetrics(path=request.url.path))
        request.state.request_id = request_id
        try:
            response = await call_next(request)
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            _request_id_var.reset(token)
            _metrics_var.reset(metrics_token)
