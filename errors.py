"""Application errors and n8n-friendly FastAPI exception handlers.

All client-facing error bodies use ``{"detail": "...", "error_code": "..."}``.
Never put stack traces, API keys, or raw upstream payloads in ``detail``.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException

from observability import REQUEST_ID_HEADER, get_request_id, log_error


class AppError(Exception):
    """Base application error with HTTP status and stable error_code."""

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, detail: str, *, error_code: str | None = None) -> None:
        self.detail = detail
        if error_code is not None:
            self.error_code = error_code
        super().__init__(detail)


class BadRequestError(AppError):
    """Client / input problems → HTTP 400."""

    status_code = 400
    error_code = "bad_request"


class UpstreamError(AppError):
    """Upstream (OpenAI, etc.) failure → HTTP 502 by default."""

    status_code = 502
    error_code = "upstream_error"


class AuthUpstreamError(UpstreamError):
    """OpenAI auth failure → HTTP 401."""

    status_code = 401
    error_code = "openai_auth_failed"


class RateLimitUpstreamError(UpstreamError):
    """OpenAI rate limit → HTTP 429."""

    status_code = 429
    error_code = "openai_rate_limit"


def error_body(detail: str, error_code: str) -> dict[str, str]:
    return {"detail": detail, "error_code": error_code}


def _safe_detail_from_http_exception(exc: StarletteHTTPException) -> tuple[str, str]:
    """Map Starlette/FastAPI HTTPException to (detail, error_code) without leaking internals."""
    raw = exc.detail
    if isinstance(raw, dict):
        detail = str(raw.get("detail") or raw.get("message") or "Request failed")
        code = str(raw.get("error_code") or f"http_{exc.status_code}")
        return detail, code
    detail = str(raw) if raw is not None else "Request failed"
    # Prefer short generic messages for unexpected 500s raised as HTTPException
    code_map = {
        400: "bad_request",
        401: "unauthorized",
        422: "validation_error",
        429: "rate_limit",
        500: "internal_error",
        502: "upstream_error",
    }
    return detail, code_map.get(exc.status_code, f"http_{exc.status_code}")


def _json_error(status_code: int, detail: str, error_code: str) -> JSONResponse:
    """Build error JSON and echo X-Request-ID when present in context."""
    headers: dict[str, str] = {}
    rid = get_request_id()
    if rid:
        headers[REQUEST_ID_HEADER] = rid
    return JSONResponse(
        status_code=status_code,
        content=error_body(detail, error_code),
        headers=headers,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach consistent JSON error handlers (n8n-friendly)."""

    @app.exception_handler(AppError)
    async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
        log_error(error_code=exc.error_code, status=exc.status_code)
        return _json_error(exc.status_code, exc.detail, exc.error_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Prefer 400 for n8n simplicity (avoid treating 422 as a special case).
        errors = exc.errors()
        if errors:
            first = errors[0]
            loc = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
            msg = first.get("msg", "Invalid request")
            detail = f"{loc}: {msg}" if loc else str(msg)
        else:
            detail = "Request validation failed"
        log_error(error_code="validation_error", status=400)
        return _json_error(400, detail, "validation_error")

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        _request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        detail, code = _safe_detail_from_http_exception(exc)
        log_error(error_code=code, status=exc.status_code)
        return _json_error(exc.status_code, detail, code)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        _request: Request, exc: Exception
    ) -> JSONResponse:
        # Log full exception server-side; never send stack/secrets to the client.
        log_error(error_code="internal_error", status=500)
        logger.exception("Unhandled error: {}", type(exc).__name__)
        return _json_error(
            500,
            "An unexpected error occurred. Please try again or contact support.",
            "internal_error",
        )


# Re-export for OpenAPI response models
class ErrorBodyModel:
    """Shape documented in OpenAPI; actual responses built via error_body()."""

    @staticmethod
    def schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "detail": {"type": "string"},
                "error_code": {"type": "string"},
            },
            "required": ["detail", "error_code"],
        }
