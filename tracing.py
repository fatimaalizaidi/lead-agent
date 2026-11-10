"""Optional LangSmith tracing for the LLM pipelines.

Off by default. Turn on with ``LANGSMITH_TRACING=true`` and ``LANGSMITH_API_KEY`` (env or ``.env``).
When on, every POST request becomes one trace in LangSmith with nested runs for:

- the LangGraph graph and each agent node (traced automatically by LangChain),
- every OpenAI chat call, with tokens, latency and cost (``wrap_client``),
- knowledge-base retrieval (``traced_retriever``),
- every tool call, marked as an error when the tool returns ``ok: false`` (``run_tool``).

When off, all helpers are pass-throughs. Tool outcomes are logged either way (``log_tool``).
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Callable
from typing import Any, TypeVar

from langsmith import trace, traceable
from langsmith.wrappers import wrap_openai
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from config import get_settings
from observability import elapsed_ms, get_request_id, get_request_metrics, log_tool

_enabled = False
ClientT = TypeVar("ClientT")
ToolHandler = Callable[[str, dict[str, Any]], dict[str, Any]]

PARSE_ERROR_RESULT = {"ok": False, "error": "Could not parse tool arguments as JSON"}


def configure_tracing() -> bool:
    """Read settings once at startup and point the LangSmith / LangChain SDKs at them."""
    global _enabled
    settings = get_settings()
    _enabled = bool(settings.langsmith_tracing and settings.langsmith_api_key.strip())
    flag = "true" if _enabled else "false"
    # pydantic-settings reads .env without exporting it; the SDKs only look at os.environ.
    os.environ["LANGSMITH_TRACING"] = flag
    os.environ["LANGCHAIN_TRACING_V2"] = flag  # LangGraph / LangChain auto-tracing
    if _enabled:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.strip()
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        if settings.langsmith_endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
        if settings.langsmith_hide_inputs:
            os.environ["LANGSMITH_HIDE_INPUTS"] = "true"
    _clear_sdk_env_cache()
    if settings.langsmith_tracing and not _enabled:
        logger.warning("LANGSMITH_TRACING=true but LANGSMITH_API_KEY is empty; tracing stays off.")
    logger.info("tracing langsmith={} project={}", _enabled, settings.langsmith_project if _enabled else "-")
    return _enabled


def _clear_sdk_env_cache() -> None:
    """The LangSmith SDK caches env lookups; drop the cache so the new values apply."""
    try:
        from langsmith import utils as ls_utils

        ls_utils.get_env_var.cache_clear()
    except AttributeError:
        pass


def tracing_enabled() -> bool:
    return _enabled


def wrap_client(client: ClientT) -> ClientT:
    """Trace every chat completion (prompt, reply, tokens, cost) when tracing is on."""
    return wrap_openai(client) if _enabled else client  # type: ignore[arg-type]


def _retrieval_outputs(outputs: Any) -> dict[str, Any]:
    """Show retrieved chunks the way LangSmith renders retriever runs."""
    result = outputs.get("output") if isinstance(outputs, dict) else outputs
    if not isinstance(result, tuple) or len(result) != 2:
        return {"output": result}
    chunks, best_score = result
    docs = [
        {"page_content": c.get("content", ""), "type": "Document", "metadata": {"title": c.get("title", "")}}
        for c in chunks
    ]
    return {"documents": docs, "best_vector_distance": best_score}


traced_retriever = traceable(run_type="retriever", name="retrieve_context", process_outputs=_retrieval_outputs)


def run_tool(
    name: str,
    arguments: dict[str, Any],
    handler: ToolHandler,
    *,
    parse_error: bool = False,
) -> dict[str, Any]:
    """
    Run one tool call: trace it (when on) and log its outcome (always).

    A result with ``ok: false`` (bad arguments, unknown tool, failed webhook…) counts as a
    failed function call: the trace run is marked as an error and the log line is a WARNING.
    """
    started = time.perf_counter()
    span = (
        trace(name=name, run_type="tool", inputs={"arguments": arguments})
        if _enabled
        else contextlib.nullcontext()
    )
    try:
        with span as run:
            result = dict(PARSE_ERROR_RESULT) if parse_error else handler(name, arguments)
            ok = not (isinstance(result, dict) and result.get("ok") is False)
            error = None if ok else str(result.get("error") or "tool returned ok=false")
            if run is not None:
                run.end(outputs={"result": result}, error=error)
    except Exception as exc:
        log_tool(name=name, ok=False, latency_ms=elapsed_ms(started), error=type(exc).__name__)
        raise
    log_tool(name=name, ok=ok, latency_ms=elapsed_ms(started), error=error)
    return result


class TracingMiddleware(BaseHTTPMiddleware):
    """
    One root trace per POST request, so retrieval, agents, LLM and tool runs nest under it.

    Must sit inside ``RequestIdMiddleware`` (add it first) so the request id and per-request
    cost/latency metrics are available. GET requests (``/health`` probes) are never traced.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not _enabled or request.method != "POST":
            return await call_next(request)
        path = request.url.path
        with trace(
            name=f"POST {path}",
            run_type="chain",
            inputs={"path": path},
            metadata={"request_id": get_request_id()},
            tags=["api"],
        ) as run:
            response = await call_next(request)
            metrics = get_request_metrics()
            summary: dict[str, Any] = {"status_code": response.status_code}
            if metrics is not None:
                summary.update(
                    llm_calls=metrics.llm_calls,
                    total_tokens=metrics.total_tokens,
                    cost_usd=round(metrics.cost_usd, 6) if metrics.has_cost else None,
                    tool_calls=metrics.tool_calls,
                    tool_failures=metrics.tool_failures,
                )
            error = f"HTTP {response.status_code}" if response.status_code >= 500 else None
            run.end(outputs=summary, error=error)
        return response
