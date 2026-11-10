"""Tracing, cost estimates and tool-failure tracking — offline (no LangSmith, no OpenAI)."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from io import StringIO
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import tracing
from observability import (
    RequestIdMiddleware,
    begin_request_metrics,
    estimate_cost_usd,
    get_request_metrics,
    log_llm,
    log_request_end,
    setup_logging,
)


@pytest.fixture()
def logs() -> Iterator[StringIO]:
    buf = StringIO()
    setup_logging(log_to_file=False, log_json=False)
    from loguru import logger

    logger.remove()
    logger.add(lambda m: buf.write(str(m)), format="{level} {message}", level="INFO")
    yield buf
    setup_logging(log_to_file=False, log_json=False)


@pytest.fixture()
def restore_tracing_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """configure_tracing() writes os.environ; put everything back afterwards."""
    keys = [
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
        "LANGSMITH_HIDE_INPUTS",
    ]
    saved = {k: os.environ.get(k) for k in keys}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    monkeypatch.setattr(tracing, "_enabled", False)
    tracing._clear_sdk_env_cache()


class FakeRun:
    """Stands in for a LangSmith RunTree; records what end() was called with."""

    def __init__(self, **kwargs: Any) -> None:
        self.start = kwargs
        self.ended: dict[str, Any] | None = None

    def end(self, *, outputs: dict[str, Any] | None = None, error: str | None = None) -> None:
        self.ended = {"outputs": outputs, "error": error}


@pytest.fixture()
def fake_langsmith(monkeypatch: pytest.MonkeyPatch) -> list[FakeRun]:
    """Turn tracing on with an in-memory trace() so nothing is sent anywhere."""
    runs: list[FakeRun] = []

    @contextlib.contextmanager
    def fake_trace(**kwargs: Any) -> Iterator[FakeRun]:
        run = FakeRun(**kwargs)
        runs.append(run)
        yield run

    monkeypatch.setattr(tracing, "trace", fake_trace)
    monkeypatch.setattr(tracing, "_enabled", True)
    return runs


# --- cost ---------------------------------------------------------------------------


def test_estimate_cost_known_model_and_snapshot() -> None:
    # gpt-4o-mini: $0.15 / 1M input, $0.60 / 1M output
    assert estimate_cost_usd("gpt-4o-mini", 1000, 500) == pytest.approx(0.00045)
    assert estimate_cost_usd("gpt-4o-mini-2024-07-18", 1000, 500) == pytest.approx(0.00045)
    # Longest prefix wins: gpt-4o-mini is not priced as gpt-4o.
    assert estimate_cost_usd("gpt-4o", 1000, 500) == pytest.approx(0.0075)


def test_estimate_cost_unknown_model_or_usage() -> None:
    assert estimate_cost_usd("some-new-model", 1000, 500) is None
    assert estimate_cost_usd("gpt-4o-mini", None, None) is None


def test_estimate_cost_price_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from config import get_settings

    monkeypatch.setenv("LLM_PRICE_INPUT_PER_1M", "1.0")
    monkeypatch.setenv("LLM_PRICE_OUTPUT_PER_1M", "2.0")
    get_settings.cache_clear()
    assert estimate_cost_usd("some-new-model", 1_000_000, 1_000_000) == pytest.approx(3.0)


def test_request_end_reports_cost_and_tool_counts(logs: StringIO) -> None:
    begin_request_metrics(path="/multi-agent")
    log_llm(model="gpt-4o-mini", llm_latency_ms=10.0, prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    log_llm(model="gpt-4o-mini", llm_latency_ms=10.0, prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    tracing.run_tool("good", {}, lambda n, a: {"ok": True})
    tracing.run_tool("bad", {}, lambda n, a: {"ok": False, "error": "webhook 500"})
    log_request_end(status=200, total_ms=50.0)

    metrics = get_request_metrics()
    assert metrics is not None
    assert metrics.cost_usd == pytest.approx(0.0009)
    assert (metrics.tool_calls, metrics.tool_failures) == (2, 1)
    text = logs.getvalue()
    assert "cost_usd=0.000450" in text
    assert "cost_usd=0.000900 tool_calls=2 tool_failures=1" in text
    assert "WARNING tool name=bad ok=False" in text
    assert "INFO tool name=good ok=True" in text


# --- run_tool -----------------------------------------------------------------------


def test_run_tool_parse_error_skips_handler(logs: StringIO) -> None:
    def handler(name: str, args: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("must not run")

    result = tracing.run_tool("quote", {}, handler, parse_error=True)
    assert result == tracing.PARSE_ERROR_RESULT
    assert result is not tracing.PARSE_ERROR_RESULT  # callers may mutate it
    assert "tool name=quote ok=False" in logs.getvalue()


def test_run_tool_exception_is_logged_and_raised(logs: StringIO) -> None:
    def handler(name: str, args: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("boom")

    with pytest.raises(ValueError):
        tracing.run_tool("quote", {}, handler)
    assert "tool name=quote ok=False" in logs.getvalue()
    assert "error=ValueError" in logs.getvalue()


def test_run_tool_marks_failed_call_as_trace_error(fake_langsmith: list[FakeRun]) -> None:
    tracing.run_tool("trigger_slack_workflow", {"summary": "x"}, lambda n, a: {"ok": False, "error": "HTTP 500"})
    tracing.run_tool("check_calendar_availability", {}, lambda n, a: {"ok": True, "slots": []})

    failed, passed = fake_langsmith
    assert failed.start["name"] == "trigger_slack_workflow"
    assert failed.start["run_type"] == "tool"
    assert failed.start["inputs"] == {"arguments": {"summary": "x"}}
    assert failed.ended == {"outputs": {"result": {"ok": False, "error": "HTTP 500"}}, "error": "HTTP 500"}
    assert passed.ended is not None and passed.ended["error"] is None


# --- configuration ------------------------------------------------------------------


def test_tracing_needs_flag_and_key(monkeypatch: pytest.MonkeyPatch, restore_tracing_env: None) -> None:
    from config import get_settings

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "")
    get_settings.cache_clear()
    assert tracing.configure_tracing() is False
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"

    client = object()
    assert tracing.wrap_client(client) is client


def test_tracing_on_exports_env(monkeypatch: pytest.MonkeyPatch, restore_tracing_env: None) -> None:
    from config import get_settings

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-test-not-real")
    monkeypatch.setenv("LANGSMITH_PROJECT", "lead-agent-test")
    monkeypatch.setenv("LANGSMITH_HIDE_INPUTS", "true")
    get_settings.cache_clear()
    assert tracing.configure_tracing() is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == "lead-agent-test"
    assert os.environ["LANGSMITH_HIDE_INPUTS"] == "true"

    monkeypatch.setattr(tracing, "wrap_openai", lambda client: ("wrapped", client))
    client = object()
    assert tracing.wrap_client(client) == ("wrapped", client)


# --- middleware ---------------------------------------------------------------------


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(tracing.TracingMiddleware)
    app.add_middleware(RequestIdMiddleware)

    @app.post("/work")
    async def work() -> dict[str, str]:
        log_llm(model="gpt-4o-mini", llm_latency_ms=5.0, prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        tracing.run_tool("escalate", {}, lambda n, a: {"ok": False, "error": "down"})
        return {"status": "done"}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


async def test_middleware_one_root_trace_per_post(fake_langsmith: list[FakeRun]) -> None:
    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/health")).status_code == 200
        response = await client.post("/work", headers={"X-Request-ID": "rid-123"})
    assert response.status_code == 200

    names = [r.start["name"] for r in fake_langsmith]
    assert names == ["POST /work", "escalate"], "GET /health must not be traced"
    root = fake_langsmith[0]
    assert root.start["metadata"] == {"request_id": "rid-123"}
    assert root.ended is not None
    assert root.ended["error"] is None
    assert root.ended["outputs"] == {
        "status_code": 200,
        "llm_calls": 1,
        "total_tokens": 1500,
        "cost_usd": pytest.approx(0.00045),
        "tool_calls": 1,
        "tool_failures": 1,
    }


async def test_middleware_passthrough_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracing, "_enabled", False)

    def boom(**kwargs: Any) -> None:
        raise AssertionError("trace() must not be called when tracing is off")

    monkeypatch.setattr(tracing, "trace", boom)
    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.post("/work")).status_code == 200
