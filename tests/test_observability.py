"""Unit tests for structured logging helpers (no network / OpenAI)."""

from __future__ import annotations

import json
import logging
from io import StringIO

import pytest

from observability import (
    begin_request_metrics,
    get_request_metrics,
    log_llm,
    log_request_end,
    log_request_start,
    setup_logging,
)


@pytest.fixture()
def capture_logs(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    """Reconfigure loguru to human format writing into an in-memory buffer."""
    buf = StringIO()

    def _write(message: object) -> None:
        buf.write(str(message))

    # Avoid touching logs/app.log during unit tests.
    setup_logging(log_to_file=False, log_json=False)
    from loguru import logger

    logger.remove()
    logger.add(_write, format="{message}", level="INFO")
    yield buf
    setup_logging(log_to_file=False, log_json=False)


@pytest.fixture()
def capture_json_logs() -> StringIO:
    """JSON log lines into an in-memory buffer."""
    buf = StringIO()
    lines: list[str] = []

    def _sink(message: object) -> None:
        record = message.record  # type: ignore[attr-defined]
        from observability import _build_json_payload

        lines.append(json.dumps(_build_json_payload(record), default=str))

    setup_logging(log_to_file=False, log_json=True)
    from loguru import logger

    logger.remove()
    logger.configure(patcher=lambda r: r["extra"].setdefault("request_id", "test-rid"))
    logger.add(_sink, level="INFO")
    buf.lines = lines  # type: ignore[attr-defined]
    yield buf
    setup_logging(log_to_file=False, log_json=False)


def test_request_metrics_accumulate_llm(capture_logs: StringIO) -> None:
    begin_request_metrics(path="/agent-action")
    log_request_start(path="/agent-action", email_domain="acme.com")
    log_llm(
        model="gpt-4o-mini",
        llm_latency_ms=100.0,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    log_llm(
        model="gpt-4o-mini",
        llm_latency_ms=50.5,
        prompt_tokens=20,
        completion_tokens=7,
        total_tokens=27,
    )
    log_request_end(status=200, total_ms=200.0, path="/agent-action")

    metrics = get_request_metrics()
    assert metrics is not None
    assert metrics.llm_calls == 2
    assert metrics.llm_latency_ms == 150.5
    assert metrics.prompt_tokens == 30
    assert metrics.completion_tokens == 12
    assert metrics.total_tokens == 42

    text = capture_logs.getvalue()
    assert "request_start path=/agent-action" in text
    assert "request_end path=/agent-action status=200" in text
    assert "llm_latency_ms=150.5" in text
    assert "total_tokens=42" in text
    assert "error_code=-" in text


def test_request_end_includes_error_code(capture_logs: StringIO) -> None:
    begin_request_metrics(path="/multi-agent")
    log_request_end(
        status=502,
        total_ms=12.3,
        path="/multi-agent",
        error_code="openai_timeout",
    )
    text = capture_logs.getvalue()
    assert "error_code=openai_timeout" in text
    assert "path=/multi-agent" in text


def test_json_payload_shape(capture_json_logs: StringIO) -> None:
    begin_request_metrics(path="/analyze-lead")
    log_request_start(path="/analyze-lead", email_domain="acme.com")
    log_llm(
        model="gpt-4o-mini",
        llm_latency_ms=10.0,
        prompt_tokens=1,
        completion_tokens=2,
        total_tokens=3,
    )
    log_request_end(status=200, total_ms=11.0, path="/analyze-lead")

    lines = capture_json_logs.lines  # type: ignore[attr-defined]
    assert len(lines) >= 3
    end = json.loads(lines[-1])
    assert end["event"] == "request_end"
    assert end["path"] == "/analyze-lead"
    assert end["status"] == 200
    assert end["total_ms"] == 11.0
    assert end["llm_latency_ms"] == 10.0
    assert end["total_tokens"] == 3
    assert end["request_id"] == "test-rid"
    assert "OPENAI" not in json.dumps(end)


def test_env_flag_log_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_JSON", "true")
    # Should not raise; configures JSON sinks without writing files.
    setup_logging(log_to_file=False)
    # Reset to human for other tests / app import side effects.
    monkeypatch.delenv("LOG_JSON", raising=False)
    setup_logging(log_to_file=False, log_json=False)
    # Silence unused import noise from stdlib intercept
    assert logging.getLogger().handlers
