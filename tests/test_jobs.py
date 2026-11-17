"""Background job queue (jobs.py): enqueue → worker → status, retries, ownership, 503 without Redis.

arq runs on fakeredis; the worker runs in burst mode (process every queued job, then return).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import patch

import arq.worker
import fakeredis
import pytest
import redis.asyncio as aioredis
from arq.connections import ArqRedis
from arq.worker import Worker
from fakeredis.aioredis import FakeConnection
from httpx import AsyncClient

import jobs
from errors import BadRequestError, RateLimitUpstreamError
from tests.conftest import SAMPLE_LEAD, SAMPLE_MULTI
from tests.openai_fakes import completion_with_content, completion_with_json, make_async_openai_client

FAKE_RESULT = {"priority": "High", "suggested_response": "Happy to help.", "rationale": "Clear intent."}
N8N_KEY = "la_n8n-test-secret-0123456789abcdef"
DEMO_KEY = "la_demo-test-secret-0123456789abcdef"


@pytest.fixture
async def queue(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[ArqRedis]:
    """arq pool on fakeredis, injected as the app's pool."""

    async def _no_info(*_args, **_kwargs) -> None:  # fakeredis has no INFO command
        return None

    monkeypatch.setattr(arq.worker, "log_redis_info", _no_info)
    pool = ArqRedis(
        connection_pool=aioredis.ConnectionPool(connection_class=FakeConnection, server=fakeredis.FakeServer())
    )
    monkeypatch.setattr(jobs, "_pool", pool)
    yield pool
    await pool.aclose()


async def run_worker(pool: ArqRedis) -> None:
    """Process every queued (and due) job, then return."""
    worker = Worker(redis_pool=pool, burst=True, poll_delay=0.01, handle_signals=False, **jobs._worker_options())
    await worker.async_run()


@pytest.fixture
def fake_llm(mock_rag):
    client = make_async_openai_client(lambda *a, **k: completion_with_json(FAKE_RESULT))
    with patch("main._get_async_client", return_value=client):
        yield client


# ---- happy paths --------------------------------------------------------------------------------


async def test_analyze_lead_job_end_to_end(client: AsyncClient, queue: ArqRedis, fake_llm) -> None:
    accepted = await client.post("/jobs/analyze-lead", json=SAMPLE_LEAD, headers={"X-Request-ID": "req-123"})
    assert accepted.status_code == 202
    body = accepted.json()
    assert body["status"] == "queued"
    assert body["status_url"] == f"/jobs/{body['job_id']}"

    queued = (await client.get(body["status_url"])).json()
    assert (queued["status"], queued["kind"]) == ("queued", "analyze-lead")

    await run_worker(queue)

    done = (await client.get(body["status_url"])).json()
    assert done["status"] == "complete"
    assert done["result"]["priority"] == "High"
    assert done["result"]["suggested_response"] == FAKE_RESULT["suggested_response"]
    assert done["error"] is None and done["finished_at"]

    info = await jobs.Job(body["job_id"], queue).info()
    assert info.kwargs["request_id"] == "req-123", "worker logs keep the original request id"


async def test_multi_agent_job(client: AsyncClient, queue: ArqRedis, mock_rag_multiagent) -> None:
    # Any model output is fine here: this checks the job wiring, not the graph (test_multi_agent.py).
    openai_client = make_async_openai_client(lambda *a, **k: completion_with_content("Happy to help."))
    with patch("multiagent._get_async_client", return_value=openai_client):
        job_id = (await client.post("/jobs/multi-agent", json=SAMPLE_MULTI)).json()["job_id"]
        await run_worker(queue)

    done = (await client.get(f"/jobs/{job_id}")).json()
    assert done["status"] == "complete", done
    assert done["kind"] == "multi-agent"
    assert done["result"]["agent_reply"]
    assert done["result"]["email"] == SAMPLE_MULTI["email"]


# ---- failures and retries -----------------------------------------------------------------------


async def test_transient_openai_error_is_retried(client: AsyncClient, queue: ArqRedis, mock_rag, monkeypatch) -> None:
    monkeypatch.setattr(jobs, "retry_delay_seconds", lambda _try: 0)
    calls = {"n": 0}

    async def flaky(body):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitUpstreamError("OpenAI rate limit exceeded.")
        from main import AnalyzeLeadResponse

        return AnalyzeLeadResponse(
            email=body.email, priority="Low", suggested_response="ok", rationale="r", thread_id="t"
        )

    with patch("main.run_analyze_lead", flaky):
        job_id = (await client.post("/jobs/analyze-lead", json=SAMPLE_LEAD)).json()["job_id"]
        await run_worker(queue)

    done = (await client.get(f"/jobs/{job_id}")).json()
    assert (done["status"], done["tries"], calls["n"]) == ("complete", 2, 2)


async def test_retries_stop_at_max_tries(client: AsyncClient, queue: ArqRedis, monkeypatch) -> None:
    monkeypatch.setattr(jobs, "retry_delay_seconds", lambda _try: 0)

    async def always_429(_body):
        raise RateLimitUpstreamError("OpenAI rate limit exceeded.")

    with patch("main.run_analyze_lead", always_429):
        job_id = (await client.post("/jobs/analyze-lead", json=SAMPLE_LEAD)).json()["job_id"]
        await run_worker(queue)

    done = (await client.get(f"/jobs/{job_id}")).json()
    assert done["status"] == "failed"
    assert done["tries"] == 3
    assert done["error"]["error_code"] == "openai_rate_limit"


async def test_non_retryable_error_fails_at_once(client: AsyncClient, queue: ArqRedis) -> None:
    calls = {"n": 0}

    async def bad(_body):
        calls["n"] += 1
        raise BadRequestError("message: too long", error_code="validation_error")

    with patch("main.run_analyze_lead", bad):
        job_id = (await client.post("/jobs/analyze-lead", json=SAMPLE_LEAD)).json()["job_id"]
        await run_worker(queue)

    done = (await client.get(f"/jobs/{job_id}")).json()
    assert (done["status"], done["error"]["status_code"], calls["n"]) == ("failed", 400, 1)


async def test_unexpected_crash_is_reported_without_internals(client: AsyncClient, queue: ArqRedis) -> None:
    async def crash(_body):
        raise RuntimeError("secret internal detail")

    with patch("main.run_analyze_lead", crash):
        job_id = (await client.post("/jobs/analyze-lead", json=SAMPLE_LEAD)).json()["job_id"]
        await run_worker(queue)

    done = (await client.get(f"/jobs/{job_id}")).json()
    assert done["status"] == "failed"
    assert done["error"]["error_code"] == "job_crashed"
    assert "secret" not in str(done)


# ---- validation, ownership, availability --------------------------------------------------------


async def test_invalid_body_is_rejected_before_queueing(client: AsyncClient, queue: ArqRedis) -> None:
    resp = await client.post("/jobs/analyze-lead", json={"email": "not-an-email", "message": "hi"})
    assert resp.status_code == 400
    assert await queue.zcard("arq:queue") == 0


async def test_unknown_job_404(client: AsyncClient, queue: ArqRedis) -> None:
    resp = await client.get("/jobs/does-not-exist")
    assert (resp.status_code, resp.json()["error_code"]) == (404, "job_not_found")


async def test_jobs_are_private_to_their_api_key(
    client: AsyncClient, queue: ArqRedis, fake_llm, monkeypatch: pytest.MonkeyPatch
) -> None:
    from config import get_settings

    monkeypatch.setenv("API_KEYS", f"n8n:{N8N_KEY},demo:{DEMO_KEY}")
    get_settings.cache_clear()

    assert (await client.post("/jobs/analyze-lead", json=SAMPLE_LEAD)).status_code == 401
    accepted = await client.post("/jobs/analyze-lead", json=SAMPLE_LEAD, headers={"X-API-Key": N8N_KEY})
    status_url = accepted.json()["status_url"]

    assert (await client.get(status_url, headers={"X-API-Key": N8N_KEY})).status_code == 200
    other = await client.get(status_url, headers={"X-API-Key": DEMO_KEY})
    assert other.status_code == 404, "another key must not see (or learn about) this job"
    assert (await client.get(status_url)).status_code == 401


async def test_no_redis_returns_503(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from config import get_settings

    monkeypatch.setattr(jobs, "_pool", None)
    monkeypatch.setenv("REDIS_URL", "")
    get_settings.cache_clear()

    resp = await client.post("/jobs/analyze-lead", json=SAMPLE_LEAD)
    assert (resp.status_code, resp.json()["error_code"]) == (503, "job_queue_unavailable")
    # The synchronous endpoint keeps working without Redis.
    assert (await client.get("/health")).status_code == 200


# ---- queue stats (GET /jobs/stats) --------------------------------------------------------------


async def test_stats_on_an_empty_queue(client: AsyncClient, queue: ArqRedis) -> None:
    resp = await client.get("/jobs/stats")
    assert resp.status_code == 200, "'stats' must not be read as a job id"
    body = resp.json()
    assert {k: body[k] for k in ("queued", "scheduled_retries", "in_progress", "complete", "failed")} == dict.fromkeys(
        ("queued", "scheduled_retries", "in_progress", "complete", "failed"), 0
    )
    assert body["by_kind"] == {} and body["wait_seconds"] is None and body["oldest_queued_seconds"] is None
    assert body["worker"]["alive"] is False, "no worker has written a heartbeat yet"
    assert body["worker"]["max_concurrency"] == 2


async def test_stats_count_queued_then_finished_jobs(client: AsyncClient, queue: ArqRedis, fake_llm) -> None:
    for _ in range(2):
        await client.post("/jobs/analyze-lead", json=SAMPLE_LEAD)
    queued = (await client.get("/jobs/stats")).json()
    assert (queued["queued"], queued["complete"]) == (2, 0)
    assert queued["oldest_queued_seconds"] >= 0

    async def bad(_body):
        raise BadRequestError("nope", error_code="validation_error")

    await run_worker(queue)
    with patch("main.run_analyze_lead", bad):
        await client.post("/jobs/analyze-lead", json=SAMPLE_LEAD)
        await run_worker(queue)

    stats = (await client.get("/jobs/stats")).json()
    assert (stats["queued"], stats["in_progress"], stats["complete"], stats["failed"]) == (0, 0, 2, 1)
    assert stats["by_kind"] == {"analyze-lead": {"complete": 2, "failed": 1}}
    assert stats["run_seconds"]["p50"] >= 0 and stats["wait_seconds"]["max"] >= 0
    assert stats["worker"]["alive"] is True and "j_complete=" in stats["worker"]["heartbeat"]


async def test_stats_split_due_and_deferred_jobs(client: AsyncClient, queue: ArqRedis) -> None:
    """A retry waiting out its backoff sits in the queue with a future score: not 'queued'."""
    await queue.enqueue_job("analyze_lead_job", SAMPLE_LEAD)
    await queue.enqueue_job("analyze_lead_job", SAMPLE_LEAD, _defer_by=60)
    stats = (await client.get("/jobs/stats")).json()
    assert (stats["queued"], stats["scheduled_retries"]) == (1, 1)


async def test_stats_reveal_no_job_details(
    client: AsyncClient, queue: ArqRedis, fake_llm, monkeypatch: pytest.MonkeyPatch
) -> None:
    from config import get_settings

    monkeypatch.setenv("API_KEYS", f"n8n:{N8N_KEY},demo:{DEMO_KEY}")
    get_settings.cache_clear()
    job_id = (await client.post("/jobs/analyze-lead", json=SAMPLE_LEAD, headers={"X-API-Key": N8N_KEY})).json()[
        "job_id"
    ]
    await run_worker(queue)

    assert (await client.get("/jobs/stats")).status_code == 401
    other = await client.get("/jobs/stats", headers={"X-API-Key": DEMO_KEY})
    assert other.json()["complete"] == 1
    for secret in (job_id, SAMPLE_LEAD["email"], SAMPLE_LEAD["message"], "n8n"):
        assert secret not in other.text


async def test_stats_without_redis_503(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from config import get_settings

    monkeypatch.setattr(jobs, "_pool", None)
    monkeypatch.setenv("REDIS_URL", "")
    get_settings.cache_clear()
    resp = await client.get("/jobs/stats")
    assert (resp.status_code, resp.json()["error_code"]) == (503, "job_queue_unavailable")


def test_percentile() -> None:
    assert jobs._percentile([3.0, 1.0, 2.0], 0.5) == 2.0
    assert jobs._percentile([1.0], 0.95) == 1.0
    assert jobs._percentile([float(i) for i in range(1, 21)], 0.95) == 19.0


def test_retry_backoff() -> None:
    assert [jobs.retry_delay_seconds(t) for t in (1, 2, 3, 4, 10)] == [5, 10, 20, 40, 60]


async def test_in_process_worker_disabled_without_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    from config import get_settings

    monkeypatch.setenv("REDIS_URL", "")
    get_settings.cache_clear()
    assert await jobs.start_in_process_worker() is None
    await jobs.stop_in_process_worker(None)  # no-op


async def test_in_process_worker_starts_and_stops(queue: ArqRedis) -> None:
    running = await jobs.start_in_process_worker()
    assert running is not None and not running.task.done()
    await jobs.stop_in_process_worker(running)
    assert running.task.done()
