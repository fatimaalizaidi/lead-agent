"""Background job queue for the AI endpoints (arq on Redis).

``POST /jobs/analyze-lead`` and ``POST /jobs/multi-agent`` enqueue the same work as the
synchronous endpoints and return ``202`` with a job id at once; ``GET /jobs/{id}`` reports
``queued`` → ``in_progress`` → ``complete`` / ``failed`` and returns the result. During a traffic
spike, requests wait in Redis instead of holding HTTP connections open on a 0.1-CPU server, and at
most ``JOB_MAX_CONCURRENCY`` OpenAI calls run at once.

- **Retries:** OpenAI rate limits (429) and upstream errors (502) are retried with backoff, up to
  ``JOB_MAX_TRIES``. Other errors (bad input, auth) fail at once.
- **Ownership:** each job records the API key id that created it; other keys get 404.
- **Where it runs:** ``JOB_WORKER_IN_PROCESS=true`` (default) starts the worker inside the API
  process, so one free Render service is enough. At scale, set it to false and run
  ``arq jobs.WorkerSettings`` as a separate service (docker-compose.yml does this).

See docs/job-queue.md.
"""

# No `from __future__ import annotations`: see the note in security.py.
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from arq import Retry, create_pool
from arq.connections import ArqRedis, RedisSettings
from arq.jobs import Job, JobStatus
from arq.worker import Worker
from loguru import logger
from pydantic import BaseModel

from config import get_settings
from errors import AppError, UpstreamError
from observability import _request_id_var, get_request_id

# HTTP statuses worth retrying: OpenAI rate limit and transient upstream failures.
RETRYABLE_STATUSES = {429, 502}


class QueueUnavailableError(AppError):
    status_code = 503
    error_code = "job_queue_unavailable"


class JobNotFoundError(AppError):
    status_code = 404
    error_code = "job_not_found"


class JobAccepted(BaseModel):
    job_id: str
    status: str = "queued"
    status_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    kind: str | None = None
    status: str  # queued | in_progress | complete | failed
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    tries: int | None = None
    enqueued_at: datetime | None = None
    finished_at: datetime | None = None


# ---- worker side -------------------------------------------------------------------------------


def retry_delay_seconds(job_try: int) -> int:
    """Backoff before the next try: 5 s, 10 s, 20 s, … (capped at 60 s)."""
    return min(60, 5 * 2 ** (job_try - 1))


async def _run(ctx: dict[str, Any], request_id: str | None, call) -> dict[str, Any]:
    """Run one job body; map AppErrors to a stored error and retry the transient ones."""
    token = _request_id_var.set(request_id)
    try:
        response = await call()
        return {"ok": True, "result": response.model_dump(mode="json")}
    except AppError as exc:
        retryable = isinstance(exc, UpstreamError) and exc.status_code in RETRYABLE_STATUSES
        if retryable and ctx["job_try"] < get_settings().job_max_tries:
            delay = retry_delay_seconds(ctx["job_try"])
            logger.warning("job retry error_code={} try={} in={}s", exc.error_code, ctx["job_try"], delay)
            raise Retry(defer=delay) from exc
        return {
            "ok": False,
            "error": {"status_code": exc.status_code, "error_code": exc.error_code, "detail": exc.detail},
        }
    finally:
        _request_id_var.reset(token)


async def analyze_lead_job(
    ctx: dict[str, Any], payload: dict[str, Any], owner: str | None = None, request_id: str | None = None
) -> dict[str, Any]:
    from main import AnalyzeLeadRequest, run_analyze_lead

    return await _run(ctx, request_id, lambda: run_analyze_lead(AnalyzeLeadRequest.model_validate(payload)))


async def multi_agent_job(
    ctx: dict[str, Any], payload: dict[str, Any], owner: str | None = None, request_id: str | None = None
) -> dict[str, Any]:
    from main import run_multi_agent_request
    from multiagent import MultiAgentRequest

    return await _run(
        ctx, request_id, lambda: run_multi_agent_request(MultiAgentRequest.model_validate(payload))
    )


JOB_FUNCTIONS = [analyze_lead_job, multi_agent_job]
JOB_KINDS = {"analyze_lead_job": "analyze-lead", "multi_agent_job": "multi-agent"}


def _redis_settings() -> RedisSettings:
    url = (get_settings().redis_url or "").strip()
    redis_settings = RedisSettings.from_dsn(url or "redis://localhost:6379/0")
    # Fail fast (503) instead of arq's default 5 retries when Redis is down.
    redis_settings.conn_retries = 1
    redis_settings.conn_timeout = 3
    return redis_settings


def _worker_options() -> dict[str, Any]:
    settings = get_settings()
    return {
        "functions": JOB_FUNCTIONS,
        "max_jobs": settings.job_max_concurrency,
        "max_tries": settings.job_max_tries,
        "job_timeout": settings.job_timeout_seconds,
        "keep_result": settings.job_result_ttl_seconds,
    }


class WorkerSettings:
    """Separate worker process: ``arq jobs.WorkerSettings``."""

    functions = JOB_FUNCTIONS
    redis_settings = _redis_settings()
    max_jobs = get_settings().job_max_concurrency
    max_tries = get_settings().job_max_tries
    job_timeout = get_settings().job_timeout_seconds
    keep_result = get_settings().job_result_ttl_seconds


@dataclass
class InProcessWorker:
    worker: Worker
    task: asyncio.Task


async def start_in_process_worker() -> InProcessWorker | None:
    """Run the worker inside the API process (free plan: one service). None when disabled."""
    settings = get_settings()
    if not settings.job_worker_in_process or not (settings.redis_url or "").strip():
        if not (settings.redis_url or "").strip():
            logger.warning("REDIS_URL is empty: /jobs endpoints are off (503).")
        return None
    try:
        pool = await get_pool()
    except QueueUnavailableError:
        logger.warning("Job worker not started: Redis unreachable. /jobs endpoints will return 503.")
        return None
    worker = Worker(redis_pool=pool, handle_signals=False, **_worker_options())
    task = asyncio.create_task(worker.async_run())
    logger.info("job worker started in-process (max_jobs={})", settings.job_max_concurrency)
    return InProcessWorker(worker, task)


async def stop_in_process_worker(running: InProcessWorker | None) -> None:
    if running is None:
        return
    await running.worker.close()
    running.task.cancel()
    try:
        await running.task
    except (asyncio.CancelledError, Exception):
        pass


# ---- API side ----------------------------------------------------------------------------------

_pool: ArqRedis | None = None


async def get_pool() -> ArqRedis:
    """Shared arq Redis pool; 503 when REDIS_URL is unset or Redis can't be reached."""
    global _pool
    if _pool is not None:
        return _pool
    if not (get_settings().redis_url or "").strip():
        raise QueueUnavailableError("Background jobs need REDIS_URL; use the synchronous endpoint instead.")
    try:
        _pool = await create_pool(_redis_settings(), retry=1)
    except Exception as exc:
        logger.warning("job queue: Redis unreachable ({})", type(exc).__name__)
        raise QueueUnavailableError("The job queue is unavailable; retry or use the synchronous endpoint.") from exc
    return _pool


async def enqueue(function: str, payload: dict[str, Any], owner: str | None) -> JobAccepted:
    pool = await get_pool()
    job = await pool.enqueue_job(function, payload, owner=owner, request_id=get_request_id())
    if job is None:  # only happens with a duplicate custom job id
        raise QueueUnavailableError("Could not enqueue the job.")
    return JobAccepted(job_id=job.job_id, status_url=f"/jobs/{job.job_id}")


async def job_status(job_id: str, owner: str | None) -> JobStatusResponse:
    """Status and result of a job the caller owns; 404 for unknown jobs and other keys' jobs."""
    pool = await get_pool()
    job = Job(job_id, pool)
    status = await job.status()
    info = await job.info() if status != JobStatus.not_found else None
    if info is None or info.kwargs.get("owner") != owner:
        raise JobNotFoundError("No such job (or it expired).")

    response = JobStatusResponse(
        job_id=job_id,
        kind=JOB_KINDS.get(info.function, info.function),
        status="queued",
        tries=info.job_try,
        enqueued_at=info.enqueue_time,
    )
    if status == JobStatus.in_progress:
        response.status = "in_progress"
    elif status == JobStatus.complete:
        result = await job.result_info()
        response.finished_at = result.finish_time if result else None
        outcome = result.result if result and result.success else None
        if isinstance(outcome, dict) and outcome.get("ok"):
            response.status = "complete"
            response.result = outcome["result"]
        else:
            response.status = "failed"
            response.error = (
                outcome.get("error")
                if isinstance(outcome, dict)
                else {"status_code": 500, "error_code": "job_crashed", "detail": "The job failed unexpectedly."}
            )
    return response
