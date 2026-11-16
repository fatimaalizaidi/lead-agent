# Background job queue (arq + Redis)

`jobs.py` lets clients hand the slow AI work to a background worker instead of holding an HTTP request open while OpenAI answers.

```
client ──POST /jobs/analyze-lead──▶ API ──enqueue──▶ Redis ◀──poll── worker ──▶ OpenAI + RAG
   ◀── 202 {job_id, status_url} ──┘                                   │
client ──GET /jobs/{id}──────────▶ API ──read result──▶ Redis ◀──result┘
```

| Endpoint | Returns |
|---|---|
| `POST /jobs/analyze-lead` | `202` `{"job_id", "status": "queued", "status_url"}`. Same body as `/analyze-lead`. |
| `POST /jobs/multi-agent` | `202`, same shape. Same body as `/multi-agent`. |
| `GET /jobs/{job_id}` | `{"status": "queued" \| "in_progress" \| "complete" \| "failed", "result", "error", "tries", "enqueued_at", "finished_at"}`. `result` holds the same JSON the synchronous endpoint returns. |

The synchronous endpoints are unchanged. Use them when the caller can wait a few seconds, and the job endpoints when it can't, or during bursts.

## Why it helps under load

- **Spikes queue up instead of piling up.** Each synchronous request holds a connection and a slot on a 0.1-CPU server for 2–5 s. A job request returns in milliseconds, and the work waits in Redis.
- **Bounded concurrency.** A worker runs at most `JOB_MAX_CONCURRENCY` jobs at once (default 2), so a burst never sends 100 parallel calls to OpenAI and trips its rate limit.
- **Retries with backoff.** OpenAI rate limits (429) and upstream errors (502) are retried after 5 s, 10 s, 20 s…, up to `JOB_MAX_TRIES` (default 3). Bad input or auth errors fail at once; retrying can't fix them.
- **Scales out.** Add workers to drain the queue faster; the API doesn't change.

## Running it

It needs Redis (`REDIS_URL`), which also powers conversation memory. Without it, `/jobs/*` return `503 job_queue_unavailable` and everything else keeps working.

| Setup | How the worker runs |
|---|---|
| **Render free plan** | `JOB_WORKER_IN_PROCESS=true` (default): the API process starts the worker at startup, so one free service is enough. |
| **docker compose** | A separate `worker` service runs `arq jobs.WorkerSettings`; the API has `JOB_WORKER_IN_PROCESS=false`. Scale it with `docker compose up -d --scale worker=3`. |
| **Paid / larger deploys** | One web service plus N background-worker services running `arq jobs.WorkerSettings`. |

### Redis for Render (free)

1. Create a Redis-compatible instance on a free tier: **Render → New → Key Value**, free instance type, same region as the API (or Upstash's free tier). Check that the plan you pick doesn't need a card.
2. Copy its **internal** connection URL (`redis://red-…:6379`) into the API service's `REDIS_URL` environment variable, then save and deploy.
3. The startup log should then show `job worker started in-process`.

Free Redis instances have no persistence: a restart drops queued jobs and stored results. That's acceptable for a demo; use a persistent plan in production.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `REDIS_URL` | empty | Redis connection URL. Empty means the job endpoints return 503. |
| `JOB_WORKER_IN_PROCESS` | `true` | Run the worker inside the API process. |
| `JOB_MAX_CONCURRENCY` | `2` | Jobs at once per worker. |
| `JOB_MAX_TRIES` | `3` | Tries per job for retryable errors. |
| `JOB_TIMEOUT_SECONDS` | `120` | A job running longer is cancelled and marked failed. |
| `JOB_RESULT_TTL_SECONDS` | `3600` | How long a finished result stays readable. |
| `RATE_LIMIT_POLL` | `120/minute` | Per-key limit on `GET /jobs/{id}`. Enqueueing uses `RATE_LIMIT`, like the synchronous endpoints. |

## Security

- The job endpoints use the same API keys / JWTs and rate limits as the rest of the API ([security.md](./security.md)).
- Each job records the key id that created it. Another key gets `404` for it, the same answer as for a job that doesn't exist, so job ids can't be probed.
- Failed jobs store only `status_code`, `error_code` and the safe `detail`. Unexpected crashes are reported as `job_crashed`, with no stack trace or internal message.

## Example

```bash
curl -s -X POST "$URL/jobs/analyze-lead" -H "X-API-Key: la_..." -H "Content-Type: application/json" \
  -d '{"email":"buyer@acme.com","message":"Pricing for 20 seats?"}'
# {"job_id":"2129…","status":"queued","status_url":"/jobs/2129…"}

curl -s "$URL/jobs/2129…" -H "X-API-Key: la_..."
# {"status":"complete","result":{"priority":"Medium","suggested_response":"…",…},"tries":1,…}
```

In n8n, use **HTTP Request (POST /jobs/analyze-lead)** → **Wait** a few seconds → **HTTP Request (GET status_url)**, looping until `status` is `complete` or `failed`.

## Design notes

- **Why arq:** it's asyncio-native like the FastAPI app, so jobs reuse the same `async` code paths (`main.run_analyze_lead`, `main.run_multi_agent_request`) with no duplication, and it only needs Redis. Celery would add a second programming model and a heavier worker for the same job.
- **Request ids carry over:** the enqueueing request's `X-Request-ID` is stored with the job, and the worker's logs use it, so one id follows the work from the HTTP call to the OpenAI call.
- **Tested end to end** on fakeredis (`tests/test_jobs.py`) and checked by hand against a real Redis with the in-process worker.
