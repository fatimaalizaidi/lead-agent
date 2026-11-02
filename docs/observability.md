# Observability

App: `~/Projects/lead-agent`

## What you get
- **loguru** structured logs → stderr + rotating `logs/app.log`
- **`X-Request-ID`** middleware (accept from n8n or auto-generate; echo on response)
- Per request on **`/analyze-lead`**, **`/agent-action`**, and **`/multi-agent`**:
  - `request_id`, `path`, `total_ms`
  - aggregated `llm_latency_ms` + `llm_calls`
  - token usage when OpenAI returns it (`prompt_tokens` / `completion_tokens` / `total_tokens`)
  - `error_code` on failures (also logged on the dedicated `error` line)
- `/analyze-lead` also logs RAG: Chroma latency, hit count, best score, low-confidence / unavailable
- Never logged: API keys, full prompts, message bodies, raw upstream payloads

## Formats

| Mode | How to enable | Shape |
|------|---------------|--------|
| **Human** (default) | unset / `LOG_JSON=false` | `time \| LEVEL \| request_id=… \| module:func:line - message` |
| **JSON** | `LOG_JSON=true` (also `1` / `yes` / `on`) | One flat JSON object per line (collector-friendly) |

Optional: `LOG_LEVEL=DEBUG` (default `INFO`).

### Sample human line (`request_end`)
```text
request_end path=/agent-action status=200 total_ms=870.1 llm_latency_ms=810.5 llm_calls=2 prompt_tokens=420 completion_tokens=55 total_tokens=475 needs_human_review=- error_code=-
```

### Sample JSON object
```json
{
  "timestamp": "2026-11-02T13:00:00.123Z",
  "level": "INFO",
  "request_id": "demo-n8n-001",
  "event": "request_end",
  "path": "/multi-agent",
  "status": 200,
  "total_ms": 1200.5,
  "llm_latency_ms": 1100.0,
  "llm_calls": 3,
  "prompt_tokens": 900,
  "completion_tokens": 120,
  "total_tokens": 1020,
  "needs_human_review": "-",
  "error_code": "-",
  "message": "request_end path=/multi-agent status=200 ..."
}
```

## Event reference

| Event | Fields |
|-------|--------|
| `request_start` | path, email_domain (domain only) |
| `rag` | chroma_latency_ms, hit_count, best_score, low_confidence, rag_unavailable |
| `llm` | model, llm_latency_ms, prompt_tokens, completion_tokens, total_tokens (per model call) |
| `request_end` | path, status, total_ms, llm_latency_ms (sum), llm_calls, tokens (sum when available), needs_human_review, error_code |
| `error` | error_code, status |

Multi-turn agent / multi-agent paths emit one `llm` line **per** OpenAI call; `request_end` sums latency and tokens for the whole request.

## Try
```bash
# Human logs (default)
uvicorn main:app --reload --port 8000

# JSON logs for collectors
LOG_JSON=true uvicorn main:app --port 8000

curl -s -D - -X POST "http://127.0.0.1:8000/analyze-lead" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: demo-n8n-001" \
  -d '{"message":"Do you offer annual discounts?","email":"buyer@acme.com"}'
```
Watch the terminal or `logs/app.log` for matching `request_id=demo-n8n-001`.
