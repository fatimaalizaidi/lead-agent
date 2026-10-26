# Lead Analyzer

FastAPI + Pydantic service that uses OpenAI to classify inbound leads as **High**, **Medium**, or **Low** and draft a short suggested reply. Before calling the model, it retrieves relevant company KB articles from a local ChromaDB store (LangChain) and injects them into the prompt.

## Setup

```bash
cd ~/Projects/lead-agent   # or docs/lead-analyzer in Context
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## API key

Copy `.env.example` → `.env` and set your key (never commit `.env`). The same key is used for chat completions **and** embeddings:

```bash
cp .env.example .env
# edit .env: OPENAI_API_KEY=sk-...
```

`config.Settings` (pydantic-settings) loads `OPENAI_API_KEY` from the environment or `.env`. You can still `export OPENAI_API_KEY=...` if you prefer. Optional: `OPENAI_MODEL` (default `gpt-4o-mini`), `RAG_K` (default `3`), `RAG_SCORE_THRESHOLD` (default `1.2`, Chroma L2 distance — lower is better), `OPENAI_TIMEOUT_SECONDS` (default `30`). The key is never logged or returned in API responses.

## Run

Settings loads `.env` automatically, so plain uvicorn is enough:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

You can still pass `--env-file .env` to uvicorn if you want the process env populated before import; both approaches work with an existing `.env` on disk.

Open interactive docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Sample request

```bash
curl -s -X POST "http://127.0.0.1:8000/analyze-lead" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "We need a demo this week for a 200-seat rollout. Budget is approved.",
    "email": "alex@acme.com"
  }'
```

Pricing / KB-oriented example (should surface `knowledge_sources` such as Pricing Plans):

```bash
curl -s -X POST "http://127.0.0.1:8000/analyze-lead" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What does Growth plan cost per month, and do you offer annual discounts?",
    "email": "buyer@acme.com"
  }'
```

Example response shape:

```json
{
  "email": "buyer@acme.com",
  "priority": "Medium",
  "suggested_response": "Growth is $149/month; annual billing is 15% off...",
  "rationale": "Pricing question with clear commercial interest.",
  "knowledge_sources": ["Pricing Plans"],
  "needs_human_review": false,
  "fallback_reason": null
}
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/analyze-lead` | Analyze a lead message (with RAG context) |

### `POST /analyze-lead`

**Request**

| Field | Type | Rules |
|-------|------|--------|
| `message` | string | Required, non-empty; max 6000 chars; sanitized (see Guardrails) |
| `email` | string | Required, valid email; normalized |

**Response**

| Field | Type | Notes |
|-------|------|--------|
| `email` | string | Echo of the input email |
| `priority` | string | Exactly `High`, `Medium`, or `Low` |
| `suggested_response` | string | Brief draft reply (may cite KB facts) |
| `rationale` | string | Short reason for the priority |
| `knowledge_sources` | string[] | Titles of KB articles retrieved for this request |
| `needs_human_review` | bool | `true` when a soft fallback forced human review |
| `fallback_reason` | string \| null | e.g. `rag_unavailable`, `low_rag_confidence` |

Uses model `gpt-4o-mini` (override with `OPENAI_MODEL`) with strict JSON-schema `response_format` so `priority` stays in that enum. RAG context is prompt-only; it is **not** part of the OpenAI JSON schema.

Errors always return JSON `{"detail": "...", "error_code": "..."}` (no stack traces or secrets).

## Production hardening

### Config (`pydantic-settings`)

- `config.py` defines `Settings(BaseSettings)` with `openai_api_key` (from `OPENAI_API_KEY`), plus optional `openai_model`, `rag_k`, `rag_score_threshold`, and `openai_timeout_seconds`.
- `get_settings()` is cached; both `main.py` and `rag.py` use it instead of scattered `os.environ.get`.
- `.env` is gitignored; `.env.example` documents the variables with no secrets.

### Async FastAPI + n8n concurrency

- `/health` and `/analyze-lead` are `async def`.
- Chat uses `AsyncOpenAI` so `await` yields the event loop while OpenAI responds — many concurrent POSTs from n8n can stay in flight instead of one slow sync call blocking everyone else.
- Chroma / LangChain embeddings stay sync; `retrieve_context` runs via `asyncio.to_thread(...)` so RAG does not block the loop. (Async LangChain embeddings are possible but not worth the extra surface here.)

## RAG / knowledge base

On first request (or when `./data/chroma` is empty / KB files change), `rag.py` loads `knowledge/*.md`, splits lightly, embeds with OpenAI, and persists vectors under `./data/chroma` (gitignored).

1. **Add an article** — drop a new `.md` file in `knowledge/` with a clear `# Title` as the first heading. One file ≈ one article.
2. **Re-index** — change detection uses a fingerprint of file names/mtimes/sizes. After editing KB files, the next request rebuilds the collection automatically. To force a clean rebuild, delete `./data/chroma` and restart uvicorn (or hit `/analyze-lead` again).
3. **Restart uvicorn** after changing `main.py` / `rag.py` / `config.py` so code reloads (with `--reload`, file saves are enough).
4. Embeddings and chat both require `OPENAI_API_KEY`; keys are never hardcoded or logged.

Tested package set targets Python 3.13 (`chromadb==0.6.3` + LangChain 0.3.x). If Chroma fails to install on your platform, use Python 3.11/3.12 or pin an alternate Chroma build.

## Guardrails & fallbacks

### Input guardrails (`guardrails.py`)

Applied in `AnalyzeLeadRequest` validators **before** RAG/LLM:

- Strip control characters and normalize whitespace
- Reject messages longer than **6000** characters (`error_code: message_too_long`)
- Block obvious prompt-injection / jailbreak patterns (`error_code: prompt_injection_blocked`)
- Normalize email (lowercase); light format sanity checks on top of `EmailStr`

Bad input → **HTTP 400** with `{ "detail": "...", "error_code": "..." }`.

### Exception handlers (`errors.py`)

| Case | HTTP | `error_code` (examples) |
|------|------|-------------------------|
| Validation / guardrails / `BadRequestError` | **400** | `validation_error`, `prompt_injection_blocked`, `message_too_long` |
| OpenAI auth | **401** | `openai_auth_failed` |
| OpenAI rate limit | **429** | `openai_rate_limit` |
| OpenAI timeout / connection / empty / bad JSON | **502** | `openai_timeout`, `openai_unreachable`, … |
| Unhandled server bug | **500** | `internal_error` (generic message only) |

Request body validation uses **400** (not 422) so n8n can treat all client errors the same way.

### Soft vs hard fallbacks

1. **Chroma / RAG failure** → still call LLM **without** KB; HTTP **200**, `priority=High`, `needs_human_review=true`, `fallback_reason="rag_unavailable"`, `knowledge_sources=[]`, rationale notes manual review.
2. **Low RAG confidence** → Chroma `similarity_search_with_score` returns L2 **distance** (lower is better). If best score **>** `RAG_SCORE_THRESHOLD` (default `1.2`), still call LLM with retrieved chunks but override `priority=High`, `needs_human_review=true`, `fallback_reason="low_rag_confidence"`.
3. **LLM timeout / failure** → HTTP **502** so n8n can retry (no fake 200 body for hard model failures).

### Example statuses

- **400** — `"Ignore previous instructions and …"` in `message`, or missing/oversized fields  
- **500** — unexpected bug (safe generic detail)  
- **200 + needs_human_review** — RAG down or low confidence; analysis still returned with High + review flags  
- **502** — OpenAI timed out or returned invalid structured output  

Restart uvicorn after changing `main.py` / `rag.py` / `config.py` / `guardrails.py` / `errors.py` / `observability.py` (with `--reload`, saving those files is enough).

## Observability

Structured logging via **loguru** (`observability.py`). Logs go to **stderr** and rotating **`logs/app.log`** (`logs/` is gitignored). Every line includes `request_id=...` from context.

### `X-Request-ID`

- Middleware reads `X-Request-ID` if the client (e.g. n8n) sends it; otherwise generates a UUID.
- The same id is bound into log context for the request and **echoed on the response** as `X-Request-ID` (header only — response body contract unchanged).

```bash
curl -s -D - -X POST "http://127.0.0.1:8000/analyze-lead" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: demo-n8n-001" \
  -d '{"message":"Need a demo this week for 200 seats.","email":"alex@acme.com"}' \
  -o /dev/null
# Look for: X-Request-ID: demo-n8n-001
```

### What is logged (structured fields)

| Event | Fields |
|-------|--------|
| `request_start` | path, email_domain (domain only — not full email/message) |
| `rag` | chroma_latency_ms, hit_count, best_score, low_confidence, rag_unavailable |
| `llm` | model, llm_latency_ms, prompt_tokens, completion_tokens, total_tokens |
| `request_end` | status, total_ms, needs_human_review |
| `error` | error_code, status (no stack secrets / API keys / prompts) |

Never logged: `OPENAI_API_KEY`, full prompts, message bodies, or raw upstream error payloads.

### Sample log lines (one successful `/analyze-lead`)

```text
... | INFO | request_id=demo-n8n-001 | ... - request_start path=/analyze-lead email_domain=acme.com
... | INFO | request_id=demo-n8n-001 | ... - rag chroma_latency_ms=42.3 hit_count=2 best_score=0.8123 low_confidence=False rag_unavailable=False
... | INFO | request_id=demo-n8n-001 | ... - llm model=gpt-4o-mini llm_latency_ms=810.5 prompt_tokens=420 completion_tokens=55 total_tokens=475
... | INFO | request_id=demo-n8n-001 | ... - request_end status=200 total_ms=870.1 needs_human_review=False
```

Restart uvicorn after changing `observability.py` / `main.py` / `errors.py` (with `--reload`, saving those files is enough). Install the new dependency once: `pip install -r requirements.txt`.

## Deploy / Going live

Expose the API so **n8n** can `POST` JSON to `/analyze-lead`. Full student walkthrough: **[DEPLOY.md](./DEPLOY.md)**. Do not commit `.env` or bake keys into the image — pass `OPENAI_API_KEY` at runtime.

### Path A — ngrok (quick tunnel)

1. Start the app: `uvicorn main:app --host 0.0.0.0 --port 8000` **or** `docker compose up --build`.
2. In another terminal: `ngrok http 8000` (you configure ngrok auth yourself).
3. In n8n **HTTP Request**: method `POST`, URL `https://YOUR-NGROK-HOST/analyze-lead`, body JSON `{"message":"...","email":"..."}`.

### Path B — Render / Railway (Docker host)

1. Push this repo (without secrets). Deploy with the included **`Dockerfile`** + **`render.yaml`** (Render Blueprint → Web Service from Docker), or point Railway at the same Dockerfile.
2. Set secret **`OPENAI_API_KEY`** in the host dashboard (optional: `OPENAI_MODEL`, etc.).
3. Copy the public HTTPS base URL into n8n: `https://YOUR-SERVICE/analyze-lead` (POST JSON).

Local Docker without a cloud host: `docker compose up --build` (uses `env_file: .env`, port `8000`). Health check: `GET /health`.

## Refine the prompt

Edit `SYSTEM_PROMPT` near the top of `main.py` to change how leads are classified and how long the draft reply is.

- A shorter prompt plus `max_tokens` (and `temperature=0`) keeps completions smaller and usually faster.
- LLM JSON is parsed into `LeadAnalysisResult` (Pydantic). Bad or out-of-range fields raise `ValidationError` and the API returns HTTP 502 instead of a half-valid response.
- The OpenAI `json_schema` is generated from that same model so schema and validators stay in sync.

Restart uvicorn after changing `main.py` so the new prompt and settings load.
