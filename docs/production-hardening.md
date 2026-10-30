# Production hardening — done

App: `~/Projects/lead-agent`

## 1. Environment security (`pydantic-settings`)
- New `config.py`: `Settings(BaseSettings)` loads `OPENAI_API_KEY` from environment or `.env`
- Optional: `OPENAI_MODEL`, `RAG_K`
- `main.py` / `rag.py` use `get_settings()` — no raw `os.environ` for the key
- Key is never logged or returned in API responses

## 2. Async performance
- `/health` and `/analyze-lead` are `async def`
- OpenAI chat via `AsyncOpenAI` + `await` (event loop free while waiting on the network)
- Sync Chroma/LangChain retrieval wrapped in `asyncio.to_thread(...)` so embeddings/search don’t block other requests
- Same Pydantic strict JSON + `knowledge_sources` / RAG behavior

## Why this helps n8n
Many concurrent webhook POSTs can be in flight: while one request awaits OpenAI, others still progress instead of queuing behind a single blocking call.

## Restart
```bash
cd ~/Projects/lead-agent
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
