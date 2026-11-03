# Testing (pytest + httpx)

Offline API / E2E tests for the Lead Analyzer FastAPI app. Suites **do not** call OpenAI, Chroma embeddings, Redis, Slack, or email webhooks.

## Install

```bash
cd ~/Projects/lead-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

`requirements-dev.txt` pulls in `requirements.txt` plus `pytest`, `pytest-asyncio`, and `fakeredis`. (`httpx` is already a runtime dependency.)

## Run

```bash
pytest
# or
pytest -q
pytest tests/test_health.py -v
```

No `.env` secrets are required. Tests set `OPENAI_API_KEY=sk-test-offline-key-not-real` in `tests/conftest.py` (placeholder only — never a real key).

## What is covered

| Endpoint | File | Notes |
|----------|------|--------|
| `GET /health` | `tests/test_health.py` | Liveness |
| `POST /analyze-lead` | `tests/test_analyze_lead.py` | Success, memory persist, RAG soft-fallback, validation |
| `POST /agent-action` | `tests/test_agent_action.py` | Tool loop (quote) + no-tools reply |
| `POST /multi-agent` | `tests/test_multi_agent.py` | Knowledge / Sales/Pricing / Escalation routes |

## What is mocked

| Dependency | How |
|------------|-----|
| **OpenAI** chat completions | `unittest.mock` patches `_get_async_client` in `main` / `agent` / `multiagent`; fake completions from `tests/openai_fakes.py` |
| **RAG / Chroma** | `retrieve_context` patched to return fixed chunks + score (no embeddings) |
| **Redis memory** | `fakeredis.FakeRedis` injected into `memory._redis_client` |
| **Slack / email webhooks** | Empty webhook URLs → stub path; when URLs are set in a test, `httpx.post` is patched so nothing leaves the process |

Local Python tools (`calculate_custom_quote`, calendar stubs, etc.) still run — they are pure in-process stubs with no network.

## Layout

```
tests/
  __init__.py
  conftest.py          # env, fakeredis, AsyncClient, RAG fixtures
  openai_fakes.py      # FakeCompletion helpers
  test_health.py
  test_analyze_lead.py
  test_agent_action.py
  test_multi_agent.py
pytest.ini             # asyncio_mode = auto
requirements-dev.txt
```

## CI notes

- Run `pip install -r requirements-dev.txt && pytest` with no Redis / OpenAI credentials.
- Do not commit `.env` or real API keys. The test key in `conftest.py` is intentionally fake.
