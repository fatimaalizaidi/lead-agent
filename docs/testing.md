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
| Demo page | `tests/test_demo.py` | `/demo` page, key-free API, daily budget fallback, per-IP 429, guardrail, thread isolation |
| Classifier routing | `tests/test_routing.py` | Confident Low skips RAG + OpenAI, buyers and follow-ups still use the LLM, switch-off and missing-model fallback, `/routing/stats` counts and auth, threshold never routes a non-Low test lead |
| Answer cache | `tests/test_semantic_cache.py` | Exact and near-duplicate hits skip RAG + GPT; different numbers or classifier labels never share an answer; follow-ups and flagged answers not cached; Redis index survives a restart; cached demo clicks use no budget |
| Background jobs | `tests/test_jobs.py` | Enqueue → worker → result, retries with backoff, max tries, per-key privacy, 503 without Redis, `/jobs/stats` counts and privacy |
| LangSmith export | `tests/test_export_langsmith.py` | Parses wrapped-OpenAI runs, masks PII, skips errors / hidden inputs / other endpoints / follow-ups, dedupes, never leaks the test set, time split, private gitignored output |
| Input guardrails | `tests/test_guardrails.py` | Injection markers blocked; business text with "Instagram", "install", "Einstein", "CRM system:" allowed; zero false positives on the 1,464-lead dataset |
| Priority classifier | `tests/test_classifier.py` | Shipped ONNX model on clear cases, **quality gate** on the hand-written test set, `/classify-priority` auth / validation / 503, dataset disjoint from test, teacher labelling (fake OpenAI) |
| Auth + rate limits | `tests/test_security.py` | API keys, rotation, `/auth/token` JWTs (expired / tampered / revoked), 429s |

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

## Live test run (real server, real OpenAI)

The tests above mock OpenAI. To check the deployed system end to end, `eval/live_run.py` posts the 84 labelled, hand-written test leads (`finetune/data/test.jsonl`) to `POST /analyze-lead` one at a time and writes a report. The report covers:
- accuracy and High recall against the known labels;
- how many leads the classifier answered without OpenAI, and whether any of those were wrong;
- GPT's accuracy on the other leads;
- latency p50/p95 for each path, the human-review rate and errors;
- the estimated OpenAI cost.

```bash
export LEAD_AGENT_API_KEY=la_...           # or a LEAD_AGENT_API_KEY= line in .env; never printed
python -m eval.live_run                     # dry run: shows what would be sent and the max cost
python -m eval.live_run --yes --limit 12    # quick check, labels mixed evenly
python -m eval.live_run --yes               # all 84 leads, ~5 minutes, about $0.01
```

- The script calls the API directly, not the n8n webhook, so a run doesn't send Slack alerts or auto-reply emails.
- Requests are paced under the default 20/minute rate limit, and a 429 waits for `Retry-After`. The script wakes a sleeping Render service first.
- Reports go to `eval/results/live-<time>.md`, `.csv` and `.json` (gitignored).
- `--url http://127.0.0.1:8000 --pace 0` runs it against a local server.
- The leads are fictional. They do show up in LangSmith traces, but `finetune/export_langsmith.py` drops test-set leads, so they never leak into training.

## CI notes

- Run `pip install -r requirements-dev.txt && pytest` with no Redis / OpenAI credentials.
- Do not commit `.env` or real API keys. The test key in `conftest.py` is intentionally fake.
