# Lead Agent — Project Context guides

User-facing notes copied from Cursor Project Context so Claude Desktop (and anyone cloning this repo) can study how the stack was built and operated.

| Guide | Description |
|-------|-------------|
| [agent-tools.md](./agent-tools.md) | Agent tool-calling (`POST /agent-action`) |
| [conversation-memory.md](./conversation-memory.md) | Redis multi-turn memory (`thread_id`) |
| [semantic-cache.md](./semantic-cache.md) | Exact + semantic answer cache on `/analyze-lead`, with guards and threshold tuning |
| [llm-judge.md](./llm-judge.md) | LLM-as-a-judge: grade `/analyze-lead` replies for groundedness, relevance, tone and next step, with a calibration check |
| [multi-agent-langgraph.md](./multi-agent-langgraph.md) | LangGraph supervisor (Sales/Pricing, Knowledge, Escalation) |
| [ci-cd.md](./ci-cd.md) | GitHub Actions CI (ruff, pytest, Docker smoke test) gating Render deploys |
| [deploy-going-live.md](./deploy-going-live.md) | Deploy / going live on Render (overview) |
| [lead-analyzer/DEPLOY.md](./lead-analyzer/DEPLOY.md) | Full Render deploy walkthrough for n8n |
| [docker-compose.md](./docker-compose.md) | Docker Compose for the FastAPI backend |
| [guardrails-fallbacks.md](./guardrails-fallbacks.md) | Guardrails and fallbacks |
| [how-i-built-local-rag.md](./how-i-built-local-rag.md) | How local RAG was built (student notes) |
| [n8n-priority-routing.md](./n8n-priority-routing.md) | n8n priority routing hub |
| [n8n-priority-routing-workflow.json](./n8n-priority-routing-workflow.json) | Importable n8n workflow JSON |
| [observability.md](./observability.md) | Request / RAG / LLM logging |
| [tracing.md](./tracing.md) | LangSmith tracing, per-request LLM cost, failed tool-call tracking |
| [production-hardening.md](./production-hardening.md) | Production hardening notes |
| [pydantic-strict-json-plan.md](./pydantic-strict-json-plan.md) | Stricter Pydantic + JSON plan |
| [rag-upgrade.md](./rag-upgrade.md) | ChromaDB + LangChain RAG upgrade |
| [hybrid-rag-reranking.md](./hybrid-rag-reranking.md) | Hybrid search (Chroma + BM25) and cross-encoder / Cohere re-ranking |
| [rag-evaluation.md](./rag-evaluation.md) | Ragas evaluation: faithfulness, answer relevancy, context recall/precision (`python -m eval.run_ragas`) |
| [demo.md](./demo.md) | Public demo page at `/demo`: key-free API, per-IP limits, daily OpenAI budget |
| [job-queue.md](./job-queue.md) | Background jobs: `POST /jobs/*` → 202, arq worker, retries, scaling |
| [fine-tuning.md](./fine-tuning.md) | LoRA fine-tuned priority classifier: data, training, ONNX export, eval gate, `POST /classify-priority` |
| [security.md](./security.md) | API keys, OAuth2/JWT tokens, key rotation and `slowapi` rate limiting |
| [testing.md](./testing.md) | Offline pytest + httpx API tests (mocked OpenAI / RAG / Redis / webhooks) |

App setup and API usage live in the [root README](../README.md).
