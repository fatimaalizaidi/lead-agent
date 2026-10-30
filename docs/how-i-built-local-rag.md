# How I built local RAG for the lead analyzer

Student-facing notes for explaining the Memory & RAG upgrade on `~/Projects/lead-agent`.

## What problem this solves

Without RAG, the model only uses its general training knowledge. With RAG, each customer message is **matched against private company articles** first, then those facts are passed into the prompt so the suggested reply can use **our** pricing, SLA, support hours, etc.

## Big picture (request flow)

1. Client calls `POST /analyze-lead` with `{ "message", "email" }`.
2. FastAPI validates the body with Pydantic (`AnalyzeLeadRequest`).
3. **RAG:** embed the message → search ChromaDB → get top matching KB chunks.
4. Build a user prompt that includes the lead text **plus** retrieved company knowledge.
5. Call OpenAI with **strict JSON schema** (still Pydantic-validated).
6. Return JSON: `email`, `priority`, `suggested_response`, `rationale`, **`knowledge_sources`** (titles of articles used).

```text
message → embeddings → Chroma similarity search → prompt + KB context
                                                    ↓
                                              OpenAI (strict JSON)
                                                    ↓
                                         Pydantic validate → response
```

## What I added (files)

### New: `rag.py` (the RAG engine)

| Piece | Role |
|--------|------|
| `get_vectorstore()` | Opens or builds the Chroma collection `company_kb` under `./data/chroma` |
| `_kb_fingerprint()` | Hash of each `knowledge/*.md` (name, mtime, size) so we know when to re-index |
| `_build_index()` | Load markdown → split into chunks → OpenAI embeddings → store in Chroma |
| `retrieve_context(query, k=3)` | Similarity search; returns `[{title, content}, ...]` |
| `format_context_for_prompt(...)` | Turns hits into text for the LLM |
| `knowledge_source_titles(...)` | List of article titles for the API response |

**Indexing (lazy):** first request (or when the fingerprint is stale / store empty) rebuilds the index. Embeddings use OpenAI’s embedding API (default model used by `OpenAIEmbeddings`, e.g. `text-embedding-ada-002`) with the same `OPENAI_API_KEY`.

**Persistence:** vectors live in `./data/chroma` (local disk, not Pinecone). That folder is gitignored so we don’t commit the database.

### New: `knowledge/*.md` (company knowledge base)

Sample private domain articles (fictional company docs):

1. **Pricing Plans** — `pricing.md`
2. **Service Level Agreement (SLA)** — `sla.md`
3. **Enterprise Onboarding** — `enterprise-onboarding.md`
4. **Support Hours** — `support-hours.md`
5. **Refund Policy** — `refund-policy.md`

Each file is a short markdown article with a `# Title`. To add real company data later: drop another `.md` in `knowledge/`; the fingerprint changes and the index rebuilds.

### Changed: `main.py` (wire RAG into the existing API)

- Still `POST /analyze-lead` and the same strict JSON / Pydantic path.
- Before calling OpenAI, call retrieve on `body.message`.
- Inject a block like: `COMPANY KNOWLEDGE (retrieved; use when relevant):` + chunks into the **user** message.
- System prompt still asks for structured lead analysis (priority + short suggested reply + rationale).
- Extended response model with `knowledge_sources: list[str]` — filled **on the server** from retrieval titles (not invented by the model), so citations stay honest.

### Changed: `requirements.txt` (RAG stack)

- `chromadb` — local vector database  
- `langchain` / `langchain-core` — orchestration  
- `langchain-openai` — OpenAI embeddings (and LangChain OpenAI helpers)  
- `langchain-chroma` — Chroma integration  
- `langchain-text-splitters` — chunk markdown before embedding  

(Plus existing FastAPI, Pydantic, uvicorn, openai.)

### Other supporting updates

- `.gitignore` — ignore `data/`, `data/chroma/`, `.venv`, `.env`, etc.
- `README.md` — how RAG works and how to add articles
- Docker files updated so `rag.py` + `knowledge/` ship with the app

## Why ChromaDB (not Pinecone)

Chroma runs **locally** in the project folder — good for demos/class: no vector-DB cloud account, data stays on the machine. Pinecone would be similar conceptually (embed → store → query) but hosted remotely.

## What stayed the same (important for the teacher)

- Endpoint path: `POST /analyze-lead`
- Request validation: non-empty `message` + valid `email`
- OpenAI **strict JSON** via `response_format` + `LeadAnalysisResult.model_validate`
- API key only from environment / `.env` (never in code)

## How to demo it

1. Start the server from `~/Projects/lead-agent` with `OPENAI_API_KEY` set.
2. Ask something that appears in the KB, e.g. Growth plan pricing.
3. Show that `knowledge_sources` includes **Pricing Plans** and the `suggested_response` uses those facts.

## One-sentence summary for oral defense

“I added a local RAG layer: company markdown is embedded into ChromaDB with LangChain; each lead message retrieves the top matching articles, those are injected into the prompt, and the API still returns Pydantic-validated strict JSON including which knowledge sources were used.”
