# Guardrails & fallbacks — done

App: `~/Projects/lead-agent`

## What was added
- **`errors.py`** — custom exceptions + FastAPI handlers → `{ "detail", "error_code" }` for n8n  
- **`guardrails.py`** — sanitize message/email input; block prompt-injection patterns; max length  
- **RAG scores** — low confidence or Chroma failure → still respond with `priority=High`, `needs_human_review=true`  
- **LLM timeout** → HTTP **502** so n8n can retry  
- Config: `rag_score_threshold`, `openai_timeout_seconds`

## Response extras
- `needs_human_review: bool`  
- `fallback_reason: str | null`

## Quick examples
| Situation | Result |
|-----------|--------|
| Injection / bad input | **400** + `error_code` |
| Unhandled crash | **500** safe message |
| RAG down / low confidence | **200**, `priority=High`, `needs_human_review=true` |
| OpenAI timeout | **502** |
