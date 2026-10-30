# Plan: stricter Pydantic + JSON (lead-agent)

**Status:** waiting for your approval — no code edits yet  
**Primary app path:** `~/Projects/lead-agent`  
**Also sync after approve:** Project Context `docs/lead-analyzer/`

## What exists today

| Piece | Current state |
|---|---|
| Endpoint | `POST /analyze-lead` (+ `GET /health`) |
| Request | `AnalyzeLeadRequest`: `message` (non-empty str), `email` (EmailStr) |
| Response | `email`, `priority` (`High`\|`Medium`\|`Low`), `suggested_response`, `rationale` |
| LLM | OpenAI `gpt-4o-mini`, strict `json_schema`, validated via `LeadAnalysisResult` |
| Secrets | `OPENAI_API_KEY` from env / `.env` only |

You already have Pydantic request/response models and structured JSON. The gap vs your new requirements is mainly: add **`sentiment`**, switch **`priority`** to lowercase (`low`\|`medium`\|`high`), and confirm request field naming.

## Files to change (and why)

1. **`~/Projects/lead-agent/main.py`** (required)  
   - Add `Sentiment` enum: `positive` \| `neutral` \| `negative`  
   - Change `Priority` values to `low` \| `medium` \| `high`  
   - Extend `LeadAnalysisResult` + `AnalyzeLeadResponse` with `sentiment`  
   - Keep FastAPI `response_model=AnalyzeLeadResponse` so invalid response shapes are rejected  
   - Keep OpenAI `response_format` strict JSON schema derived from the Pydantic LLM model  
   - Update `SYSTEM_PROMPT` so the model fills the new fields  
   - Keep `POST /analyze-lead` and `GET /health`  
   - No API keys in code  

2. **`~/Projects/lead-agent/README.md`** (required)  
   - Update sample request/response curl and field docs to match the new schema  

3. **Project Context `docs/lead-analyzer/`** (sync after approve)  
   - Same `main.py` + `README.md` so Library stays aligned  

**Not changing:** `requirements.txt` (already has FastAPI/Pydantic/OpenAI), `.env` (secrets stay local).

## Proposed API shape (after change)

**Request** `POST /analyze-lead` — keep path:

```json
{
  "message": "non-empty lead text",
  "email": "user@example.com"
}
```

**Response:**

```json
{
  "email": "user@example.com",
  "sentiment": "positive",
  "priority": "high",
  "suggested_response": "...",
  "rationale": "..."
}
```

## Open choices (need your call)

1. **Request text field name**  
   - **A (recommended):** keep `message` (no break for your current `/docs` trials)  
   - **B:** rename to `question`  
   - **C:** accept both (`question` alias → same field)

2. **JSON key for the reply**  
   - **A (recommended):** `suggested_response` (valid Python + OpenAPI-friendly)  
   - **B:** serialize as `suggested-response` via Pydantic alias (works, slightly awkward in Python)

3. **Priority casing**  
   - Your spec wants `low`/`medium`/`high` — this **breaks** current clients expecting `High`/`Medium`/`Low`. Confirm that’s OK.

## Out of scope unless you ask

- New endpoints  
- Auth / rate limiting beyond current OpenAI error mapping  
- Committing secrets or logging the API key  
