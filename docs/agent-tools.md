# Agent tool-calling (`POST /agent-action`)

The lead-agent FastAPI app exposes an autonomous agent endpoint that uses **OpenAI function/tool calling**. The model chooses among real Python tools, the server executes them, and the response includes tool args/results plus a short reply.

`POST /analyze-lead` is unchanged.

## Request

Same shape as `/analyze-lead`:

```json
{
  "email": "alex@acme.com",
  "message": "Can I get a quote for 12 Growth seats?"
}
```

Inputs are sanitized with the existing guardrails (length, injection patterns, email format).

## Response

```json
{
  "email": "alex@acme.com",
  "agent_reply": "Growth covers your 12 seats for $149/month ($1,519.80/year with annual billing).",
  "tools_called": [
    {
      "name": "calculate_custom_quote",
      "arguments": { "tier": "growth", "seats": 12 },
      "result": {
        "ok": true,
        "tier": "growth",
        "requested_tier": "growth",
        "seats": 12,
        "seat_limit": 15,
        "monthly_total_usd": 149.0,
        "annual_total_usd": 1519.8,
        "currency": "USD",
        "note": "Flat monthly price up to the seat limit; annual billing saves 15%. Not a binding offer."
      }
    }
  ],
  "iterations": 2
}
```

| Field | Meaning |
| --- | --- |
| `agent_reply` | Short natural-language reply after tools (or clarification) |
| `tools_called` | Ordered list of tool name / args / result |
| `iterations` | Model turns used (capped; typically 1–3) |

## Available tools

| Tool | Purpose |
| --- | --- |
| `calculate_custom_quote(tier, seats)` | Monthly quote from the price list in `knowledge/pricing.md`: `starter` $49/month up to 3 seats, `growth` $149/month up to 15 seats (flat, not per seat). Too many seats for a plan moves to the plan that fits; `enterprise` (or more than 15 seats) returns no price, because it is a custom quote |
| `check_calendar_availability()` | Stub available demo slots (fake UTC windows) |
| `escalate_to_human_support()` | Stub escalation ticket id + `queued` status |

The agent loop may call tools across multiple turns (max 5 iterations), then returns a final message.

## Run locally

```bash
cd ~/Projects/lead-agent
source .venv/bin/activate
uvicorn main:app --host 127.0.0.1 --port 8000
```

Requires `OPENAI_API_KEY` in `.env` (see `.env.example`).

## Example curls

### Quote intent

```bash
curl -sS http://127.0.0.1:8000/agent-action \
  -H 'Content-Type: application/json' \
  -d '{"email":"alex@acme.com","message":"We need a custom quote for 25 Growth seats."}'
```

### Calendar / demo intent

```bash
curl -sS http://127.0.0.1:8000/agent-action \
  -H 'Content-Type: application/json' \
  -d '{"email":"alex@acme.com","message":"What demo times are available this week?"}'
```

### Escalate intent

```bash
curl -sS http://127.0.0.1:8000/agent-action \
  -H 'Content-Type: application/json' \
  -d '{"email":"alex@acme.com","message":"This is urgent — please escalate me to a human support agent now."}'
```

### Health check

```bash
curl -sS http://127.0.0.1:8000/health
```

## Code map

- `tools.py` — tool schemas + stub implementations
- `agent.py` — OpenAI tool-calling loop + request/response models
- `main.py` — wires `POST /agent-action`
