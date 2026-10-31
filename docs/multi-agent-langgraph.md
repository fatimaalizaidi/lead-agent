# Multi-agent system (LangGraph)

`POST /multi-agent` runs a **Supervisor** that routes each turn to specialized micro-agents instead of one mega-prompt.

| Agent | Role |
| --- | --- |
| **Supervisor** | Classifies intent → `sales` / `support` / `router` / `finish` (max 4 hops) |
| **Sales** | Tier/seat quotes + demo calendar (`calculate_custom_quote`, `check_calendar_availability`) |
| **Support** | Queries local ChromaDB (LangChain RAG) for SLA / refunds / onboarding / docs |
| **Router** | Sentiment + triggers Slack/email workflows (`trigger_slack_workflow`, `trigger_email_workflow`) |

**CrewAI is not used** — LangGraph fits the existing LangChain RAG stack. `/analyze-lead` and `/agent-action` stay unchanged.

## Request

```json
{
  "email": "alex@acme.com",
  "message": "We need a Growth quote for 25 seats, and what's your SLA?",
  "thread_id": null
}
```

`thread_id` is optional (same Redis memory as other endpoints).

## Response

```json
{
  "email": "alex@acme.com",
  "agent_reply": "…",
  "route_trace": ["sales", "support"],
  "tools_called": [],
  "knowledge_sources": ["Service Level Agreement (SLA)"],
  "sentiment": null,
  "notifications": [],
  "hops": 3,
  "thread_id": "…"
}
```

## Router webhooks

Set optional env vars (see `.env.example`):

| Variable | Effect |
| --- | --- |
| `SLACK_WEBHOOK_URL` | POST JSON alert; unset → stub `notifications[].stub=true` |
| `EMAIL_WEBHOOK_URL` | POST JSON email payload; unset → stub |

Wire these to n8n webhooks if you want real Slack/email branches.

## Example curls

```bash
# Sales
curl -sS http://127.0.0.1:8000/multi-agent \
  -H 'Content-Type: application/json' \
  -d '{"email":"alex@acme.com","message":"Quote for 12 Growth seats"}'

# Support (KB)
curl -sS http://127.0.0.1:8000/multi-agent \
  -H 'Content-Type: application/json' \
  -d '{"email":"alex@acme.com","message":"What is your uptime SLA?"}'

# Router (negative / urgent)
curl -sS http://127.0.0.1:8000/multi-agent \
  -H 'Content-Type: application/json' \
  -d '{"email":"alex@acme.com","message":"This is urgent — your billing is broken and I want a human NOW"}'
```
