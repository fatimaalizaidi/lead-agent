# Multi-agent system (LangGraph)

`POST /multi-agent` runs a **Supervisor** that routes each turn to specialized micro-agents instead of one mega-prompt.

| Agent | Role |
| --- | --- |
| **Supervisor** | Analyzes incoming intent and delegates → `sales` / `knowledge` / `escalation` / `finish` (max 4 hops) |
| **Sales/Pricing** | Seat and tier pricing function calls (`calculate_custom_quote`; demo slots via `check_calendar_availability`) |
| **Knowledge** | Queries local ChromaDB for SLAs, policies, and technical documentation |
| **Escalation** | High-priority Slack/email dispatch (`trigger_slack_workflow`, `trigger_email_workflow`) |

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
  "route_trace": ["sales", "knowledge"],
  "tools_called": [],
  "knowledge_sources": ["Service Level Agreement (SLA)"],
  "sentiment": null,
  "notifications": [],
  "hops": 3,
  "thread_id": "…"
}
```

## Escalation webhooks

Set optional env vars (see `.env.example`):

| Variable | Effect |
| --- | --- |
| `SLACK_WEBHOOK_URL` | POST JSON alert; unset → stub `notifications[].stub=true` |
| `EMAIL_WEBHOOK_URL` | POST JSON email payload; unset → stub |

Wire these to n8n webhooks if you want real Slack/email branches.

## Example curls

```bash
# Sales/Pricing
curl -sS http://127.0.0.1:8000/multi-agent \
  -H 'Content-Type: application/json' \
  -d '{"email":"alex@acme.com","message":"Quote for 12 Growth seats"}'

# Knowledge (ChromaDB)
curl -sS http://127.0.0.1:8000/multi-agent \
  -H 'Content-Type: application/json' \
  -d '{"email":"alex@acme.com","message":"What is your uptime SLA?"}'

# Escalation (high priority)
curl -sS http://127.0.0.1:8000/multi-agent \
  -H 'Content-Type: application/json' \
  -d '{"email":"alex@acme.com","message":"This is urgent — your billing is broken and I want a human NOW"}'
```
