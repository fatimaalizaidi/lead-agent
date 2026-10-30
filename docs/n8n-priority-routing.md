# n8n priority routing hub

Enterprise-style lead hub: **Webhook → Edit Fields → Analyze Lead (`/analyze-lead`) → Switch on `priority` → High / Medium / Low → Respond to Webhook**.

Importable workflow: [`n8n-priority-routing-workflow.json`](./n8n-priority-routing-workflow.json)  
Repo copy: `~/Projects/lead-agent/n8n/lead-priority-routing.json`

## Architecture

```text
POST /webhook/lead-priority
        │
        ▼
   Edit Fields          { message, email } from body
        │
        ▼
   Analyze Lead         POST http://host.docker.internal:8000/analyze-lead
        │               returns priority, suggested_response, rationale, ...
        ▼
 Route by Priority      Switch on normalized priority
   ├── High  → Notify Slack (Incoming Webhook HTTP POST)
   ├── Medium → Convert JSON → write file under `~/.n8n-files`
   ├── Low   → Send Email (SMTP auto-reply using suggested_response)
   └── Other → (fallback) skip side effects
        │
        ▼
 Build Webhook Response → Respond to Webhook (analyze-lead JSON + routed_branch)
```

**Do not** Switch on `/agent-action` for this milestone — that endpoint has no `priority`. Optional later: a second HTTP call after analyze for tool actions.

## Prerequisites

1. Lead Analyzer on the Mac: `http://127.0.0.1:8000/health` → `{"status":"ok"}`.
2. n8n (Docker) can reach the host API via `host.docker.internal:8000`.
3. Workflow JSON from this folder (or the repo `n8n/` copy).

## Import

### UI

1. Open n8n → **Workflows** → **⋯** / **Import from File**.
2. Select `n8n-priority-routing-workflow.json`.
3. Open **Lead Priority Routing Hub**.

### CLI (local Docker)

```bash
# from a path mounted or copied into the container
docker cp "/path/to/n8n-priority-routing-workflow.json" n8n-workspace-n8n-1:/tmp/lead-priority-routing.json
docker exec n8n-workspace-n8n-1 n8n import:workflow \
  --input=/tmp/lead-priority-routing.json \
  --userId=aeec7432-71bc-4655-b6a1-ed52ef67b773
```

Then open the workflow in the UI, fix credentials (below), and **Activate**.

### Webhook URL

| Mode | URL |
|------|-----|
| Production (active) | `http://127.0.0.1:5678/webhook/lead-priority` |
| Test (listen once) | `http://127.0.0.1:5678/webhook-test/lead-priority` |
| Public via ngrok | `https://YOUR-NGROK-HOST/webhook/lead-priority` |

Body (Postman / curl):

```json
{
  "email": "alex@acme.com",
  "message": "We need a demo this week for a 200-seat rollout. Budget is approved."
}
```

## Credentials Ali must paste

### High — Slack Incoming Webhook (required for live notify)

1. [api.slack.com/apps](https://api.slack.com/apps) → Create App → **Incoming Webhooks** → On → **Add New Webhook to Workspace**.
2. Copy `https://hooks.slack.com/services/...`.
3. In n8n open **Notify Slack (High)** → replace the URL  
   `https://hooks.slack.com/services/REPLACE/WITH/YOUR_WEBHOOK`  
   with your URL.
4. Optional alternatives (not wired): Slack app OAuth, Telegram Bot API, or SMTP to a sales alias — Incoming Webhook is the zero-paid-secrets path.

Without a real URL the High branch still runs (`continueOnFail`); Slack returns an error but the webhook response still returns analyze-lead JSON.

### Medium — local file store (works with no cloud credentials)

No Google OAuth / HubSpot key required.

- Writes one JSON file per Medium lead to:  
  `/home/node/.n8n-files/medium-leads/`  
  (n8n default allowlist `N8N_RESTRICT_FILE_ACCESS_TO=~/.n8n-files`; `.n8n/` is blocked)
- Inspect:

```bash
docker exec n8n-workspace-n8n-1 ls -la /home/node/.n8n-files/medium-leads
docker exec n8n-workspace-n8n-1 sh -c 'head -c 800 /home/node/.n8n-files/medium-leads/*.json | head'
```

Upgrade later: swap **Store Medium Lead** for Google Sheets or HubSpot with an API key credential.

### Low — SMTP / Gmail (required for live email)

1. In n8n: **Credentials** → **SMTP** (host, port, user, password / app password).
2. Open **Send Auto-Reply (Low)** → select that credential.
3. Set **From Email** to an address your SMTP allows.
4. Or replace the node with **Gmail** and connect Google OAuth.

Without SMTP the Low branch continues (`continueOnFail`) so Postman still gets the analyze JSON; no email is sent until creds exist.

## Switch expressions

Node **Route by Priority** (Switch v3.2), rules mode:

| Output | Expression (left) | Equals |
|--------|-------------------|--------|
| High | `{{ ($json.priority \|\| '').toString().trim().toLowerCase() }}` | `high` |
| Medium | same | `medium` |
| Low | same | `low` |
| Other | fallback when priority missing / unexpected | — |

Options: **Ignore Case** on; **Fallback Output** = Extra (`Other`).

API returns exactly `High` | `Medium` | `Low`; normalizing with `.toLowerCase()` avoids brittle matches if casing ever drifts.

## Test each branch (Postman / curl)

Keep lead-agent on `:8000`. Prefer **production webhook** after Activate (or click **Listen for test event** for `/webhook-test/...`).

Priorities are LLM-classified — examples below were verified against local `/analyze-lead` and usually land as labeled; if one flips, check the response `priority` field.

### High → Slack

```bash
curl -sS -X POST "http://127.0.0.1:5678/webhook/lead-priority" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "alex@acme.com",
    "message": "We need a demo this week for a 200-seat rollout. Budget is approved."
  }'
```

Expect: Slack message (if URL pasted); JSON with `"priority":"High"`, `"routed_branch":"High"`.

### Medium → local JSON file

```bash
curl -sS -X POST "http://127.0.0.1:5678/webhook/lead-priority" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "jane@startup.io",
    "message": "Curious about your product for a small team of 5. Not urgent — just researching options for next quarter."
  }'
```

Expect: new file under `medium-leads/`; JSON with `"priority":"Medium"`.

### Low → auto-reply email

```bash
curl -sS -X POST "http://127.0.0.1:5678/webhook/lead-priority" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "curious@example.com",
    "message": "Just browsing, send me a brochure when you can."
  }'
```

Expect: SMTP send when credential configured; JSON with `"priority":"Low"` and body built from `suggested_response`.

### Direct API check (bypass n8n)

```bash
curl -sS -X POST "http://127.0.0.1:8000/analyze-lead" \
  -H "Content-Type: application/json" \
  -d '{"email":"jane@startup.io","message":"Curious about your product for a small team of 5. Not urgent — just researching options for next quarter."}'
```

## Optional second call (`/agent-action`)

Useful after Medium/High when you want quote / calendar / escalate tools. Add another **HTTP Request** to `http://host.docker.internal:8000/agent-action` with the same `{email,message}` — **after** analyze+branch, not as the Switch input (response has `agent_reply` / `tools_called`, not `priority`).

## Troubleshooting

| Symptom | Check |
|---------|--------|
| Analyze Lead fails / timeout | `curl http://127.0.0.1:8000/health`; from container: `wget -qO- http://host.docker.internal:8000/health` |
| Always Other branch | Inspect Analyze Lead output for `priority` |
| Slack 404 / invalid_token | Webhook URL not replaced |
| Medium write error / “Access to the file is not allowed” | Path must be under `~/.n8n-files`. Do not write into `~/.n8n/` |
| Email error | SMTP credential missing; node is continue-on-fail so webhook still 200 |
