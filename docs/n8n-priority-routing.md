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
   Analyze Lead         POST https://lead-agent-fjxq.onrender.com/analyze-lead
        │               Header Auth X-API-Key, retry 5x on fail
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

The live setup calls the Lead Agent deployed on Render (`https://lead-agent-fjxq.onrender.com`). A local API at `http://host.docker.internal:8000` is optional and only for local development (see [Local development](#local-development-optional)).

**Do not** Switch on `/agent-action` for this milestone — that endpoint has no `priority`. Optional later: a second HTTP call after analyze for tool actions.

## Prerequisites

1. Lead Agent on Render is up: `curl https://lead-agent-fjxq.onrender.com/health` → `{"status":"ok"}`. The first call after the service has been idle can take 30–50 s (see [Render cold start](#render-cold-start)).
2. An API key for n8n is in the Render service's `API_KEYS`, and n8n has a **Header Auth** credential for it (see [API key](#api-key-for-lead-agent-required-once-api_keys-is-set)).
3. n8n running (currently local Docker). See [Who can call the webhook](#who-can-call-the-webhook) for what that means for outside services.
4. Workflow JSON from this folder (or the repo `n8n/` copy).
5. Optional, local development only: Lead Agent on the Mac at `http://127.0.0.1:8000/health`, reachable from the n8n container as `host.docker.internal:8000`.

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

| Mode | URL | When it works |
|------|-----|---------------|
| Test | `http://127.0.0.1:5678/webhook-test/lead-priority` | Only after you click **Listen for test event** in the editor. It accepts **one** request, then stops listening. The run shows live in the editor. |
| Production | `http://127.0.0.1:5678/webhook/lead-priority` | Only while the workflow is **Active** (toggle at top right). Runs appear under **Executions**. Returns 404 while inactive. |
| Public via ngrok | `https://YOUR-NGROK-HOST/webhook/lead-priority` | Production path through a tunnel (see below). |

Calling `/webhook-test/...` without **Listen for test event**, or `/webhook/...` while the workflow is inactive, returns 404 `webhook not registered`.

### Who can call the webhook

n8n currently runs in local Docker, so `127.0.0.1:5678` is reachable only from this Mac (Postman, curl). Outside services such as a website form or WhatsApp can call the webhook only when n8n is reachable online: host it on **n8n Cloud**, or expose the local instance with **ngrok** (`ngrok http 5678`) and set `WEBHOOK_URL` to the public URL so n8n shows the right webhook address. The Render API is already public, so only n8n needs this.

Body (Postman / curl):

```json
{
  "email": "alex@example.com",
  "message": "We need a demo this week for a 200-seat rollout. Budget is approved."
}
```

## Credentials Ali must paste

### API key for Lead Agent (required once `API_KEYS` is set)

When the API has `API_KEYS` set (always the case on Render), every call to `/analyze-lead`, `/multi-agent` and `/agent-action` needs an API key, or it returns `401 missing_credentials`. See [security.md](./security.md).

1. Generate a key: `python -m security new-key n8n` prints `n8n:la_…`. Add the whole line to the API's `API_KEYS`.
2. In n8n: **Credentials → Add credential → Header Auth**, named **Header Auth account** (the name the workflow JSON refers to).
   - **Name:** `X-API-Key`. HTTP header names are case-insensitive, so `x-api-key` works too.
   - **Value:** only the `la_…` part, **without** the `n8n:` prefix. The prefix is the key's label in `API_KEYS`, not part of the key.
3. Open the **Analyze Lead** HTTP Request node → **Authentication: Generic Credential Type** → **Generic Auth Type: Header Auth** → pick **Header Auth account**. Do the same on any extra HTTP Request node, such as `/agent-action`.

The exported workflow references the credential **by name only**. After import, n8n links it to your local credential of that name; if none exists, pick it in the node. No credential id or key value is stored in the JSON.

Keep the key in an n8n credential rather than a plain header parameter: credentials are encrypted and aren't included when you export the workflow JSON. Running the API locally with `API_KEYS` empty turns auth off, and the header is then ignored.

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

## Render cold start

Render's free plan puts the service to sleep after about 15 minutes without traffic. The first request after that wakes it, which takes **30–50 s**. While it wakes, Render answers with **HTTP 503** and an HTML page that says "Application loading", not JSON from the API.

That is why **Analyze Lead** has:

- **Timeout:** 60000 ms (60 s) per try
- **Retry On Fail:** on, **Max Tries:** 5 (n8n's maximum), **Wait Between Tries:** 20000 ms in the JSON

Caveat: the n8n engine caps **Wait Between Tries** at **5000 ms** (and Max Tries at 5), whatever the node says. So the real budget is 5 tries with about 5 s between them (about 20 s of waiting plus the request times). If Render answers 503 quickly every time, the tries can still run out before the service is awake. If the first production lead of the day still fails with 503:

- open `https://lead-agent-fjxq.onrender.com/health` (browser or curl) a minute before testing to wake the service, or
- add a **Wait** node plus a second HTTP Request (Max Tries is already at n8n's maximum of 5), or
- move the service to a paid Render plan, which does not sleep.

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

Use the **production webhook** after Activate, or click **Listen for test event** and swap the path for `/webhook-test/lead-priority`. If Render was asleep, the first call can take up to a minute (see [Render cold start](#render-cold-start)).

Priorities are LLM-classified. The examples below use fake data and usually land as labeled; if one flips, check the response `priority` field.

### High → Slack

```bash
curl -sS -X POST "http://127.0.0.1:5678/webhook/lead-priority" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "alex@example.com",
    "message": "We need a demo this week for a 200-seat rollout. Budget is approved."
  }'
```

Expect: Slack message (if URL pasted); JSON with `"priority":"High"`, `"routed_branch":"High"`.

### Medium → local JSON file

```bash
curl -sS -X POST "http://127.0.0.1:5678/webhook/lead-priority" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "jane@example.com",
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
curl -sS -X POST "https://lead-agent-fjxq.onrender.com/analyze-lead" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: la_..." \
  -d '{"email":"jane@example.com","message":"Curious about your product for a small team of 5. Not urgent — just researching options for next quarter."}'
```

## Optional second call (`/agent-action`)

Useful after Medium/High when you want quote / calendar / escalate tools. Add another **HTTP Request** to `https://lead-agent-fjxq.onrender.com/agent-action` (same Header Auth credential and retry settings) with the same `{email,message}` — **after** analyze+branch, not as the Switch input (response has `agent_reply` / `tools_called`, not `priority`).

## Troubleshooting

| Symptom | Check |
|---------|--------|
| Analyze Lead returns 401 | `missing_credentials`: the node has no Header Auth credential. `invalid_api_key`: the value is wrong. Use only the `la_…` part, not `n8n:la_…` |
| Analyze Lead returns 429 `rate_limited` | Over `RATE_LIMIT` (default 20/minute per key). Wait `Retry-After` seconds, or raise the limit on the API |
| Analyze Lead returns 503 with "Application loading" HTML | Render cold start. Wait about a minute and retry, or wake it first with `curl https://lead-agent-fjxq.onrender.com/health` (see [Render cold start](#render-cold-start)) |
| Analyze Lead fails / timeout | `curl https://lead-agent-fjxq.onrender.com/health`. Local dev: `curl http://127.0.0.1:8000/health`; from the container: `wget -qO- http://host.docker.internal:8000/health` |
| Webhook returns 404 `webhook not registered` | Test URL without **Listen for test event**, or production URL while the workflow is inactive |
| Always Other branch | Inspect Analyze Lead output for `priority` |
| Slack 404 / invalid_token | Webhook URL not replaced |
| Medium write error / “Access to the file is not allowed” | Path must be under `~/.n8n-files`. Do not write into `~/.n8n/` |
| Email error | SMTP credential missing; node is continue-on-fail so webhook still 200 |

## Local development (optional)

To run against a Lead Agent on your Mac instead of Render:

1. Start the API locally (`uvicorn main:app --port 8000`) and check `http://127.0.0.1:8000/health`.
2. In **Analyze Lead**, set the URL to `http://host.docker.internal:8000/analyze-lead` (n8n in Docker reaches the host through `host.docker.internal`).
3. With `API_KEYS` empty locally, auth is off and the Header Auth credential is ignored. Retries are harmless, since there is no cold start locally.

Switch the URL back to `https://lead-agent-fjxq.onrender.com/analyze-lead` before exporting the workflow to the repo.
