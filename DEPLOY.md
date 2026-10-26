# Deploy / Going live (Render)

Expose the Lead Analyzer so **n8n** can `POST` JSON to `/analyze-lead` over HTTPS.

**Chosen host: [Render](https://render.com)** — Web Service from Docker (`Dockerfile` + optional `render.yaml`).

You need an OpenAI API key as env var `OPENAI_API_KEY`. Never commit `.env`, bake keys into the image, or paste keys into git/chat/screenshots. `.gitignore` already ignores `.env` and `.env.*` (keeps `.env.example`).

---

## 1. Init git and push to GitHub (you do this)

Render deploys from a GitHub/GitLab repo. If this folder is not a git repo yet:

```bash
cd ~/Projects/lead-agent

# Confirm secrets will not be committed
grep -n '^\.env' .gitignore   # should list .env and .env.*

git init
git add .
git status   # must NOT list .env — only .env.example is OK
git commit -m "Initial commit: Lead Analyzer API with Docker/Render deploy assets"
```

Create an empty GitHub repo (e.g. `lead-agent`), then:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/lead-agent.git
git push -u origin main
```

(Agents will not create the remote or push for you — no GitHub credentials assumed.)

---

## 2. Create a Render Web Service (exact click-path)

### Option A — Manual Web Service (clearest for first deploy)

1. Open [https://dashboard.render.com](https://dashboard.render.com) and sign in.
2. Click **New +** → **Web Service**.
3. **Connect a repository** → authorize GitHub if needed → select your `lead-agent` repo → **Connect**.
4. Configure:
   | Field | Value |
   |-------|--------|
   | **Name** | `lead-analyzer` (or any name) |
   | **Region** | closest to you |
   | **Branch** | `main` |
   | **Runtime** | **Docker** |
   | **Dockerfile Path** | `./Dockerfile` (default) |
   | **Docker Context** | `.` |
   | **Instance type** | **Free** is fine for class demos |
5. **Health Check Path**: `/health`
6. **Start Command**: leave **empty** — the Dockerfile `CMD` already runs  
   `uvicorn main:app --host 0.0.0.0 --port $PORT`  
   (Render injects `PORT`; do not hardcode `8000` in the dashboard.)
7. Under **Environment** → **Add Environment Variable**:
   | Key | Value |
   |-----|--------|
   | `OPENAI_API_KEY` | your key (use **Secret** / leave unsynced) |

   Optional: `OPENAI_MODEL=gpt-4o-mini`, `RAG_K`, `RAG_SCORE_THRESHOLD`, `OPENAI_TIMEOUT_SECONDS` (see `.env.example`).
8. Click **Create Web Service** and wait for the first build/deploy (several minutes).

### Option B — Blueprint (`render.yaml`)

1. Dashboard → **New +** → **Blueprint**.
2. Connect the same repo (must contain `render.yaml` at the root).
3. Apply the Blueprint. Set `OPENAI_API_KEY` when prompted (`sync: false` = you enter it in the dashboard; it is not stored in the yaml).

---

## 3. Confirm the public URL

After status is **Live**, Render shows a URL like:

```text
https://lead-analyzer-xxxx.onrender.com
```

Smoke tests:

```bash
# Liveness
curl -s https://YOUR-SERVICE.onrender.com/health

# Analyze (replace host; never put your API key in curl for the public URL — the server already has it)
curl -s -X POST "https://YOUR-SERVICE.onrender.com/analyze-lead" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: demo-n8n-001" \
  -d '{"message":"Need a demo this week for 200 seats. Budget approved.","email":"alex@acme.com"}'
```

**Notes**

- Free instances **sleep when idle**; the first request after sleep can take ~30–60s.
- Ephemeral disk: Chroma under `data/chroma` rebuilds on first request after a fresh deploy/restart.
- Never put `OPENAI_API_KEY` in the Dockerfile or commit it.

---

## 4. Wire n8n (HTTP Request node)

| Setting | Value |
|---------|--------|
| **Method** | `POST` |
| **URL** | `https://YOUR-SERVICE.onrender.com/analyze-lead` |
| **Send Body** | On |
| **Body Content Type** | JSON |
| **JSON** | see below |
| **Headers** | `Content-Type: application/json` |
| **Optional header** | `X-Request-ID` = any string (or `{{ $execution.id }}`) for log correlation |

**JSON body** (map fields from your workflow):

```json
{
  "message": "{{ $json.message }}",
  "email": "{{ $json.email }}"
}
```

**URL pattern to remember:**

```text
POST {public_url}/analyze-lead
```

Example: `https://lead-analyzer-xxxx.onrender.com/analyze-lead`

Expected JSON fields in the response include `priority`, `suggested_response`, `rationale`, `knowledge_sources`, `needs_human_review`, `fallback_reason`.

---

## Path A (alternate) — ngrok quick tunnel

Use only if you want a temporary URL without Render. Start locally (`uvicorn` or `docker compose up --build`), then in another terminal: `ngrok http 8000`. Point n8n at `https://YOUR-NGROK-HOST/analyze-lead` the same way as above. Free ngrok URLs change on restart. Do not ask agents to run `ngrok` auth for you.

---

## Checklist

- [ ] `.env` is **not** in git (`git status` clean of secrets)
- [ ] Repo pushed to GitHub; Render Web Service = **Docker**
- [ ] Env `OPENAI_API_KEY` set as a **secret** on Render
- [ ] Health check path `/health`; start command left empty (Dockerfile CMD)
- [ ] `GET /health` OK on the public URL
- [ ] n8n: `POST {public_url}/analyze-lead` with JSON `message` + `email` (optional `X-Request-ID`)

## Local Docker reference (optional)

```bash
docker build -t lead-analyzer .
docker run --rm -p 8000:8000 --env-file .env lead-analyzer
# or: docker compose up --build
```
