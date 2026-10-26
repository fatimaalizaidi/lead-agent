# Deploy / Going live

Student-friendly steps to expose the Lead Analyzer so **n8n** can `POST` JSON to `/analyze-lead`.

You need an OpenAI API key in the environment as `OPENAI_API_KEY`. Never commit `.env` or paste keys into git, chat, or screenshots.

---

## Path A — Quick tunnel (ngrok)

Best for demos and homework when you do not want a cloud host yet.

### 1. Start the API locally

**Option A1 — uvicorn (venv)**

```bash
cd ~/Projects/lead-agent
source .venv/bin/activate
# Ensure .env has OPENAI_API_KEY=...
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Option A2 — Docker Compose**

```bash
cd ~/Projects/lead-agent
# Ensure .env exists with OPENAI_API_KEY (compose loads env_file: .env)
docker compose up --build
```

Confirm locally:

```bash
curl -s http://127.0.0.1:8000/health
```

### 2. Install and run ngrok

1. Create a free account at [ngrok.com](https://ngrok.com) and install the CLI.
2. One-time auth (you run this yourself — agents will not):

```bash
ngrok config add-authtoken YOUR_TOKEN_FROM_DASHBOARD
```

3. In a **second** terminal, tunnel port 8000:

```bash
ngrok http 8000
```

Copy the **https** Forwarding URL (looks like `https://abcd-12-34-56-78.ngrok-free.app`).

### 3. Wire n8n

In an **HTTP Request** node:

| Setting | Value |
|---------|--------|
| Method | `POST` |
| URL | `https://YOUR-NGROK-HOST/analyze-lead` |
| Body | JSON |
| JSON body | `{"message": "{{ $json.message }}", "email": "{{ $json.email }}"}` (adjust to your fields) |
| Headers | `Content-Type: application/json` (optional: `X-Request-ID` for tracing) |

Smoke test from your machine:

```bash
curl -s -X POST "https://YOUR-NGROK-HOST/analyze-lead" \
  -H "Content-Type: application/json" \
  -d '{"message":"Need a demo this week for 200 seats. Budget approved.","email":"alex@acme.com"}'
```

**Notes**

- Free ngrok URLs change when you restart the tunnel — update n8n each time.
- Keep uvicorn/Docker and ngrok both running while you test.
- Do not share the URL publicly for long; it reaches your laptop.

---

## Path B — Cloud host (Render)

Best when you want a stable HTTPS URL without keeping your laptop on.

This repo includes `Dockerfile` + `render.yaml` (Blueprint). Railway also works with the same Dockerfile (set `OPENAI_API_KEY` as a secret; the image listens on `0.0.0.0:$PORT`).

### 1. Put the project on GitHub (or GitLab)

Push this folder (without `.env`, `.venv`, `data/`, `logs/`) to a repo Render can access. `.gitignore` already excludes secrets and local data.

### 2. Create the Render Web Service

**Blueprint (preferred)**

1. [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**.
2. Connect the repo that contains `render.yaml` and `Dockerfile`.
3. Apply the Blueprint. Render builds from Docker.

**Manual**

1. **New** → **Web Service** → connect repo.
2. Runtime: **Docker** (uses `./Dockerfile`).
3. Plan: Free is fine for class demos (spins down when idle).

### 3. Set the secret

In the service **Environment**:

| Key | Value |
|-----|--------|
| `OPENAI_API_KEY` | your key (mark as secret) |

Optional: `OPENAI_MODEL`, `RAG_K`, `RAG_SCORE_THRESHOLD`, `OPENAI_TIMEOUT_SECONDS` (see `.env.example`).

Render injects `PORT`; the Dockerfile already runs `uvicorn` on `0.0.0.0:$PORT`.

### 4. Deploy and copy the public URL

After the first successful deploy, open:

- Health: `https://YOUR-SERVICE.onrender.com/health`
- Analyze: `https://YOUR-SERVICE.onrender.com/analyze-lead`

### 5. Wire n8n

Same as Path A, but use the Render URL:

| Setting | Value |
|---------|--------|
| Method | `POST` |
| URL | `https://YOUR-SERVICE.onrender.com/analyze-lead` |
| Body | JSON with `message` + `email` |

```bash
curl -s -X POST "https://YOUR-SERVICE.onrender.com/analyze-lead" \
  -H "Content-Type: application/json" \
  -d '{"message":"What does Growth plan cost per month?","email":"buyer@acme.com"}'
```

**Notes**

- Free Render services sleep after idle time — the first request after sleep can take ~30–60s.
- Chroma rebuilds on first request if `data/chroma` is empty (ephemeral disk on free tier).
- Never put `OPENAI_API_KEY` in the Dockerfile or commit it.

### Railway (alternate)

1. New project → Deploy from GitHub → Dockerfile.
2. Variables: set `OPENAI_API_KEY` (secret).
3. Use the generated `*.up.railway.app` URL + `/analyze-lead` in n8n.

---

## Checklist before calling it “live”

- [ ] `GET /health` returns OK on the public URL
- [ ] `POST /analyze-lead` returns JSON with `priority` / `suggested_response`
- [ ] n8n HTTP Request uses **https** + path `/analyze-lead`
- [ ] `OPENAI_API_KEY` is only in local `.env` or the host’s secret store
- [ ] No API keys in git, README screenshots, or shared docs

## Local Docker reference

```bash
# Build (no secrets in the image)
docker build -t lead-analyzer .

# Run with key from env (example — use your own key / .env, never commit it)
docker run --rm -p 8000:8000 --env-file .env lead-analyzer
```

Or: `docker compose up --build` (reads `.env` via `env_file`).
