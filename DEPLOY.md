# Deploy / Going live (Render)

Expose the Lead Analyzer so **n8n** can `POST` JSON to `/analyze-lead` over HTTPS.

**Chosen host: [Render](https://render.com)** — Web Service from Docker (`Dockerfile` + optional `render.yaml`).

**GitHub target:** [https://github.com/fatimaalizaidi/lead-agent](https://github.com/fatimaalizaidi/lead-agent) (create this empty repo yourself, then push).

You need an OpenAI API key as env var `OPENAI_API_KEY`. Never commit `.env`, bake keys into the image, or paste keys into git/chat/screenshots. `.gitignore` already ignores `.env`, `.venv/`, `data/`, and `logs/` (keeps `.env.example`).

Local git is already initialized with commits on `main` (no remote / no push from agents).

---

## 1. Create empty GitHub repo + push (you do this)

1. Open [https://github.com/new](https://github.com/new) while signed in as **fatimaalizaidi**.
2. **Repository name:** `lead-agent`
3. Leave it **Public** or **Private** (your choice).
4. **Do not** add a README, `.gitignore`, or license (empty repo — avoids merge conflicts).
5. Click **Create repository**.

Then on your Mac:

```bash
cd ~/Projects/lead-agent

# Safety check — must NOT list .env
git status
git check-ignore -v .env .venv data logs

git remote add origin https://github.com/fatimaalizaidi/lead-agent.git
git branch -M main
git push -u origin main
```

If `remote add` says the remote already exists:

```bash
git remote set-url origin https://github.com/fatimaalizaidi/lead-agent.git
git branch -M main
git push -u origin main
```

---

## 2. Create a Render Web Service (exact click-path)

1. Open [https://dashboard.render.com](https://dashboard.render.com) and sign in.
2. Click **New +** → **Web Service**.
3. **Connect a repository** → authorize GitHub if needed → select **fatimaalizaidi/lead-agent** → **Connect**.
4. Configure:
   | Field | Value |
   |-------|--------|
   | **Name** | `lead-analyzer` (or any name) |
   | **Region** | closest to you |
   | **Branch** | `main` |
   | **Runtime** | **Docker** |
   | **Dockerfile Path** | `./Dockerfile` |
   | **Docker Context** | `.` |
   | **Instance type** | **Free** is fine for class demos |
5. **Health Check Path:** `/health`
6. **Start Command:** leave **empty** — the Dockerfile `CMD` already runs  
   `uvicorn main:app --host 0.0.0.0 --port $PORT`  
   (Render injects `PORT`; do not hardcode `8000` in the dashboard.)
7. **Environment** → **Add Environment Variable**:
   | Key | Value |
   |-----|--------|
   | `OPENAI_API_KEY` | your key (mark as **Secret**) |

   Optional: `OPENAI_MODEL=gpt-4o-mini` (see `.env.example`).
8. Click **Create Web Service** and wait for the first build/deploy.

**Blueprint alternate:** **New +** → **Blueprint** → connect the same repo (`render.yaml` at root) → set `OPENAI_API_KEY` when prompted.

---

## 3. Public URL + n8n

After status is **Live**, Render shows a URL like:

```text
https://lead-analyzer-xxxx.onrender.com
```

| Check | URL |
|-------|-----|
| Health | `GET https://YOUR-SERVICE.onrender.com/health` |
| Analyze | `POST https://YOUR-SERVICE.onrender.com/analyze-lead` |

### n8n HTTP Request node

| Setting | Value |
|---------|--------|
| **Method** | `POST` |
| **URL** | `https://YOUR-SERVICE.onrender.com/analyze-lead` |
| **Send Body** | On |
| **Body Content Type** | JSON |
| **Headers** | `Content-Type: application/json` |
| **Optional** | `X-Request-ID` = `{{ $execution.id }}` (or any string) |

**JSON body:**

```json
{
  "message": "{{ $json.message }}",
  "email": "{{ $json.email }}"
}
```

**URL pattern:**

```text
POST {public_url}/analyze-lead
```

Smoke test:

```bash
curl -s https://YOUR-SERVICE.onrender.com/health

curl -s -X POST "https://YOUR-SERVICE.onrender.com/analyze-lead" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: demo-n8n-001" \
  -d '{"message":"Need a demo this week for 200 seats. Budget approved.","email":"alex@acme.com"}'
```

**Notes:** Free Render instances sleep when idle (first hit ~30–60s). Chroma rebuilds on a fresh disk. Never put `OPENAI_API_KEY` in the Dockerfile or git.

---

## Checklist

- [ ] Empty GitHub repo `fatimaalizaidi/lead-agent` created (no README)
- [ ] `git push -u origin main` succeeded; `.env` not on GitHub
- [ ] Render Web Service = **Docker**, health `/health`, start command empty
- [ ] Secret `OPENAI_API_KEY` set on Render
- [ ] n8n: `POST {public_url}/analyze-lead` with JSON `message` + `email`

## Local Docker (optional)

```bash
docker compose up --build
# or: docker build -t lead-analyzer . && docker run --rm -p 8000:8000 --env-file .env lead-analyzer
```
