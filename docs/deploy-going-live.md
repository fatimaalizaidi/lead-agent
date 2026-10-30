# Deploy / Going live — Lead Analyzer on Render

**Host:** Render (Docker Web Service).  
**GitHub target:** https://github.com/fatimaalizaidi/lead-agent (Ali creates empty repo + pushes).  
Full guide: `docs/lead-analyzer/DEPLOY.md` (synced from `~/Projects/lead-agent/DEPLOY.md`).

## Project files

| File | Role |
|------|------|
| `Dockerfile` | uvicorn on `0.0.0.0:$PORT`; app + `knowledge/`; no `.env` baked; build-essential for chroma then purged |
| `render.yaml` | Blueprint Web Service, health `/health`, `OPENAI_API_KEY` sync:false |
| `docker-compose.yml` | Local: build, port 8000, `env_file: .env` |
| `.gitignore` | `.env`, `.venv/`, `data/`, `logs/` |

Local git on `main` has commits; **no remote / no push** from agents.

## Ali: push commands

1. GitHub → **New repository** → name `lead-agent` → **no README / no .gitignore / no license** → Create.
2. Then:

```bash
cd ~/Projects/lead-agent
git remote add origin https://github.com/fatimaalizaidi/lead-agent.git
git branch -M main
git push -u origin main
```

## Render click-path

1. [dashboard.render.com](https://dashboard.render.com) → **New +** → **Web Service**
2. Connect **fatimaalizaidi/lead-agent**
3. **Runtime: Docker** · Health Check Path **`/health`** · Start Command **empty**
4. Env secret **`OPENAI_API_KEY`** → **Create Web Service**

## n8n

```text
POST {public_url}/analyze-lead
```

JSON body: `{"message":"...","email":"..."}` · optional header `X-Request-ID`.
