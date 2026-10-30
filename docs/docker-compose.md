# Docker Compose — lead-agent backend

Boots the FastAPI lead analyzer (with persistent Chroma) via `docker compose`. **n8n is not in this stack** — run it separately and call the API on port 8000.

## Prerequisites

1. Docker Desktop (or Engine + Compose plugin).
2. Repo checkout at `~/Projects/lead-agent`.
3. Local `.env` (gitignored):

```bash
cp .env.example .env
# edit .env → set OPENAI_API_KEY=sk-...
```

Local `uvicorn` still works the same way; Compose does not replace it.

## Up / down

```bash
cd ~/Projects/lead-agent
docker compose up -d --build
curl -s http://localhost:8000/health
# → {"status":"ok"}

# OpenAPI / routes
curl -s http://localhost:8000/openapi.json | python3 -c "import sys,json; print(sorted(json.load(sys.stdin)['paths']))"

docker compose logs -f lead-analyzer   # follow logs
docker compose down                    # stop; keep Chroma volume
docker compose down -v                 # stop and wipe Chroma + log volumes
```

Override the **host** port (container still listens on 8000):

```bash
LEAD_AGENT_PORT=8080 docker compose up -d
curl -s http://localhost:8080/health
```

## Volumes

| Volume        | Mount                  | Purpose                                      |
|---------------|------------------------|----------------------------------------------|
| `chroma_data` | `/app/data/chroma`     | Persisted Chroma index (`rag.py` `CHROMA_DIR`) |
| `lead_logs`   | `/app/logs`            | Rotating app logs                            |

Knowledge markdown is **baked into the image** from `knowledge/`. After editing KB files, rebuild (`docker compose up -d --build`). The next request reindexes when the fingerprint changes.

## Environment

Compose loads `.env` via `env_file` and never bakes secrets into the image.

| Variable | Required | Notes |
|----------|----------|--------|
| `OPENAI_API_KEY` | yes | From `.env` |
| `OPENAI_MODEL` | no | default `gpt-4o-mini` |
| `RAG_K` | no | default `3` |
| `RAG_SCORE_THRESHOLD` | no | default `1.2` |
| `OPENAI_TIMEOUT_SECONDS` | no | default `30` |
| `LEAD_AGENT_PORT` | no | Host publish port; default `8000` |
| `PORT` | set by compose | Container listen port `8000` |

See `.env.example` for the template.

## n8n → lead-agent

n8n (Docker or desktop) should POST to the published API, not into this compose file.

| Where n8n runs | Base URL for HTTP Request nodes |
|----------------|----------------------------------|
| Same machine, n8n **outside** Docker | `http://localhost:8000` |
| n8n **in Docker** on Docker Desktop (Mac/Windows) | `http://host.docker.internal:8000` |
| n8n in Docker on Linux | `http://host.docker.internal:8000` (add `extra_hosts: ["host.docker.internal:host-gateway"]` on the n8n service) or use the host LAN IP |

Example: `POST http://host.docker.internal:8000/analyze-lead` with JSON `{"message":"...","email":"..."}`.

Healthcheck path for probes: `GET /health`.

## Healthcheck

The `lead-analyzer` service probes `GET /health` with Python’s stdlib (no extra packages). `docker compose ps` shows `healthy` once the app is up.
