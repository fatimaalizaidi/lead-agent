# Conversation memory (Redis)

Multi-turn chat history for `POST /analyze-lead` and `POST /agent-action`. Turns are stored in Redis keyed by `thread_id`, loaded before each OpenAI call, then appended after a successful response.

Without Redis (empty `REDIS_URL`), endpoints stay **stateless** and still work. Omitting `thread_id` always works: the server generates a UUID and returns it.

## Request / response

Both endpoints accept optional `thread_id` and always return it:

```json
{
  "email": "alex@acme.com",
  "message": "We need Growth seats for 20 people next month.",
  "thread_id": "optional-uuid-from-prior-turn"
}
```

Response includes `"thread_id": "..."`. Reuse that value on the next POST for the same conversation.

## Behaviour

| Piece | Detail |
| --- | --- |
| Store | Redis list `thread:{id}:messages` — JSON `{role, content}` |
| Roles kept | `user` / `assistant` (tool internals from `/agent-action` are not persisted) |
| TTL | `MEMORY_TTL_SECONDS` (default **7 days** / `604800`) refreshed on each append |
| Cap | `MEMORY_MAX_MESSAGES` (default **20**); oldest messages trimmed |
| Soft fail | Redis down → empty history; request still completes |

## Compose / env

`docker-compose.yml` runs a `redis:7-alpine` service and sets:

```bash
REDIS_URL=redis://redis:6379/0
MEMORY_TTL_SECONDS=604800
MEMORY_MAX_MESSAGES=20
```

See `.env.example`. Local uvicorn against a published Redis: `REDIS_URL=redis://localhost:6379/0`.

```bash
docker compose up -d --build
curl -s http://localhost:8000/health
```

## n8n: pass `thread_id` through

1. **Webhook** — Accept `thread_id` from the caller (or leave it out on the first turn).
2. **HTTP Request** — Include `thread_id` in the JSON body to `/analyze-lead` or `/agent-action`.
3. **Persist** — After the response, save `thread_id` for the next turn:
   - **Workflow static data** (same execution / sub-workflow):  
     `const staticData = $getWorkflowStaticData('global'); staticData.threadId = $json.thread_id;`
   - **Return to caller** — Put `thread_id` in the webhook response so the client sends it back next time.
4. **Follow-ups** — On the next webhook, read static data (or the client-provided id) and pass it again.

Example HTTP Request body expression:

```json
{
  "email": "{{ $json.email }}",
  "message": "{{ $json.message }}",
  "thread_id": "{{ $json.thread_id || $getWorkflowStaticData('global').threadId || undefined }}"
}
```

Omit `thread_id` (or send `null`) on the first message; always echo the response `thread_id` back for turns 2+.

## Smoke test

```bash
# Turn 1 — capture thread_id
THREAD=$(curl -sS http://localhost:8000/analyze-lead \
  -H 'Content-Type: application/json' \
  -d '{"email":"alex@acme.com","message":"Our company is Contoso and we need 50 Growth seats."}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['thread_id'])")

# Turn 2 — same thread; model should see prior context (e.g. Contoso / seat count)
curl -sS http://localhost:8000/analyze-lead \
  -H 'Content-Type: application/json' \
  -d "{\"email\":\"alex@acme.com\",\"message\":\"What did I say our company was called?\",\"thread_id\":\"$THREAD\"}"
```

Inspect Redis (optional):

```bash
docker compose exec redis redis-cli LRANGE "thread:${THREAD}:messages" 0 -1
```
