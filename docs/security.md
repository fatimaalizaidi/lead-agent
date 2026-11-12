# API security: API keys, OAuth2/JWT and rate limiting

`security.py` protects the three AI endpoints (`/analyze-lead`, `/multi-agent`, `/agent-action`). Each call costs OpenAI money, so an open endpoint lets anyone spend your credit. `/health` and `/docs` stay public.

| Layer | What it does |
|---|---|
| **API keys** | Clients send `X-API-Key: <secret>`. Several keys can be live at once, so keys rotate without downtime. |
| **OAuth2 / JWT** | `POST /auth/token` (client-credentials grant) trades an API key for a 1-hour HS256 bearer token. |
| **Rate limiting** | slowapi: 20 requests a minute and 200 a day per API key on the AI endpoints, and 10 a minute per IP on the token endpoint. Over the limit returns HTTP 429. |

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `API_KEYS` | empty | Comma-separated `key_id:secret` pairs, e.g. `n8n:la_…,demo:la_…`. **Empty = auth off** (a warning is logged at startup). A malformed value stops the app from starting, so a typo can't silently turn auth off. |
| `JWT_SECRET` | empty | Signing secret for `/auth/token`. Empty = tokens disabled; API keys still work. |
| `JWT_TTL_SECONDS` | `3600` | Token lifetime. |
| `RATE_LIMIT` | `20/minute;200/day` | Per key on the AI endpoints ([limits syntax](https://limits.readthedocs.io/en/stable/quickstart.html#rate-limit-string-notation)). |
| `RATE_LIMIT_TOKEN` | `10/minute` | Per IP on `/auth/token` (slows down secret guessing). |
| `RATE_LIMIT_ENABLED` | `true` | `false` turns rate limiting off. |
| `RATE_LIMIT_STORAGE_URI` | `memory://` | Counters in process memory: right for one instance. With several instances, use a shared store such as `redis://…` so they share one budget. |

Generate values:

```bash
python -m security new-key n8n     # → n8n:la_<43 random characters>
python -m security new-secret      # → a JWT_SECRET
```

## Turning it on (Render)

1. Generate a key for each client (`n8n`, and `demo` if you want to try it from `/docs`), plus a JWT secret.
2. Render → service → **Environment**: add `API_KEYS=n8n:la_…,demo:la_…` and `JWT_SECRET=…`, then **Save and deploy**.
3. In n8n, open the **HTTP Request** node that calls the API → **Authentication: Generic Credential Type → Header Auth**, with name `X-API-Key` and the `la_…` secret as the value. Store it as an n8n credential, not as a plain header, so it isn't exported with the workflow JSON.
4. Check: a request without the header now returns `401 missing_credentials`.

## Using it

```bash
# API key
curl -X POST "$URL/analyze-lead" -H "X-API-Key: la_..." -H "Content-Type: application/json" \
  -d '{"email":"buyer@acme.com","message":"Pricing for 20 seats?"}'

# OAuth2 client credentials → JWT
curl -X POST "$URL/auth/token" -d grant_type=client_credentials -d client_id=n8n -d client_secret=la_...
# {"access_token":"eyJ...","token_type":"bearer","expires_in":3600}
curl -X POST "$URL/analyze-lead" -H "Authorization: Bearer eyJ..." -H "Content-Type: application/json" -d '...'
```

In `/docs`, click **Authorize** and paste a key (or a token) to try the endpoints.

## Rotating a key (no downtime)

1. Generate a new key with a new id: `API_KEYS=n8n:la_old…,n8n-2026-11:la_new…`. Both work.
2. Switch n8n to the new secret.
3. Remove the old pair: `API_KEYS=n8n-2026-11:la_new…`. The old key stops working at once, **and so do any JWTs issued for it**, because a token is only accepted while its key id is still in `API_KEYS`.

If a key leaks, do step 3 straight away. Rotating `JWT_SECRET` invalidates every outstanding token.

## Errors

All errors use the API's usual body, `{"detail": "...", "error_code": "..."}`:

| Status | `error_code` | When |
|---|---|---|
| 401 | `missing_credentials` | No `X-API-Key` and no bearer token |
| 401 | `invalid_api_key` | Unknown key |
| 401 | `invalid_token` / `token_expired` / `key_revoked` | Bad, expired, or revoked-key token |
| 401 | `invalid_client` | Wrong `client_id` / `client_secret` on `/auth/token` |
| 401 | `jwt_disabled` | `/auth/token` called while `JWT_SECRET` is empty |
| 400 | `unsupported_grant_type` | `grant_type` isn't `client_credentials` |
| 429 | `rate_limited` | Over the limit; the `Retry-After` header gives the seconds to wait |

n8n already retries on 429.

## Design notes

- **Why API keys first:** the callers are machines (n8n), not users signing in, and API keys are the standard for that. The JWT endpoint follows the OAuth2 client-credentials flow, so tokens can be short-lived and the long-lived key is sent once an hour instead of on every request.
- **Secrets are never logged.** Logs only carry error codes and, at startup, key ids. Keys are compared in constant time (`hmac.compare_digest`) against every configured key.
- **Client IP behind Render's proxy:** uvicorn runs with `--proxy-headers`, so per-IP limits use the address in `X-Forwarded-For`. That header can be spoofed, so per-IP limits are only a speed bump; the real protection is the API key, whose limit is keyed by the verified key id.
- **Failed requests aren't counted:** auth runs before the rate limit, so 401s don't use up a client's budget, and they are cheap because they never reach OpenAI.
