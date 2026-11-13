"""API authentication (API keys + OAuth2/JWT) and rate limiting for the AI endpoints.

- **API keys:** clients send ``X-API-Key: <secret>``. ``API_KEYS`` holds ``key_id:secret`` pairs,
  so several keys can be live at once: add the new key, move clients over, then remove the old one
  (rotation without downtime). Only the key id ever reaches logs.
- **JWT:** ``POST /auth/token`` (OAuth2 client-credentials grant, client_id = key id,
  client_secret = API key) returns a short-lived HS256 bearer token. A token stops working as soon
  as its key is removed from ``API_KEYS``, so revoking a key revokes its tokens too.
- **Rate limits:** slowapi, per key id on the AI endpoints (per IP when auth is off) and per IP on
  the token endpoint. Over the limit → HTTP 429 in the API's usual error format, with Retry-After.

Auth is off while ``API_KEYS`` is empty, so a deploy never locks clients out before keys exist;
a warning is logged at startup. See docs/security.md.

Generate credentials:  python -m security new-key n8n   |   python -m security new-secret
"""

# No `from __future__ import annotations` here or in main.py: slowapi's decorator wraps the
# endpoints, and FastAPI can't resolve string annotations through the wrapper.
import hmac
import secrets
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Annotated

import jwt
from fastapi import APIRouter, FastAPI, Form, Request, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

from config import get_settings
from errors import AppError, error_body
from observability import REQUEST_ID_HEADER, get_request_id, log_error

API_KEY_HEADER = "X-API-Key"
JWT_ALGORITHM = "HS256"
JWT_ISSUER = "lead-agent"
KEY_PREFIX = "la_"
MIN_SECRET_LENGTH = 24


class UnauthorizedError(AppError):
    status_code = 401
    error_code = "unauthorized"


class InvalidGrantError(AppError):
    status_code = 400
    error_code = "unsupported_grant_type"


@dataclass(frozen=True)
class ApiKey:
    key_id: str
    secret: str


def parse_api_keys(raw: str) -> list[ApiKey]:
    """``"id1:secret1,id2:secret2"`` → keys. Misconfiguration raises, so auth never silently turns off."""
    keys: list[ApiKey] = []
    for item in (part.strip() for part in raw.split(",")):
        if not item:
            continue
        key_id, sep, secret = item.partition(":")
        key_id, secret = key_id.strip(), secret.strip()
        if not sep or not key_id or not secret:
            raise ValueError("API_KEYS entries must look like key_id:secret")
        if len(secret) < MIN_SECRET_LENGTH:
            raise ValueError(f"API key {key_id!r} is shorter than {MIN_SECRET_LENGTH} characters")
        if any(k.key_id == key_id for k in keys):
            raise ValueError(f"API key id {key_id!r} appears twice")
        keys.append(ApiKey(key_id, secret))
    return keys


def active_keys() -> list[ApiKey]:
    return parse_api_keys(get_settings().api_keys)


def auth_enabled() -> bool:
    return bool(active_keys())


def match_key(secret: str, keys: list[ApiKey]) -> str | None:
    """Key id for ``secret``; compares against every key in constant time."""
    found = None
    for key in keys:
        if hmac.compare_digest(key.secret.encode(), secret.encode()):
            found = key.key_id
    return found


def create_token(key_id: str, *, now: float | None = None) -> tuple[str, int]:
    settings = get_settings()
    if not settings.jwt_secret:
        raise UnauthorizedError("Token auth is not configured on this server.", error_code="jwt_disabled")
    issued = int(now if now is not None else time.time())
    ttl = settings.jwt_ttl_seconds
    claims = {"sub": key_id, "iss": JWT_ISSUER, "iat": issued, "exp": issued + ttl, "jti": uuid.uuid4().hex}
    return jwt.encode(claims, settings.jwt_secret, algorithm=JWT_ALGORITHM), ttl


def verify_token(token: str, keys: list[ApiKey]) -> str:
    """Key id from a valid token whose key is still active; raises UnauthorizedError otherwise."""
    secret = get_settings().jwt_secret
    if not secret:
        raise UnauthorizedError("Token auth is not configured on this server.", error_code="jwt_disabled")
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER,
            options={"require": ["exp", "iat", "sub", "iss"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise UnauthorizedError("Token expired.", error_code="token_expired") from exc
    except jwt.PyJWTError as exc:
        raise UnauthorizedError("Invalid token.", error_code="invalid_token") from exc
    key_id = str(claims["sub"])
    if all(k.key_id != key_id for k in keys):
        raise UnauthorizedError("This token's API key was revoked.", error_code="key_revoked")
    return key_id


_api_key_scheme = APIKeyHeader(
    name=API_KEY_HEADER, auto_error=False, description="API key (`key_id:secret` pairs live in API_KEYS)"
)
_bearer_scheme = HTTPBearer(auto_error=False, description="JWT from POST /auth/token")


async def require_client(
    request: Request,
    api_key: Annotated[str | None, Security(_api_key_scheme)] = None,
    bearer: Annotated[HTTPAuthorizationCredentials | None, Security(_bearer_scheme)] = None,
) -> str | None:
    """FastAPI dependency: the caller's key id (``None`` while auth is off).

    401s are logged (error_code only) by the shared AppError handler in errors.py.
    """
    keys = active_keys()
    if not keys:
        request.state.client_id = None
        return None
    if api_key:
        key_id = match_key(api_key, keys)
        if key_id is None:
            raise UnauthorizedError("Invalid API key.", error_code="invalid_api_key")
    elif bearer:
        key_id = verify_token(bearer.credentials, keys)
    else:
        raise UnauthorizedError(
            f"Send an {API_KEY_HEADER} header or an Authorization: Bearer token.",
            error_code="missing_credentials",
        )
    request.state.client_id = key_id
    return key_id


def _client_key(request: Request) -> str:
    """Rate-limit bucket: the authenticated key id, else the client IP."""
    client_id = getattr(request.state, "client_id", None)
    return f"key:{client_id}" if client_id else f"ip:{get_remote_address(request)}"


def _rate_limits_off() -> bool:
    return not get_settings().rate_limit_enabled


limiter = Limiter(key_func=_client_key, storage_uri=get_settings().rate_limit_storage_uri)

# Decorators for endpoints (they need a ``request: Request`` parameter). Limits are read from
# Settings on every request, so changing RATE_LIMIT needs only a restart, no code change.
limit_ai = limiter.limit(lambda: get_settings().rate_limit, exempt_when=_rate_limits_off)
limit_token = limiter.limit(
    lambda: get_settings().rate_limit_token, key_func=get_remote_address, exempt_when=_rate_limits_off
)


async def _rate_limit_handler(_request: Request, exc: RateLimitExceeded) -> JSONResponse:
    log_error(error_code="rate_limited", status=429)
    headers = {"Retry-After": str(exc.limit.limit.get_expiry())}
    if rid := get_request_id():
        headers[REQUEST_ID_HEADER] = rid
    body = error_body(f"Rate limit exceeded ({exc.limit.limit}). Retry later.", "rate_limited")
    return JSONResponse(status_code=429, content=body, headers=headers)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


router = APIRouter(tags=["auth"])


@router.post("/auth/token", response_model=TokenResponse)
@limit_token
async def issue_token(
    request: Request,
    grant_type: Annotated[str, Form(description="Must be client_credentials")],
    client_id: Annotated[str, Form(description="API key id")],
    client_secret: Annotated[str, Form(description="API key")],
) -> TokenResponse:
    """OAuth2 client-credentials grant: trade an API key for a short-lived bearer token."""
    if grant_type != "client_credentials":
        raise InvalidGrantError("Only grant_type=client_credentials is supported.")
    keys = active_keys()
    key_id = match_key(client_secret, keys)
    if key_id is None or key_id != client_id:
        raise UnauthorizedError("Invalid client_id or client_secret.", error_code="invalid_client")
    token, ttl = create_token(key_id)
    return TokenResponse(access_token=token, expires_in=ttl)


def install(app: FastAPI) -> None:
    """Wire the limiter, its 429 handler and /auth/token into the app; warn when auth is off."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)  # type: ignore[arg-type]
    app.include_router(router)
    keys = active_keys()  # raises on a malformed API_KEYS, so a bad config fails the deploy
    if not keys:
        logger.warning("API_KEYS is empty: authentication is OFF. Set it before exposing the API.")
    else:
        logger.info("API auth on: {} key(s) ({})", len(keys), ", ".join(k.key_id for k in keys))


def _cli(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "new-key":
        print(f"{argv[1]}:{KEY_PREFIX}{secrets.token_urlsafe(32)}")
        return 0
    if argv == ["new-secret"]:
        print(secrets.token_urlsafe(48))
        return 0
    print("usage: python -m security new-key <key_id> | new-secret", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
