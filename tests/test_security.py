"""API keys, OAuth2/JWT tokens and rate limiting (security.py). OpenAI and RAG are mocked."""

from __future__ import annotations

import time
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from httpx import AsyncClient

import security
from tests.conftest import SAMPLE_LEAD
from tests.openai_fakes import completion_with_json, make_async_openai_client

N8N_KEY = "la_n8n-test-secret-0123456789abcdef"
DEMO_KEY = "la_demo-test-secret-0123456789abcdef"
KEYS = f"n8n:{N8N_KEY}, demo:{DEMO_KEY}"
JWT_SECRET = "test-jwt-secret-not-real-0123456789"
FAKE_RESULT = {"priority": "High", "suggested_response": "Happy to help.", "rationale": "Clear intent."}


@pytest.fixture
def auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", KEYS)
    monkeypatch.setenv("JWT_SECRET", JWT_SECRET)
    from config import get_settings

    get_settings.cache_clear()


@pytest.fixture
def fake_llm(mock_rag) -> Iterator[None]:
    """Every /analyze-lead call gets a valid OpenAI answer."""
    client = make_async_openai_client(lambda *a, **k: completion_with_json(FAKE_RESULT))
    with patch("main._get_async_client", return_value=client):
        yield


def _settings_env(monkeypatch: pytest.MonkeyPatch, **values: str) -> None:
    from config import get_settings

    for name, value in values.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()


async def _token(client: AsyncClient, client_id: str = "n8n", secret: str = N8N_KEY, **extra: str):
    data = {"grant_type": "client_credentials", "client_id": client_id, "client_secret": secret, **extra}
    return await client.post("/auth/token", data=data)


# ---- API_KEYS parsing ------------------------------------------------------------------------


def test_parse_api_keys() -> None:
    keys = security.parse_api_keys(KEYS)
    assert [k.key_id for k in keys] == ["n8n", "demo"]
    assert security.parse_api_keys("") == []
    assert security.parse_api_keys(" , ") == []


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("no-colon-here-0123456789abcdef", "key_id:secret"),
        (":la_0123456789abcdef0123456789", "key_id:secret"),
        ("n8n:short", "shorter than"),
        (f"n8n:{N8N_KEY},n8n:{DEMO_KEY}", "twice"),
    ],
)
def test_parse_api_keys_rejects_bad_config(raw: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        security.parse_api_keys(raw)


def test_match_key() -> None:
    keys = security.parse_api_keys(KEYS)
    assert security.match_key(DEMO_KEY, keys) == "demo"
    assert security.match_key(DEMO_KEY + "x", keys) is None
    assert security.match_key("", keys) is None


def test_cli_new_key_is_valid(capsys: pytest.CaptureFixture[str]) -> None:
    assert security._cli(["new-key", "n8n"]) == 0
    line = capsys.readouterr().out.strip()
    [key] = security.parse_api_keys(line)
    assert key.key_id == "n8n" and key.secret.startswith(security.KEY_PREFIX)
    assert security._cli(["oops"]) == 2


# ---- auth off (no API_KEYS) --------------------------------------------------------------------


async def test_auth_off_allows_requests_without_credentials(client: AsyncClient, fake_llm) -> None:
    assert security.auth_enabled() is False
    resp = await client.post("/analyze-lead", json=SAMPLE_LEAD)
    assert resp.status_code == 200


# ---- API keys ----------------------------------------------------------------------------------


async def test_missing_credentials_401(client: AsyncClient, auth_env, fake_llm) -> None:
    resp = await client.post("/analyze-lead", json=SAMPLE_LEAD)
    assert resp.status_code == 401
    assert resp.json()["error_code"] == "missing_credentials"


async def test_wrong_api_key_401(client: AsyncClient, auth_env, fake_llm) -> None:
    resp = await client.post("/analyze-lead", json=SAMPLE_LEAD, headers={"X-API-Key": "la_wrong-key"})
    assert resp.status_code == 401
    assert resp.json()["error_code"] == "invalid_api_key"


async def test_valid_api_key_200(client: AsyncClient, auth_env, fake_llm) -> None:
    resp = await client.post("/analyze-lead", json=SAMPLE_LEAD, headers={"X-API-Key": DEMO_KEY})
    assert resp.status_code == 200
    assert resp.json()["priority"] == "High"


@pytest.mark.parametrize("path", ["/multi-agent", "/agent-action"])
async def test_every_ai_endpoint_is_protected(client: AsyncClient, auth_env, path: str) -> None:
    resp = await client.post(path, json=SAMPLE_LEAD)
    assert resp.status_code == 401


async def test_health_and_docs_stay_public(client: AsyncClient, auth_env) -> None:
    assert (await client.get("/health")).status_code == 200
    assert (await client.get("/docs")).status_code == 200
    schemes = (await client.get("/openapi.json")).json()["components"]["securitySchemes"]
    assert set(schemes) == {"APIKeyHeader", "HTTPBearer"}


async def test_rotation_old_key_stops_working(
    client: AsyncClient, auth_env, fake_llm, monkeypatch: pytest.MonkeyPatch
) -> None:
    new_key = "la_new-n8n-secret-0123456789abcdef"
    _settings_env(monkeypatch, API_KEYS=f"n8n:{N8N_KEY},n8n-2026:{new_key}")
    for key in (N8N_KEY, new_key):  # overlap: both work while clients switch over
        assert (await client.post("/analyze-lead", json=SAMPLE_LEAD, headers={"X-API-Key": key})).status_code == 200

    _settings_env(monkeypatch, API_KEYS=f"n8n-2026:{new_key}")
    old = await client.post("/analyze-lead", json=SAMPLE_LEAD, headers={"X-API-Key": N8N_KEY})
    assert old.status_code == 401
    new = await client.post("/analyze-lead", json=SAMPLE_LEAD, headers={"X-API-Key": new_key})
    assert new.status_code == 200


# ---- OAuth2 client credentials / JWT -----------------------------------------------------------


async def test_token_flow(client: AsyncClient, auth_env, fake_llm) -> None:
    resp = await _token(client)
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer" and body["expires_in"] == 3600

    headers = {"Authorization": f"Bearer {body['access_token']}"}
    assert (await client.post("/analyze-lead", json=SAMPLE_LEAD, headers=headers)).status_code == 200


@pytest.mark.parametrize(
    ("kwargs", "status", "code"),
    [
        ({"secret": "la_wrong-secret"}, 401, "invalid_client"),
        ({"client_id": "demo"}, 401, "invalid_client"),  # n8n's secret, demo's id
        ({"grant_type": "password"}, 400, "unsupported_grant_type"),
    ],
)
async def test_token_rejections(client: AsyncClient, auth_env, kwargs: dict, status: int, code: str) -> None:
    resp = await _token(client, **kwargs)
    assert resp.status_code == status
    assert resp.json()["error_code"] == code


async def test_token_endpoint_without_jwt_secret(
    client: AsyncClient, auth_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _settings_env(monkeypatch, JWT_SECRET="")
    resp = await _token(client)
    assert resp.status_code == 401
    assert resp.json()["error_code"] == "jwt_disabled"


async def test_expired_tampered_and_revoked_tokens(
    client: AsyncClient, auth_env, fake_llm, monkeypatch: pytest.MonkeyPatch
) -> None:
    expired, _ = security.create_token("n8n", now=time.time() - 7200)
    good, _ = security.create_token("n8n")
    cases = {expired: "token_expired", good[:-4] + "abcd": "invalid_token"}
    for token, code in cases.items():
        resp = await client.post("/analyze-lead", json=SAMPLE_LEAD, headers={"Authorization": f"Bearer {token}"})
        assert (resp.status_code, resp.json()["error_code"]) == (401, code)

    _settings_env(monkeypatch, API_KEYS=f"demo:{DEMO_KEY}")  # n8n key revoked → its tokens too
    resp = await client.post("/analyze-lead", json=SAMPLE_LEAD, headers={"Authorization": f"Bearer {good}"})
    assert (resp.status_code, resp.json()["error_code"]) == (401, "key_revoked")


# ---- rate limiting -----------------------------------------------------------------------------


async def test_rate_limit_per_key(client: AsyncClient, auth_env, fake_llm, monkeypatch: pytest.MonkeyPatch) -> None:
    _settings_env(monkeypatch, RATE_LIMIT="2/minute")
    n8n = {"X-API-Key": N8N_KEY}
    for _ in range(2):
        assert (await client.post("/analyze-lead", json=SAMPLE_LEAD, headers=n8n)).status_code == 200

    limited = await client.post("/analyze-lead", json=SAMPLE_LEAD, headers=n8n)
    assert limited.status_code == 429
    assert limited.json()["error_code"] == "rate_limited"
    assert int(limited.headers["Retry-After"]) > 0

    # Another key has its own budget.
    demo = await client.post("/analyze-lead", json=SAMPLE_LEAD, headers={"X-API-Key": DEMO_KEY})
    assert demo.status_code == 200


async def test_rate_limit_can_be_disabled(
    client: AsyncClient, auth_env, fake_llm, monkeypatch: pytest.MonkeyPatch
) -> None:
    _settings_env(monkeypatch, RATE_LIMIT="1/minute", RATE_LIMIT_ENABLED="false")
    for _ in range(3):
        resp = await client.post("/analyze-lead", json=SAMPLE_LEAD, headers={"X-API-Key": N8N_KEY})
        assert resp.status_code == 200


async def test_token_endpoint_rate_limited_per_ip(
    client: AsyncClient, auth_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _settings_env(monkeypatch, RATE_LIMIT_TOKEN="2/minute")
    assert (await _token(client, secret="la_guess-1")).status_code == 401
    assert (await _token(client, secret="la_guess-2")).status_code == 401
    blocked = await _token(client)  # even the right secret, once the IP is over its limit
    assert blocked.status_code == 429
