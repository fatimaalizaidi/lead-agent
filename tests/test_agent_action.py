"""POST /agent-action — OpenAI tool loop mocked; local tools still run."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import AsyncClient

from tests.conftest import SAMPLE_AGENT
from tests.openai_fakes import (
    completion_with_content,
    completion_with_tool_calls,
    make_async_openai_client,
)


@pytest.mark.asyncio
async def test_agent_action_quote_tool_loop(
    client: AsyncClient,
    fake_redis,
) -> None:
    """First model turn requests calculate_custom_quote; second returns a reply."""
    first = completion_with_tool_calls(
        [
            (
                "call_quote_1",
                "calculate_custom_quote",
                {"tier": "growth", "seats": 25},
            )
        ]
    )
    second = completion_with_content(
        "Growth for 25 seats is $3,725/month. Let me know if you want a demo."
    )
    openai_client = make_async_openai_client([first, second])

    with patch("agent._get_async_client", return_value=openai_client):
        resp = await client.post("/agent-action", json=SAMPLE_AGENT)

    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == SAMPLE_AGENT["email"]
    assert "3,725" in body["agent_reply"] or "3725" in body["agent_reply"].replace(",", "")
    assert body["iterations"] == 2
    assert len(body["tools_called"]) == 1
    tool = body["tools_called"][0]
    assert tool["name"] == "calculate_custom_quote"
    assert tool["arguments"] == {"tier": "growth", "seats": 25}
    assert tool["result"]["ok"] is True
    assert tool["result"]["monthly_total_usd"] == 3725.0
    assert isinstance(body["thread_id"], str) and body["thread_id"]
    assert openai_client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_agent_action_no_tools(
    client: AsyncClient,
    fake_redis,
) -> None:
    openai_client = make_async_openai_client(
        [
            completion_with_content(
                "I can help with custom quotes, demo scheduling, or escalating to support."
            )
        ]
    )

    with patch("agent._get_async_client", return_value=openai_client):
        resp = await client.post(
            "/agent-action",
            json={
                "message": "What can you help with?",
                "email": "curious@acme.com",
                "thread_id": "agent-thread-1",
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["tools_called"] == []
    assert body["iterations"] == 1
    assert body["thread_id"] == "agent-thread-1"
    assert "quotes" in body["agent_reply"].lower() or "help" in body["agent_reply"].lower()
