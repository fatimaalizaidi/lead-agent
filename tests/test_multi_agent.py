"""POST /multi-agent — LangGraph supervisor with mocked LLM / webhooks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient

from tests.conftest import SAMPLE_MULTI
from tests.openai_fakes import (
    completion_with_content,
    completion_with_json,
    completion_with_tool_calls,
    make_async_openai_client,
)


@pytest.mark.asyncio
async def test_multi_agent_knowledge_route(
    client: AsyncClient,
    fake_redis,
    mock_rag_multiagent,
) -> None:
    """Supervisor → knowledge → finish; RAG + OpenAI mocked."""
    supervisor_to_knowledge = completion_with_json(
        {"next": "knowledge", "reason": "policy question about refunds"}
    )
    knowledge_reply = completion_with_content(
        "Annual plans can be refunded within 14 days per our refund policy."
    )
    supervisor_finish = completion_with_json(
        {"next": "finish", "reason": "knowledge already answered"}
    )
    openai_client = make_async_openai_client(
        [supervisor_to_knowledge, knowledge_reply, supervisor_finish]
    )

    with patch("multiagent._get_async_client", return_value=openai_client):
        resp = await client.post("/multi-agent", json=SAMPLE_MULTI)

    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == SAMPLE_MULTI["email"]
    assert "refund" in body["agent_reply"].lower()
    assert body["route_trace"] == ["knowledge"]
    assert body["knowledge_sources"] == ["Pricing Plans"]
    assert body["hops"] >= 1
    assert isinstance(body["thread_id"], str) and body["thread_id"]
    mock_rag_multiagent.assert_called()


@pytest.mark.asyncio
async def test_multi_agent_sales_with_quote_tool(
    client: AsyncClient,
    fake_redis,
) -> None:
    supervisor_to_sales = completion_with_json(
        {"next": "sales", "reason": "pricing quote request"}
    )
    sales_tool = completion_with_tool_calls(
        [
            (
                "call_sales_1",
                "calculate_custom_quote",
                {"tier": "starter", "seats": 10},
            )
        ]
    )
    sales_reply = completion_with_content(
        "Starter for 10 seats is $490/month. I can also book a demo."
    )
    supervisor_finish = completion_with_json(
        {"next": "finish", "reason": "quote delivered"}
    )
    openai_client = make_async_openai_client(
        [supervisor_to_sales, sales_tool, sales_reply, supervisor_finish]
    )

    with patch("multiagent._get_async_client", return_value=openai_client):
        resp = await client.post(
            "/multi-agent",
            json={
                "message": "Quote starter for 10 seats please",
                "email": "sales@acme.com",
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["route_trace"] == ["sales"]
    assert len(body["tools_called"]) == 1
    assert body["tools_called"][0]["name"] == "calculate_custom_quote"
    assert body["tools_called"][0]["result"]["ok"] is True
    assert body["tools_called"][0]["result"]["monthly_total_usd"] == 490.0


@pytest.mark.asyncio
async def test_multi_agent_escalation_stubs_webhooks(
    client: AsyncClient,
    fake_redis,
) -> None:
    """Escalation tools use stubs when webhook URLs are empty — no live Slack/email."""
    supervisor_to_escalation = completion_with_json(
        {"next": "escalation", "reason": "angry customer needs escalation"}
    )
    escalation_tools = completion_with_tool_calls(
        [
            (
                "call_slack_1",
                "trigger_slack_workflow",
                {
                    "summary": "Customer is furious about billing",
                    "sentiment": "urgent",
                },
            ),
            (
                "call_email_1",
                "trigger_email_workflow",
                {
                    "subject": "Urgent follow-up",
                    "body": "Please call the customer today.",
                    "sentiment": "urgent",
                },
            ),
        ]
    )
    escalation_reply = completion_with_content(
        "I've flagged this for Slack and email follow-up. A human will reach out shortly."
    )
    supervisor_finish = completion_with_json(
        {"next": "finish", "reason": "escalation done"}
    )
    openai_client = make_async_openai_client(
        [supervisor_to_escalation, escalation_tools, escalation_reply, supervisor_finish]
    )

    # Guard against accidental live HTTP even if env is mis-set.
    fake_post = MagicMock()
    with (
        patch("multiagent._get_async_client", return_value=openai_client),
        patch("multiagent.httpx.post", fake_post),
    ):
        resp = await client.post(
            "/multi-agent",
            json={
                "message": "This is unacceptable! Cancel everything and call me NOW.",
                "email": "angry@acme.com",
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["route_trace"] == ["escalation"]
    assert body["sentiment"] == "urgent"
    assert len(body["notifications"]) == 2
    channels = {n["channel"] for n in body["notifications"]}
    assert channels == {"slack", "email"}
    assert all(n["stub"] is True for n in body["notifications"])
    assert all(n["ok"] is True for n in body["notifications"])
    fake_post.assert_not_called()


@pytest.mark.asyncio
async def test_multi_agent_escalation_mocked_webhook_urls(
    client: AsyncClient,
    fake_redis,
    monkeypatch,
) -> None:
    """When webhook URLs are set, httpx.post is mocked — still no live alerts."""
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.example.test/slack")
    monkeypatch.setenv("EMAIL_WEBHOOK_URL", "https://hooks.example.test/email")
    from config import get_settings

    get_settings.cache_clear()

    supervisor_to_escalation = completion_with_json(
        {"next": "escalation", "reason": "notify ops"}
    )
    escalation_tools = completion_with_tool_calls(
        [
            (
                "call_slack_2",
                "trigger_slack_workflow",
                {"summary": "Churn risk", "sentiment": "negative"},
            )
        ]
    )
    escalation_reply = completion_with_content("Ops has been notified on Slack.")
    supervisor_finish = completion_with_json(
        {"next": "finish", "reason": "done"}
    )
    openai_client = make_async_openai_client(
        [supervisor_to_escalation, escalation_tools, escalation_reply, supervisor_finish]
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with (
        patch("multiagent._get_async_client", return_value=openai_client),
        patch("multiagent.httpx.post", return_value=mock_resp) as fake_post,
    ):
        resp = await client.post(
            "/multi-agent",
            json={
                "message": "We're switching vendors unless someone calls today.",
                "email": "churn@acme.com",
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["route_trace"] == ["escalation"]
    assert len(body["notifications"]) == 1
    assert body["notifications"][0]["channel"] == "slack"
    assert body["notifications"][0]["stub"] is False
    assert body["notifications"][0]["ok"] is True
    fake_post.assert_called_once()
    assert fake_post.call_args.args[0] == "https://hooks.example.test/slack"
