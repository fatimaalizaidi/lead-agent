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


# ---------------------------------------------------------------------------
# Routing: policy questions → knowledge, quotes/demos → sales, mixed → both
# ---------------------------------------------------------------------------
def _sent_messages(openai_client: MagicMock, call_index: int) -> list[dict]:
    return openai_client.chat.completions.create.call_args_list[call_index].kwargs[
        "messages"
    ]


def test_supervisor_prompt_routes_policy_to_knowledge() -> None:
    """The routing rules name policy topics under knowledge and keep sales narrow."""
    from multiagent import SALES_PROMPT, SUPERVISOR_PROMPT

    knowledge_line = SUPERVISOR_PROMPT.split("- knowledge:")[1].split("- sales:")[0]
    for topic in ("volume discounts", "Sev-1", "refunds", "support hours", "onboarding"):
        assert topic in knowledge_line
    assert "volume discounts start" in SUPERVISOR_PROMPT  # worked example
    assert "Mixed messages" in SUPERVISOR_PROMPT
    assert "Knowledge Agent handles those" in SALES_PROMPT


@pytest.mark.parametrize(
    ("prior", "reply", "expected"),
    [
        (None, "Quote.", "Quote."),
        ("", "Quote.", "Quote."),
        ("Policy.", "Quote.", "Policy.\n\nQuote."),
        ("Policy.", "Policy. Quote.", "Policy. Quote."),
    ],
)
def test_combine_replies(prior: str | None, reply: str, expected: str) -> None:
    from multiagent import _combine_replies

    assert _combine_replies(prior, reply) == expected


@pytest.mark.parametrize(
    ("next_agent", "expected"),
    [
        ("knowledge", "knowledge"),
        ("sales", "sales"),
        ("escalation", "escalation"),
        ("finish", "finish"),
        ("KNOWLEDGE ", "knowledge"),
        ("bogus", "finish"),
        ("", "finish"),
    ],
)
def test_route_after_supervisor_honours_decision(next_agent: str, expected: str) -> None:
    from multiagent import _route_after_supervisor

    assert _route_after_supervisor({"next_agent": next_agent}) == expected  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_multi_agent_policy_question_reaches_knowledge(
    client: AsyncClient,
    fake_redis,
    mock_rag_multiagent,
) -> None:
    """The traced failure: volume discounts + Sev-1 must hit knowledge, not sales."""
    message = "When do volume discounts start, and what is your Sev-1 response time?"
    openai_client = make_async_openai_client(
        [
            completion_with_json({"next": "knowledge", "reason": "pricing policy + SLA"}),
            completion_with_content(
                "Volume discounts start at 50 Growth seats; Sev-1 response is 1 hour."
            ),
            completion_with_json({"next": "finish", "reason": "answered"}),
        ]
    )

    with patch("multiagent._get_async_client", return_value=openai_client):
        resp = await client.post(
            "/multi-agent", json={"message": message, "email": "lead@acme.com"}
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["route_trace"] == ["knowledge"]
    assert body["tools_called"] == []
    assert body["knowledge_sources"] == ["Pricing Plans"]
    mock_rag_multiagent.assert_called_once_with(message)


@pytest.mark.asyncio
async def test_multi_agent_mixed_message_routes_knowledge_then_sales(
    client: AsyncClient,
    fake_redis,
    mock_rag_multiagent,
) -> None:
    """Mixed policy + quote: knowledge answers first, sales sees its reply and merges."""
    knowledge_answer = "Refunds are available within 14 days on annual plans."
    openai_client = make_async_openai_client(
        [
            completion_with_json({"next": "knowledge", "reason": "refund policy part"}),
            completion_with_content(knowledge_answer),
            completion_with_json({"next": "sales", "reason": "quote still needed"}),
            completion_with_tool_calls(
                [("call_q", "calculate_custom_quote", {"tier": "starter", "seats": 10})]
            ),
            completion_with_content("Starter for 10 seats is $490/month."),
            completion_with_json({"next": "finish", "reason": "both parts answered"}),
        ]
    )

    with patch("multiagent._get_async_client", return_value=openai_client):
        resp = await client.post(
            "/multi-agent",
            json={
                "message": "What's your refund policy, and quote Starter for 10 seats?",
                "email": "mixed@acme.com",
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["route_trace"] == ["knowledge", "sales"]
    assert [t["name"] for t in body["tools_called"]] == ["calculate_custom_quote"]
    assert "refund" in body["agent_reply"].lower()
    # Code keeps knowledge's answer ahead of the sales part.
    assert body["agent_reply"] == f"{knowledge_answer}\n\nStarter for 10 seats is $490/month."
    mock_rag_multiagent.assert_called_once()

    # Second supervisor hop sees knowledge already called and its reply.
    supervisor_hop2 = _sent_messages(openai_client, 2)[-1]["content"]
    assert "agents_already_called: ['knowledge']" in supervisor_hop2
    assert knowledge_answer in supervisor_hop2
    # Sales is handed the knowledge reply so it can combine both parts.
    sales_user = next(
        m["content"] for m in _sent_messages(openai_client, 3) if m["role"] == "user"
    )
    assert f"prior_reply: {knowledge_answer}" in sales_user


@pytest.mark.asyncio
async def test_multi_agent_sales_can_hand_back_to_knowledge(
    client: AsyncClient,
    fake_redis,
    mock_rag_multiagent,
) -> None:
    """If the supervisor picks sales first, a later knowledge decision is still honoured."""
    openai_client = make_async_openai_client(
        [
            completion_with_json({"next": "sales", "reason": "quote"}),
            completion_with_content("Growth for 60 seats is $8,940/month."),
            completion_with_json({"next": "knowledge", "reason": "discount policy unanswered"}),
            completion_with_content("Volume discounts start at 50 Growth seats."),
            completion_with_json({"next": "finish", "reason": "done"}),
        ]
    )

    with patch("multiagent._get_async_client", return_value=openai_client):
        resp = await client.post(
            "/multi-agent",
            json={
                "message": "Quote Growth for 60 seats — do volume discounts apply?",
                "email": "volume@acme.com",
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["route_trace"] == ["sales", "knowledge"]
    assert body["agent_reply"] == (
        "Growth for 60 seats is $8,940/month.\n\nVolume discounts start at 50 Growth seats."
    )
    mock_rag_multiagent.assert_called_once()
