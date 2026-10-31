"""LangGraph multi-agent supervisor: Sales / Support / Router specialists."""

from __future__ import annotations

import json
import operator
import time
from typing import Annotated, Any, Literal, TypedDict

import httpx
import openai
from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel, EmailStr, Field, field_validator

from config import get_settings
from errors import (
    AppError,
    AuthUpstreamError,
    RateLimitUpstreamError,
    UpstreamError,
)
from guardrails import sanitize_email, sanitize_message
from memory import append_messages, history_for_openai
from observability import elapsed_ms, log_llm
from rag import format_context_for_prompt, knowledge_source_titles, retrieve_context
from tools import (
    calculate_custom_quote,
    check_calendar_availability,
    execute_tool,
)

# Bound supervisor hops so a mis-route cannot loop forever.
MAX_SUPERVISOR_HOPS = 4
MAX_SALES_ITERATIONS = 4
MAX_TOKENS = 256

AgentName = Literal["sales", "support", "router", "finish"]

SUPERVISOR_PROMPT = """You are the Supervisor for Acme LeadDesk multi-agent support.
Route the customer message to exactly one specialist (or finish):
- sales: pricing, quotes, tiers, seats, demos, calendar scheduling
- support: product policy questions (SLA, refunds, onboarding, support hours, docs)
- router: anger/urgency/churn risk, escalation, or requests to notify Slack/email teams
- finish: a clear final reply already exists and no more specialists are needed
Return JSON only: {"next":"sales"|"support"|"router"|"finish","reason":"one short clause"}."""

SALES_PROMPT = """You are the Sales Agent for Acme LeadDesk.
You handle tier/seat quotes and demo scheduling via tools.
Use calculate_custom_quote and/or check_calendar_availability when needed.
After tools, reply in 1–3 short professional sentences. Do not invent prices or slots."""

SUPPORT_PROMPT = """You are the Support Agent for Acme LeadDesk.
Answer using COMPANY KNOWLEDGE only. If knowledge is empty or irrelevant, say briefly
what you do not know and suggest sales or escalation. 1–3 short sentences. No invented policy numbers."""

ROUTER_PROMPT = """You are the Router Agent for Acme LeadDesk.
Analyze sentiment (positive|neutral|negative|urgent) and decide whether to notify teams.
Use trigger_slack_workflow and/or trigger_email_workflow when the customer is angry,
urgent, at churn risk, or explicitly asks for human/ops follow-up.
Then reply in 1–2 short sentences confirming next steps."""


class MultiAgentRequest(BaseModel):
    message: str = Field(..., min_length=1)
    email: EmailStr
    thread_id: str | None = Field(
        default=None,
        description="Optional conversation thread id; server generates UUID when omitted",
    )

    @field_validator("message")
    @classmethod
    def sanitize_message_field(cls, value: str) -> str:
        return sanitize_message(value)

    @field_validator("email")
    @classmethod
    def sanitize_email_field(cls, value: EmailStr) -> str:
        return sanitize_email(str(value))

    @field_validator("thread_id")
    @classmethod
    def strip_thread_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class ToolCallRecord(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)


class NotificationRecord(BaseModel):
    channel: Literal["slack", "email"]
    ok: bool
    detail: str
    stub: bool = True


class MultiAgentResponse(BaseModel):
    email: EmailStr
    agent_reply: str
    route_trace: list[str] = Field(
        default_factory=list,
        description="Ordered specialist agents the supervisor invoked",
    )
    tools_called: list[ToolCallRecord] = Field(default_factory=list)
    knowledge_sources: list[str] = Field(default_factory=list)
    sentiment: str | None = None
    notifications: list[NotificationRecord] = Field(default_factory=list)
    hops: int = Field(..., ge=1)
    thread_id: str


class GraphState(TypedDict):
    email: str
    message: str
    thread_id: str
    next_agent: str
    hops: int
    agent_reply: str
    agents_called: Annotated[list[str], operator.add]
    tools_called: Annotated[list[dict[str, Any]], operator.add]
    knowledge_sources: Annotated[list[str], operator.add]
    sentiment: str | None
    notifications: Annotated[list[dict[str, Any]], operator.add]


def _get_async_client() -> openai.AsyncOpenAI:
    settings = get_settings()
    if not settings.openai_api_key:
        raise AppError(
            "OPENAI_API_KEY is not configured. Set it in `.env` before starting the server.",
            error_code="missing_api_key",
        )
    return openai.AsyncOpenAI(
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout_seconds,
    )


def _map_openai_error(exc: Exception, *, context: str) -> AppError:
    if isinstance(exc, AppError):
        return exc
    if isinstance(exc, openai.AuthenticationError):
        return AuthUpstreamError("OpenAI authentication failed. Check OPENAI_API_KEY.")
    if isinstance(exc, openai.RateLimitError):
        return RateLimitUpstreamError("OpenAI rate limit exceeded. Retry later.")
    if isinstance(exc, openai.APITimeoutError):
        return UpstreamError("OpenAI request timed out.", error_code="openai_timeout")
    if isinstance(exc, openai.APIConnectionError):
        return UpstreamError(
            "Could not reach OpenAI API.", error_code="openai_unreachable"
        )
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code if 400 <= exc.status_code < 600 else 502
        if status == 401:
            return AuthUpstreamError("OpenAI authentication failed.")
        if status == 429:
            return RateLimitUpstreamError("OpenAI rate limit exceeded.")
        err = UpstreamError("OpenAI API returned an error.", error_code="openai_api_error")
        err.status_code = 502 if status >= 500 else status
        return err
    if isinstance(exc, openai.APIError):
        return UpstreamError("OpenAI API error.", error_code="openai_api_error")
    logger.exception("Unexpected multi-agent LLM failure context={}", context)
    return UpstreamError(
        "Multi-agent run failed due to an upstream error.",
        error_code="openai_unexpected",
    )


async def _chat(
    client: openai.AsyncOpenAI,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
) -> Any:
    settings = get_settings()
    started = time.perf_counter()
    kwargs: dict[str, Any] = {
        "model": settings.openai_model,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if response_format:
        kwargs["response_format"] = response_format
    try:
        completion = await client.chat.completions.create(**kwargs)
    except Exception as exc:
        raise _map_openai_error(exc, context="chat") from exc

    usage = getattr(completion, "usage", None)
    log_llm(
        model=settings.openai_model,
        llm_latency_ms=elapsed_ms(started),
        prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        total_tokens=getattr(usage, "total_tokens", None) if usage else None,
    )
    return completion


def _parse_tool_arguments(raw: str | None) -> dict[str, Any]:
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw, "_parse_error": True}
    if isinstance(parsed, dict):
        return parsed
    return {"_value": parsed}


# ---------------------------------------------------------------------------
# Router notification stubs (optional real webhooks via env)
# ---------------------------------------------------------------------------
SALES_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculate_custom_quote",
            "description": (
                "Calculate a monthly custom quote for a pricing tier and seat count. "
                "Tiers: starter ($49), growth ($149), enterprise ($399)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tier": {
                        "type": "string",
                        "enum": ["starter", "growth", "enterprise"],
                    },
                    "seats": {"type": "integer", "minimum": 1},
                },
                "required": ["tier", "seats"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_calendar_availability",
            "description": "Check available demo / sales call slots.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]

ROUTER_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "trigger_slack_workflow",
            "description": (
                "Notify the Slack ops channel (or stub) about an urgent/negative lead. "
                "Use for anger, churn risk, or explicit Slack follow-up."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Short alert summary for Slack",
                    },
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "neutral", "negative", "urgent"],
                    },
                },
                "required": ["summary", "sentiment"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trigger_email_workflow",
            "description": (
                "Trigger an email workflow (or stub) for human follow-up. "
                "Use when a human should email the customer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "neutral", "negative", "urgent"],
                    },
                },
                "required": ["subject", "body", "sentiment"],
                "additionalProperties": False,
            },
        },
    },
]


def trigger_slack_workflow(*, summary: str, sentiment: str) -> dict[str, Any]:
    """POST to SLACK_WEBHOOK_URL when set; otherwise return a stub payload."""
    settings = get_settings()
    url = (settings.slack_webhook_url or "").strip()
    payload = {
        "text": f"[{sentiment}] {summary}",
        "sentiment": sentiment,
        "summary": summary,
    }
    if not url:
        logger.info("slack workflow stub sentiment={}", sentiment)
        return {
            "ok": True,
            "channel": "slack",
            "stub": True,
            "detail": "SLACK_WEBHOOK_URL unset — stub notification recorded",
            "payload": payload,
        }
    try:
        resp = httpx.post(url, json=payload, timeout=10.0)
        ok = 200 <= resp.status_code < 300
        return {
            "ok": ok,
            "channel": "slack",
            "stub": False,
            "detail": f"HTTP {resp.status_code}",
            "payload": payload,
        }
    except Exception as exc:
        logger.warning("slack webhook failed: {}", type(exc).__name__)
        return {
            "ok": False,
            "channel": "slack",
            "stub": False,
            "detail": f"webhook_error:{type(exc).__name__}",
            "payload": payload,
        }


def trigger_email_workflow(*, subject: str, body: str, sentiment: str) -> dict[str, Any]:
    """POST to EMAIL_WEBHOOK_URL when set; otherwise return a stub payload."""
    settings = get_settings()
    url = (settings.email_webhook_url or "").strip()
    payload = {"subject": subject, "body": body, "sentiment": sentiment}
    if not url:
        logger.info("email workflow stub sentiment={}", sentiment)
        return {
            "ok": True,
            "channel": "email",
            "stub": True,
            "detail": "EMAIL_WEBHOOK_URL unset — stub notification recorded",
            "payload": payload,
        }
    try:
        resp = httpx.post(url, json=payload, timeout=10.0)
        ok = 200 <= resp.status_code < 300
        return {
            "ok": ok,
            "channel": "email",
            "stub": False,
            "detail": f"HTTP {resp.status_code}",
            "payload": payload,
        }
    except Exception as exc:
        logger.warning("email webhook failed: {}", type(exc).__name__)
        return {
            "ok": False,
            "channel": "email",
            "stub": False,
            "detail": f"webhook_error:{type(exc).__name__}",
            "payload": payload,
        }


ROUTER_TOOL_HANDLERS = {
    "trigger_slack_workflow": trigger_slack_workflow,
    "trigger_email_workflow": trigger_email_workflow,
}


def _execute_sales_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "calculate_custom_quote":
        return calculate_custom_quote(
            str(arguments.get("tier", "")),
            arguments.get("seats", 0),
        )
    if name == "check_calendar_availability":
        return check_calendar_availability()
    return execute_tool(name, arguments)


def _execute_router_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    handler = ROUTER_TOOL_HANDLERS.get(name)
    if handler is None:
        return {"ok": False, "error": f"Unknown router tool: {name}"}
    try:
        return handler(**arguments)
    except TypeError:
        return {"ok": False, "error": f"Invalid arguments for {name}"}
    except Exception:
        logger.exception("router tool failed name={}", name)
        return {"ok": False, "error": f"Tool {name} failed unexpectedly"}


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------
async def supervisor_node(state: GraphState) -> dict[str, Any]:
    """Choose the next specialist (or finish)."""
    hops = int(state.get("hops") or 0) + 1
    if hops > MAX_SUPERVISOR_HOPS:
        reply = (state.get("agent_reply") or "").strip() or (
            "I can help with quotes, product policy questions, or escalate to our team."
        )
        return {"hops": hops, "next_agent": "finish", "agent_reply": reply}

    client = _get_async_client()
    prior = history_for_openai(state["thread_id"])
    user_blob = (
        f"email: {state['email']}\n"
        f"message: {state['message']}\n"
        f"agents_already_called: {state.get('agents_called') or []}\n"
        f"current_reply: {state.get('agent_reply') or '(none)'}\n"
        f"hop: {hops}/{MAX_SUPERVISOR_HOPS}"
    )
    completion = await _chat(
        client,
        [
            {"role": "system", "content": SUPERVISOR_PROMPT},
            *prior,
            {"role": "user", "content": user_blob},
        ],
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content if completion.choices else None
    next_agent: AgentName = "finish"
    if raw:
        try:
            parsed = json.loads(raw)
            candidate = str(parsed.get("next") or "").strip().lower()
            if candidate in ("sales", "support", "router", "finish"):
                next_agent = candidate  # type: ignore[assignment]
        except json.JSONDecodeError:
            next_agent = "support"

    # Avoid immediately re-calling the same specialist in a tight loop.
    called = state.get("agents_called") or []
    if next_agent != "finish" and called and called[-1] == next_agent:
        next_agent = "finish"

    logger.info("supervisor hop={} next={}", hops, next_agent)
    return {"hops": hops, "next_agent": next_agent}


async def sales_node(state: GraphState) -> dict[str, Any]:
    """Sales specialist — tier/seat quotes + calendar tools."""
    client = _get_async_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SALES_PROMPT},
        *history_for_openai(state["thread_id"]),
        {
            "role": "user",
            "content": f"email: {state['email']}\nmessage: {state['message']}",
        },
    ]
    tools_called: list[dict[str, Any]] = []
    agent_reply = ""

    for _ in range(MAX_SALES_ITERATIONS):
        completion = await _chat(client, messages, tools=SALES_TOOL_DEFINITIONS)
        msg = completion.choices[0].message if completion.choices else None
        if msg is None:
            break
        tool_calls = list(msg.tool_calls or [])
        assistant_payload: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content,
        }
        if tool_calls:
            assistant_payload["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
                for tc in tool_calls
            ]
        messages.append(assistant_payload)
        if not tool_calls:
            agent_reply = (msg.content or "").strip()
            break
        for tc in tool_calls:
            args = _parse_tool_arguments(tc.function.arguments)
            call_args = {k: v for k, v in args.items() if not str(k).startswith("_")}
            if args.get("_parse_error"):
                result: dict[str, Any] = {
                    "ok": False,
                    "error": "Could not parse tool arguments as JSON",
                }
            else:
                result = _execute_sales_tool(tc.function.name, call_args)
            tools_called.append(
                {"name": tc.function.name, "arguments": call_args, "result": result}
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )
    else:
        agent_reply = (
            "I looked up sales options for you. Please review the quote or slots above."
            if tools_called
            else "I can help with custom quotes or demo scheduling."
        )

    if not agent_reply:
        agent_reply = (
            "I can help with custom quotes (starter/growth/enterprise) or demo scheduling."
        )

    return {
        "agents_called": ["sales"],
        "tools_called": tools_called,
        "agent_reply": agent_reply,
        "next_agent": "supervisor",
    }


async def support_node(state: GraphState) -> dict[str, Any]:
    """Support specialist — ChromaDB / company KB via LangChain RAG."""
    chunks: list[dict[str, str]] = []
    try:
        chunks, _score = retrieve_context(state["message"])
    except Exception:
        logger.exception("support RAG retrieve failed")
        chunks = []

    kb_block = format_context_for_prompt(chunks)
    sources = knowledge_source_titles(chunks)
    client = _get_async_client()
    completion = await _chat(
        client,
        [
            {"role": "system", "content": SUPPORT_PROMPT},
            *history_for_openai(state["thread_id"]),
            {
                "role": "user",
                "content": (
                    f"email: {state['email']}\n"
                    f"message: {state['message']}\n\n"
                    f"COMPANY KNOWLEDGE:\n{kb_block}"
                ),
            },
        ],
    )
    msg = completion.choices[0].message if completion.choices else None
    agent_reply = (msg.content or "").strip() if msg else ""
    if not agent_reply:
        agent_reply = (
            "I could not find a matching policy answer in our knowledge base. "
            "Ask about SLA, refunds, onboarding, or support hours — or request a human."
        )
    return {
        "agents_called": ["support"],
        "knowledge_sources": sources,
        "agent_reply": agent_reply,
        "next_agent": "supervisor",
    }


async def router_node(state: GraphState) -> dict[str, Any]:
    """Router specialist — sentiment + Slack/Email workflow triggers."""
    client = _get_async_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": ROUTER_PROMPT},
        *history_for_openai(state["thread_id"]),
        {
            "role": "user",
            "content": (
                f"email: {state['email']}\n"
                f"message: {state['message']}\n"
                f"prior_reply: {state.get('agent_reply') or '(none)'}"
            ),
        },
    ]
    tools_called: list[dict[str, Any]] = []
    notifications: list[dict[str, Any]] = []
    sentiment: str | None = None
    agent_reply = ""

    for _ in range(MAX_SALES_ITERATIONS):
        completion = await _chat(client, messages, tools=ROUTER_TOOL_DEFINITIONS)
        msg = completion.choices[0].message if completion.choices else None
        if msg is None:
            break
        tool_calls = list(msg.tool_calls or [])
        assistant_payload: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content,
        }
        if tool_calls:
            assistant_payload["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
                for tc in tool_calls
            ]
        messages.append(assistant_payload)
        if not tool_calls:
            agent_reply = (msg.content or "").strip()
            break
        for tc in tool_calls:
            args = _parse_tool_arguments(tc.function.arguments)
            call_args = {k: v for k, v in args.items() if not str(k).startswith("_")}
            if args.get("_parse_error"):
                result = {"ok": False, "error": "Could not parse tool arguments as JSON"}
            else:
                result = _execute_router_tool(tc.function.name, call_args)
            tools_called.append(
                {"name": tc.function.name, "arguments": call_args, "result": result}
            )
            sent = str(call_args.get("sentiment") or "").strip().lower() or None
            if sent:
                sentiment = sent
            if result.get("channel") in ("slack", "email"):
                notifications.append(
                    {
                        "channel": result["channel"],
                        "ok": bool(result.get("ok")),
                        "detail": str(result.get("detail") or ""),
                        "stub": bool(result.get("stub", True)),
                    }
                )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )
    else:
        agent_reply = (
            "I've flagged this for our team."
            if notifications
            else "I can escalate urgent issues to Slack or email workflows."
        )

    if not agent_reply:
        agent_reply = "Our team has been notified where needed. Someone will follow up shortly."

    return {
        "agents_called": ["router"],
        "tools_called": tools_called,
        "notifications": notifications,
        "sentiment": sentiment,
        "agent_reply": agent_reply,
        "next_agent": "supervisor",
    }


def _route_after_supervisor(state: GraphState) -> str:
    nxt = (state.get("next_agent") or "finish").strip().lower()
    if nxt in ("sales", "support", "router"):
        return nxt
    return "finish"


def build_multi_agent_graph():
    """Compile the LangGraph supervisor → specialists loop."""
    graph = StateGraph(GraphState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("sales", sales_node)
    graph.add_node("support", support_node)
    graph.add_node("router", router_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "sales": "sales",
            "support": "support",
            "router": "router",
            "finish": END,
        },
    )
    graph.add_edge("sales", "supervisor")
    graph.add_edge("support", "supervisor")
    graph.add_edge("router", "supervisor")
    return graph.compile()


_GRAPH = None


def get_multi_agent_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_multi_agent_graph()
    return _GRAPH


async def run_multi_agent(
    *,
    message: str,
    email: str,
    thread_id: str,
) -> MultiAgentResponse:
    """Execute the LangGraph multi-agent run and persist the conversational turn."""
    graph = get_multi_agent_graph()
    initial: GraphState = {
        "email": email,
        "message": message,
        "thread_id": thread_id,
        "next_agent": "supervisor",
        "hops": 0,
        "agent_reply": "",
        "agents_called": [],
        "tools_called": [],
        "knowledge_sources": [],
        "sentiment": None,
        "notifications": [],
    }
    try:
        final = await graph.ainvoke(initial)
    except AppError:
        raise
    except Exception as exc:
        raise _map_openai_error(exc, context="graph") from exc

    agent_reply = (final.get("agent_reply") or "").strip()
    if not agent_reply:
        agent_reply = (
            "I can help with quotes, policy questions from our docs, "
            "or escalate urgent issues to the team."
        )

    append_messages(
        thread_id,
        [
            {"role": "user", "content": message},
            {"role": "assistant", "content": agent_reply},
        ],
    )

    tools = [
        ToolCallRecord(
            name=str(t.get("name") or ""),
            arguments=dict(t.get("arguments") or {}),
            result=dict(t.get("result") or {}),
        )
        for t in (final.get("tools_called") or [])
    ]
    notes = [
        NotificationRecord(
            channel=n["channel"],  # type: ignore[arg-type]
            ok=bool(n.get("ok")),
            detail=str(n.get("detail") or ""),
            stub=bool(n.get("stub", True)),
        )
        for n in (final.get("notifications") or [])
        if n.get("channel") in ("slack", "email")
    ]

    # Dedupe knowledge sources while preserving order.
    seen: set[str] = set()
    sources: list[str] = []
    for title in final.get("knowledge_sources") or []:
        if title and title not in seen:
            seen.add(title)
            sources.append(title)

    logger.info(
        "multiagent_done hops={} route={} tools={}",
        final.get("hops"),
        final.get("agents_called"),
        [t.name for t in tools],
    )
    return MultiAgentResponse(
        email=email,  # type: ignore[arg-type]
        agent_reply=agent_reply,
        route_trace=list(final.get("agents_called") or []),
        tools_called=tools,
        knowledge_sources=sources,
        sentiment=final.get("sentiment"),
        notifications=notes,
        hops=max(1, int(final.get("hops") or 1)),
        thread_id=thread_id,
    )
