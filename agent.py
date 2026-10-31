"""OpenAI tool-calling agent loop for POST /agent-action."""

from __future__ import annotations

import json
import time
from typing import Any

import openai
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
from tools import TOOL_DEFINITIONS, execute_tool

# Cap agent loop iterations (model → tool → model …) to bound latency/cost.
MAX_AGENT_ITERATIONS = 5
# Final assistant replies stay short for n8n / chat UIs.
MAX_AGENT_TOKENS = 256

AGENT_SYSTEM_PROMPT = """You are an autonomous sales/support agent for Acme LeadDesk.
You have tools for quotes, calendar availability, and human escalation.
Choose tools based on the customer's intent. You may call multiple tools across turns.
When earlier user/assistant turns are present, continue that conversation (reuse names,
seat counts, and prior tool outcomes the customer already received).
After tool results, reply with a short professional message (1–3 sentences).
Do not invent prices, slots, or ticket ids — only use tool results.
If the request is unrelated to your tools, say briefly what you can help with."""


class AgentActionRequest(BaseModel):
    """Same shape as AnalyzeLeadRequest for consistent n8n payloads."""

    message: str = Field(..., min_length=1, description="Customer / lead message text")
    email: EmailStr = Field(..., description="Customer email address")
    thread_id: str | None = Field(
        default=None,
        description="Optional conversation thread id; server generates a UUID when omitted",
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
    """One executed tool invocation from the agent loop."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)


class AgentActionResponse(BaseModel):
    email: EmailStr
    agent_reply: str = Field(
        ...,
        description="Short natural-language reply after tool use (or clarification)",
    )
    tools_called: list[ToolCallRecord] = Field(
        default_factory=list,
        description="Tools the model chose and that were executed, in order",
    )
    iterations: int = Field(
        ...,
        ge=1,
        description="Number of model turns used in the agent loop",
    )
    thread_id: str = Field(
        ...,
        description="Conversation thread id (echoed or newly generated)",
    )


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


async def _chat_with_tools(
    client: openai.AsyncOpenAI,
    messages: list[dict[str, Any]],
) -> Any:
    """One OpenAI chat turn with tools attached. Maps API errors to AppError subclasses."""
    settings = get_settings()
    started = time.perf_counter()
    try:
        completion = await client.chat.completions.create(
            model=settings.openai_model,
            temperature=0,
            max_tokens=MAX_AGENT_TOKENS,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
        )
    except openai.AuthenticationError as exc:
        logger.warning("OpenAI auth failed (agent): {}", type(exc).__name__)
        raise AuthUpstreamError(
            "OpenAI authentication failed. Check OPENAI_API_KEY.",
        ) from exc
    except openai.RateLimitError as exc:
        logger.warning("OpenAI rate limit (agent): {}", type(exc).__name__)
        raise RateLimitUpstreamError(
            "OpenAI rate limit exceeded. Retry later.",
        ) from exc
    except openai.APITimeoutError as exc:
        logger.warning("OpenAI timeout (agent): {}", type(exc).__name__)
        raise UpstreamError(
            "OpenAI request timed out.",
            error_code="openai_timeout",
        ) from exc
    except openai.APIConnectionError as exc:
        logger.warning("OpenAI connection error (agent): {}", type(exc).__name__)
        raise UpstreamError(
            "Could not reach OpenAI API.",
            error_code="openai_unreachable",
        ) from exc
    except openai.APIStatusError as exc:
        status = exc.status_code if 400 <= exc.status_code < 600 else 502
        logger.warning("OpenAI API status {} (agent): {}", status, type(exc).__name__)
        if status == 401:
            raise AuthUpstreamError("OpenAI authentication failed.") from exc
        if status == 429:
            raise RateLimitUpstreamError("OpenAI rate limit exceeded.") from exc
        err = UpstreamError("OpenAI API returned an error.", error_code="openai_api_error")
        err.status_code = 502 if status >= 500 else status
        raise err from exc
    except openai.APIError as exc:
        logger.warning("OpenAI API error (agent): {}", type(exc).__name__)
        raise UpstreamError("OpenAI API error.", error_code="openai_api_error") from exc
    except AppError:
        raise
    except Exception as exc:
        logger.exception("Unexpected agent LLM failure")
        raise UpstreamError(
            "Agent action failed due to an upstream error.",
            error_code="openai_unexpected",
        ) from exc

    llm_latency_ms = elapsed_ms(started)
    usage = getattr(completion, "usage", None)
    log_llm(
        model=settings.openai_model,
        llm_latency_ms=llm_latency_ms,
        prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        total_tokens=getattr(usage, "total_tokens", None) if usage else None,
    )
    return completion


async def run_agent_action(
    *,
    message: str,
    email: str,
    thread_id: str,
) -> AgentActionResponse:
    """Multi-step tool-calling loop until a final message or iteration cap."""
    client = _get_async_client()
    prior = history_for_openai(thread_id)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        *prior,
        {
            "role": "user",
            "content": f"email: {email}\nmessage: {message}",
        },
    ]
    tools_called: list[ToolCallRecord] = []
    iterations = 0
    agent_reply = ""

    for _ in range(MAX_AGENT_ITERATIONS):
        iterations += 1
        completion = await _chat_with_tools(client, messages)
        choice = completion.choices[0] if completion.choices else None
        if choice is None or choice.message is None:
            raise UpstreamError(
                "OpenAI did not return an agent response.",
                error_code="openai_empty_response",
            )

        msg = choice.message
        tool_calls = list(msg.tool_calls or [])

        # Append assistant turn (including tool_calls) for the next round.
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

        # Execute each tool and feed results back.
        for tc in tool_calls:
            name = tc.function.name
            args = _parse_tool_arguments(tc.function.arguments)
            # Drop parse metadata before calling the handler when possible.
            call_args = {
                k: v for k, v in args.items() if not str(k).startswith("_")
            }
            if args.get("_parse_error"):
                result: dict[str, Any] = {
                    "ok": False,
                    "error": "Could not parse tool arguments as JSON",
                }
            else:
                result = execute_tool(name, call_args)

            tools_called.append(
                ToolCallRecord(name=name, arguments=call_args, result=result)
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )
    else:
        # Hit iteration cap while still requesting tools — ask for a closing reply.
        logger.warning(
            "agent iteration_cap reached={} tools_called={}",
            MAX_AGENT_ITERATIONS,
            len(tools_called),
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "Stop calling tools. Summarize what you did for the customer "
                    "in 1–3 short sentences using only the tool results you already have."
                ),
            }
        )
        iterations += 1
        completion = await _chat_with_tools(client, messages)
        choice = completion.choices[0] if completion.choices else None
        agent_reply = (
            (choice.message.content or "").strip()
            if choice and choice.message
            else ""
        )

    if not agent_reply:
        if tools_called:
            agent_reply = (
                "I ran the requested action(s). "
                "Please review the tool results for details."
            )
        else:
            agent_reply = (
                "I can help with custom quotes, demo scheduling, "
                "or escalating to human support."
            )

    # Persist conversational turn only (not intermediate tool messages).
    append_messages(
        thread_id,
        [
            {"role": "user", "content": message},
            {"role": "assistant", "content": agent_reply},
        ],
    )

    logger.info(
        "agent_done iterations={} tools_called={}",
        iterations,
        [t.name for t in tools_called],
    )
    return AgentActionResponse(
        email=email,  # type: ignore[arg-type]
        agent_reply=agent_reply,
        tools_called=tools_called,
        iterations=iterations,
        thread_id=thread_id,
    )
