"""Lead Analyzer — FastAPI app that classifies leads with OpenAI + local RAG."""

from __future__ import annotations

import asyncio
import json
import time
from enum import Enum
from typing import Any

import openai
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, EmailStr, Field, ValidationError, field_validator

from agent import (
    AgentActionRequest,
    AgentActionResponse,
    run_agent_action,
)
from config import get_settings
from errors import (
    AppError,
    AuthUpstreamError,
    RateLimitUpstreamError,
    UpstreamError,
    register_exception_handlers,
)
from guardrails import sanitize_email, sanitize_message
from observability import (
    RequestIdMiddleware,
    elapsed_ms,
    email_domain_only,
    log_llm,
    log_rag,
    log_request_end,
    log_request_start,
    setup_logging,
)
from rag import (
    format_context_for_prompt,
    knowledge_source_titles,
    retrieve_context,
)

setup_logging()

# Cap output tokens → lower latency + cost. 1–2 short sentences fit easily under this.
MAX_TOKENS = 160

# Keep this short: every token in the system prompt costs time/money on every request.
# Rules that must be machine-enforced live on LeadAnalysisResult (Pydantic → JSON schema).
SYSTEM_PROMPT = """You triage inbound leads. Be decisive and brief.
High = clear buying intent (budget, timeline, seats, urgency).
Medium = interested but exploratory.
Low = vague, spam, or poor fit.
When COMPANY KNOWLEDGE is provided, ground suggested_response in those facts
(pricing, SLA, onboarding, support hours, refunds) when the lead asks about them.
Do not invent policy details that contradict the knowledge. If knowledge is empty
or irrelevant, answer generally without fabricating private numbers.
Write suggested_response as 1–2 short professional sentences. Write rationale as one clause.
Do not add extra keys or markdown."""

class Priority(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class LeadAnalysisResult(BaseModel):
    """LLM output contract — Pydantic schema is sent to OpenAI (strict) and re-checked after."""

    priority: Priority = Field(
        ...,
        description="Exactly one of: High, Medium, Low",
    )
    suggested_response: str = Field(
        ...,
        min_length=1,
        max_length=280,
        description="1–2 short professional reply sentences",
    )
    rationale: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description="One short clause explaining the priority",
    )

    @field_validator("suggested_response", "rationale")
    @classmethod
    def strip_nonempty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class AnalyzeLeadRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User / lead message text")
    email: EmailStr = Field(..., description="Lead email address")

    @field_validator("message")
    @classmethod
    def sanitize_message_field(cls, value: str) -> str:
        # Guardrails raise BadRequestError → HTTP 400 via AppError handler.
        return sanitize_message(value)

    @field_validator("email")
    @classmethod
    def sanitize_email_field(cls, value: EmailStr) -> str:
        return sanitize_email(str(value))


class AnalyzeLeadResponse(BaseModel):
    email: EmailStr
    priority: Priority
    suggested_response: str
    rationale: str
    knowledge_sources: list[str] = Field(
        default_factory=list,
        description="Titles of knowledge-base articles used for this analysis",
    )
    needs_human_review: bool = Field(
        default=False,
        description="True when RAG failed/low-confidence or other soft fallbacks applied",
    )
    fallback_reason: str | None = Field(
        default=None,
        description="Short machine-readable reason when a fallback path was used",
    )


class ErrorBody(BaseModel):
    """n8n-friendly error payload — always includes error_code."""

    detail: str
    error_code: str


app = FastAPI(
    title="Lead Analyzer",
    description=(
        "Classify lead priority and draft a suggested response with OpenAI + local RAG. "
        "Also exposes POST /agent-action for autonomous tool-calling (quote, calendar, escalate)."
    ),
    version="1.5.0",
)
app.add_middleware(RequestIdMiddleware)
register_exception_handlers(app)


def _openai_strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Derive an OpenAI strict json_schema from a Pydantic model."""
    schema = model.model_json_schema()

    def enforce(node: dict[str, Any]) -> None:
        if "properties" in node:
            node["type"] = "object"
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
            # OpenAI rejects $ref siblings like "description"
            for key, prop in list(node["properties"].items()):
                if isinstance(prop, dict) and "$ref" in prop:
                    node["properties"][key] = {"$ref": prop["$ref"]}
        for defn in (node.get("$defs") or {}).values():
            if isinstance(defn, dict):
                enforce(defn)

    enforce(schema)
    return schema


LEAD_ANALYSIS_SCHEMA = _openai_strict_schema(LeadAnalysisResult)


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


def _append_rationale_note(rationale: str, note: str, *, max_len: int = 120) -> str:
    """Append a short human-review note without exceeding LeadAnalysisResult max length."""
    base = (rationale or "").strip().rstrip(".")
    note = note.strip()
    combined = f"{base}. {note}" if base else note
    if len(combined) <= max_len:
        return combined
    # Prefer keeping the note if we must truncate.
    keep = max_len - len(note) - 2  # ". "
    if keep < 8:
        return note[:max_len]
    return f"{base[:keep].rstrip()}. {note}"[:max_len]


async def _retrieve_knowledge(
    message: str,
) -> tuple[list[dict[str, str]], bool, bool, float | None]:
    """
    Run sync Chroma/LangChain retrieval off the event loop.

    Returns ``(chunks, rag_unavailable, low_confidence, best_score)``.
    RAG failures do **not** raise — caller continues without KB (HTTP 200 + High review).
    Missing API key for embeddings still raises (config error, not soft fallback).
    """
    settings = get_settings()
    started = time.perf_counter()
    try:
        chunks, best_score = await asyncio.to_thread(
            retrieve_context, message, settings.rag_k
        )
    except RuntimeError as exc:
        # Missing API key — hard config failure (safe message, no key material).
        logger.error("RAG config error: {}", type(exc).__name__)
        raise AppError(
            "OPENAI_API_KEY is not configured for embeddings.",
            error_code="missing_api_key",
        ) from exc
    except Exception:
        # Soft fallback: continue without KB context.
        chroma_latency_ms = elapsed_ms(started)
        log_rag(
            chroma_latency_ms=chroma_latency_ms,
            hit_count=0,
            best_score=None,
            low_confidence=False,
            rag_unavailable=True,
        )
        logger.exception("RAG retrieve failed; continuing without knowledge context")
        return [], True, False, None

    chroma_latency_ms = elapsed_ms(started)
    low_confidence = False
    if best_score is not None and best_score > settings.rag_score_threshold:
        # L2 distance: higher than threshold = weaker match → human review.
        low_confidence = True

    log_rag(
        chroma_latency_ms=chroma_latency_ms,
        hit_count=len(chunks),
        best_score=best_score,
        low_confidence=low_confidence,
        rag_unavailable=False,
    )
    return chunks, False, low_confidence, best_score


async def _analyze_with_openai(
    message: str,
    email: str,
    knowledge_chunks: list[dict[str, str]],
) -> LeadAnalysisResult:
    """Call OpenAI with strict JSON schema. Hard failures → UpstreamError (HTTP 502)."""
    client = _get_async_client()
    settings = get_settings()
    kb_block = format_context_for_prompt(knowledge_chunks)
    # Short user message = faster. Schema (not prose) enforces the response shape.
    user_content = (
        f"email: {email}\n"
        f"message: {message}\n\n"
        f"COMPANY KNOWLEDGE (retrieved; use when relevant):\n{kb_block}"
    )

    started = time.perf_counter()
    try:
        completion = await client.chat.completions.create(
            model=settings.openai_model,
            temperature=0,  # deterministic → less waffle, usually faster to settle
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "lead_analysis",
                    "strict": True,  # OpenAI must match LeadAnalysisResult schema
                    "schema": LEAD_ANALYSIS_SCHEMA,
                },
            },
        )
    except openai.AuthenticationError as exc:
        logger.warning("OpenAI auth failed: {}", type(exc).__name__)
        raise AuthUpstreamError(
            "OpenAI authentication failed. Check OPENAI_API_KEY.",
        ) from exc
    except openai.RateLimitError as exc:
        logger.warning("OpenAI rate limit: {}", type(exc).__name__)
        raise RateLimitUpstreamError(
            "OpenAI rate limit exceeded. Retry later.",
        ) from exc
    except openai.APITimeoutError as exc:
        logger.warning("OpenAI timeout: {}", type(exc).__name__)
        raise UpstreamError(
            "OpenAI request timed out.",
            error_code="openai_timeout",
        ) from exc
    except openai.APIConnectionError as exc:
        logger.warning("OpenAI connection error: {}", type(exc).__name__)
        raise UpstreamError(
            "Could not reach OpenAI API.",
            error_code="openai_unreachable",
        ) from exc
    except openai.APIStatusError as exc:
        status = exc.status_code if 400 <= exc.status_code < 600 else 502
        logger.warning("OpenAI API status {}: {}", status, type(exc).__name__)
        # Map to UpstreamError / Auth / RateLimit as appropriate; avoid leaking bodies.
        if status == 401:
            raise AuthUpstreamError("OpenAI authentication failed.") from exc
        if status == 429:
            raise RateLimitUpstreamError("OpenAI rate limit exceeded.") from exc
        err = UpstreamError("OpenAI API returned an error.", error_code="openai_api_error")
        err.status_code = 502 if status >= 500 else status
        raise err from exc
    except openai.APIError as exc:
        logger.warning("OpenAI API error: {}", type(exc).__name__)
        raise UpstreamError("OpenAI API error.", error_code="openai_api_error") from exc
    except AppError:
        raise
    except Exception as exc:
        logger.exception("Unexpected LLM failure")
        raise UpstreamError(
            "Lead analysis failed due to an upstream error.",
            error_code="openai_unexpected",
        ) from exc

    llm_latency_ms = elapsed_ms(started)
    usage = getattr(completion, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None
    # Token counts only — never log prompts, messages, or API keys.
    log_llm(
        model=settings.openai_model,
        llm_latency_ms=llm_latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )

    raw = completion.choices[0].message.content if completion.choices else None
    if not raw:
        raise UpstreamError(
            "OpenAI did not return a structured lead analysis.",
            error_code="openai_empty_response",
        )

    try:
        return LeadAnalysisResult.model_validate(json.loads(raw))
    except json.JSONDecodeError as exc:
        raise UpstreamError(
            "OpenAI returned invalid JSON.",
            error_code="openai_invalid_json",
        ) from exc
    except ValidationError as exc:
        raise UpstreamError(
            "OpenAI returned an analysis that failed validation.",
            error_code="openai_schema_mismatch",
        ) from exc


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/analyze-lead",
    response_model=AnalyzeLeadResponse,
    responses={
        400: {"model": ErrorBody},
        401: {"model": ErrorBody},
        429: {"model": ErrorBody},
        500: {"model": ErrorBody},
        502: {"model": ErrorBody},
    },
)
async def analyze_lead(body: AnalyzeLeadRequest) -> AnalyzeLeadResponse:
    """Classify lead priority and generate a suggested response via OpenAI + RAG.

    Async so concurrent n8n webhooks can stay in flight: each request awaits OpenAI
    (and offloads sync RAG via to_thread) instead of blocking the whole server on one call.

    Soft fallbacks (RAG unavailable / low confidence) return HTTP 200 with
    ``needs_human_review=true`` and ``priority=High``. Hard LLM failures return HTTP 502
    so n8n can retry.
    """
    started = time.perf_counter()
    # body.message / body.email already sanitized by Pydantic validators (guardrails).
    # Log domain only — never dump full email/message/PII into structured logs.
    log_request_start(
        path="/analyze-lead",
        email_domain=email_domain_only(str(body.email)),
    )

    try:
        chunks, rag_unavailable, low_confidence, _best_score = await _retrieve_knowledge(
            body.message
        )

        # If RAG is down, call LLM without KB; still require a successful model response.
        try:
            result = await _analyze_with_openai(
                message=body.message,
                email=str(body.email),
                knowledge_chunks=[] if rag_unavailable else chunks,
            )
        except AppError:
            raise
        except Exception as exc:
            # Belt-and-suspenders: anything escaping _analyze_with_openai → 502, not 500 leak.
            logger.exception("LLM path failed unexpectedly")
            raise UpstreamError(
                "Lead analysis failed due to an upstream error.",
                error_code="openai_unexpected",
            ) from exc

        needs_human_review = False
        fallback_reason: str | None = None
        priority = result.priority
        rationale = result.rationale
        sources = [] if rag_unavailable else knowledge_source_titles(chunks)

        if rag_unavailable:
            needs_human_review = True
            fallback_reason = "rag_unavailable"
            priority = Priority.HIGH
            rationale = _append_rationale_note(
                rationale,
                "KB unavailable — manual human review required.",
            )
        elif low_confidence:
            needs_human_review = True
            fallback_reason = "low_rag_confidence"
            priority = Priority.HIGH
            rationale = _append_rationale_note(
                rationale,
                "Low KB match confidence — manual human review advised.",
            )

        log_request_end(
            status=200,
            total_ms=elapsed_ms(started),
            needs_human_review=needs_human_review,
        )
        return AnalyzeLeadResponse(
            email=body.email,  # type: ignore[arg-type]
            priority=priority,
            suggested_response=result.suggested_response,
            rationale=rationale,
            knowledge_sources=sources,
            needs_human_review=needs_human_review,
            fallback_reason=fallback_reason,
        )
    except AppError as exc:
        log_request_end(
            status=exc.status_code,
            total_ms=elapsed_ms(started),
            needs_human_review=None,
        )
        raise


@app.post(
    "/agent-action",
    response_model=AgentActionResponse,
    responses={
        400: {"model": ErrorBody},
        401: {"model": ErrorBody},
        429: {"model": ErrorBody},
        500: {"model": ErrorBody},
        502: {"model": ErrorBody},
    },
)
async def agent_action(body: AgentActionRequest) -> AgentActionResponse:
    """Run an autonomous agent turn with OpenAI tool calling.

    The model chooses among ``calculate_custom_quote``, ``check_calendar_availability``,
    and ``escalate_to_human_support``. Tools are real Python functions; results are
    returned with a short ``agent_reply``. Hard LLM failures → HTTP 502 for n8n retry.
    """
    started = time.perf_counter()
    log_request_start(
        path="/agent-action",
        email_domain=email_domain_only(str(body.email)),
    )

    try:
        result = await run_agent_action(
            message=body.message,
            email=str(body.email),
        )
        log_request_end(
            status=200,
            total_ms=elapsed_ms(started),
            needs_human_review=None,
        )
        return result
    except AppError as exc:
        log_request_end(
            status=exc.status_code,
            total_ms=elapsed_ms(started),
            needs_human_review=None,
        )
        raise
    except Exception as exc:
        logger.exception("Agent path failed unexpectedly")
        log_request_end(
            status=502,
            total_ms=elapsed_ms(started),
            needs_human_review=None,
        )
        raise UpstreamError(
            "Agent action failed due to an upstream error.",
            error_code="openai_unexpected",
        ) from exc
