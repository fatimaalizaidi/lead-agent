"""Agent tools: OpenAI function schemas + Python implementations.

Stub business logic only — safe for demos / local smoke tests.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from loguru import logger

# ---------------------------------------------------------------------------
# Pricing tiers (USD / seat / month). Documented here for the model + callers.
# Starter / Growth match public list; Enterprise is a negotiated base.
# ---------------------------------------------------------------------------
TIER_BASE_USD: dict[str, float] = {
    "starter": 49.0,
    "growth": 149.0,
    "enterprise": 399.0,
}
VALID_TIERS = frozenset(TIER_BASE_USD)


def calculate_custom_quote(tier: str, seats: int) -> dict[str, Any]:
    """Return a stub monthly quote: ``tier_base × seats``.

    Tiers (USD base per seat / month):
      - starter: 49
      - growth: 149
      - enterprise: 399
    """
    normalized = (tier or "").strip().lower()
    if normalized not in VALID_TIERS:
        return {
            "ok": False,
            "error": f"Unknown tier '{tier}'. Use one of: {sorted(VALID_TIERS)}",
        }
    try:
        seat_count = int(seats)
    except (TypeError, ValueError):
        return {"ok": False, "error": "seats must be an integer"}
    if seat_count < 1:
        return {"ok": False, "error": "seats must be >= 1"}
    if seat_count > 10_000:
        return {"ok": False, "error": "seats exceeds maximum of 10000"}

    base = TIER_BASE_USD[normalized]
    monthly_total = round(base * seat_count, 2)
    logger.info(
        "tool calculate_custom_quote tier={} seats={} monthly_total={}",
        normalized,
        seat_count,
        monthly_total,
    )
    return {
        "ok": True,
        "tier": normalized,
        "seats": seat_count,
        "unit_price_usd": base,
        "monthly_total_usd": monthly_total,
        "currency": "USD",
        "note": "Stub quote — not a binding offer. Enterprise may need custom terms.",
    }


def check_calendar_availability() -> dict[str, Any]:
    """Return fake available demo slots for the next few business days."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    # Next weekday mornings/afternoons (UTC) — deterministic-ish stub.
    slots: list[dict[str, str]] = []
    day = now
    while len(slots) < 4:
        day = day + timedelta(days=1)
        if day.weekday() >= 5:  # skip Sat/Sun
            continue
        for hour in (15, 18):  # 15:00 / 18:00 UTC
            start = day.replace(hour=hour)
            end = start + timedelta(minutes=30)
            slots.append(
                {
                    "start": start.isoformat().replace("+00:00", "Z"),
                    "end": end.isoformat().replace("+00:00", "Z"),
                    "timezone": "UTC",
                }
            )
            if len(slots) >= 4:
                break

    logger.info("tool check_calendar_availability slots={}", len(slots))
    return {
        "ok": True,
        "available_slots": slots,
        "note": "Stub calendar — slots are fake and for demo only.",
    }


def escalate_to_human_support() -> dict[str, Any]:
    """Record a stub escalation and return a ticket id + status."""
    ticket_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"
    logger.info("tool escalate_to_human_support ticket_id={}", ticket_id)
    return {
        "ok": True,
        "ticket_id": ticket_id,
        "status": "queued",
        "queue": "human_support",
        "message": "Escalation recorded. A human will follow up shortly.",
    }


# OpenAI Chat Completions ``tools`` payload (function calling).
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculate_custom_quote",
            "description": (
                "Calculate a monthly custom quote for a pricing tier and seat count. "
                "Use when the customer asks about pricing, quotes, cost, or seats. "
                "Tiers: starter ($49/seat), growth ($149/seat), enterprise ($399/seat)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tier": {
                        "type": "string",
                        "enum": ["starter", "growth", "enterprise"],
                        "description": "Pricing tier name",
                    },
                    "seats": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Number of seats / licenses",
                    },
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
            "description": (
                "Check available calendar slots for a demo or sales call. "
                "Use when the customer wants to schedule a meeting, demo, or call."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human_support",
            "description": (
                "Escalate the conversation to a human support agent. "
                "Use when the customer is frustrated, asks for a person, reports a "
                "billing dispute, legal issue, or the tools cannot help."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "calculate_custom_quote": calculate_custom_quote,
    "check_calendar_availability": check_calendar_availability,
    "escalate_to_human_support": escalate_to_human_support,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool call by name. Unknown tools return an error payload."""
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        logger.warning("tool unknown name={}", name)
        return {"ok": False, "error": f"Unknown tool: {name}"}
    try:
        return handler(**arguments)
    except TypeError as exc:
        logger.warning("tool bad_args name={} err={}", name, type(exc).__name__)
        return {"ok": False, "error": f"Invalid arguments for {name}"}
    except Exception:
        logger.exception("tool failed name={}", name)
        return {"ok": False, "error": f"Tool {name} failed unexpectedly"}
