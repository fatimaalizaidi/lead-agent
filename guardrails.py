"""Input guardrails: sanitize email + message before RAG / LLM.

Rejects oversized or prompt-injection-looking input with BadRequestError (HTTP 400).
Applied during Pydantic validation so bad payloads never reach retrieve/LLM paths.
"""

from __future__ import annotations

import re
import unicodedata

from errors import BadRequestError

# Soft caps — reject (do not silently truncate) so n8n sees a clear 400.
MAX_MESSAGE_CHARS = 6000
MAX_EMAIL_CHARS = 254

# Obvious jailbreak / instruction-override patterns (case-insensitive).
# Keep this list short and high-precision; do not over-block legitimate business text.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
        r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
        r"forget\s+(all\s+)?(previous|prior|above)\s+instructions?",
        r"you\s+are\s+now\s+(dan|jailbroken|unrestricted)",
        r"\bsystem\s*:\s*",
        r"\bassistant\s*:\s*",
        r"\[?\s*INST\s*\]?",
        r"<\s*/?\s*system\s*>",
        r"role\s*-?\s*play\s+as\s+(an?\s+)?unrestricted",
        r"override\s+(your\s+)?(safety|system)\s+(rules?|prompt)",
    )
]

# Control chars except common whitespace (tab/newline/cr kept then normalized).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def strip_control_chars(text: str) -> str:
    """Remove C0/C1 control characters; keep \\t \\n \\r then collapse later."""
    return _CONTROL_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Unicode NFC + collapse runs of whitespace; strip ends."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def looks_like_prompt_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def sanitize_message(message: str) -> str:
    """Normalize message; raise BadRequestError on empty, oversize, or injection."""
    if message is None:
        raise BadRequestError("message is required", error_code="missing_message")

    cleaned = normalize_whitespace(strip_control_chars(str(message)))
    if not cleaned:
        raise BadRequestError("message must not be empty", error_code="empty_message")

    if len(cleaned) > MAX_MESSAGE_CHARS:
        raise BadRequestError(
            f"message exceeds maximum length of {MAX_MESSAGE_CHARS} characters",
            error_code="message_too_long",
        )

    if looks_like_prompt_injection(cleaned):
        raise BadRequestError(
            "message contains disallowed instruction-override patterns",
            error_code="prompt_injection_blocked",
        )

    return cleaned


def sanitize_email(email: str) -> str:
    """Normalize email string (EmailStr already validates format upstream)."""
    cleaned = normalize_whitespace(strip_control_chars(str(email))).lower()
    if not cleaned:
        raise BadRequestError("email must not be empty", error_code="empty_email")
    if len(cleaned) > MAX_EMAIL_CHARS:
        raise BadRequestError(
            f"email exceeds maximum length of {MAX_EMAIL_CHARS} characters",
            error_code="email_too_long",
        )
    # Light nonsense check: no spaces, must look like local@domain
    if " " in cleaned or cleaned.count("@") != 1:
        raise BadRequestError("email format is invalid", error_code="invalid_email")
    local, _, domain = cleaned.partition("@")
    if not local or not domain or "." not in domain:
        raise BadRequestError("email format is invalid", error_code="invalid_email")
    return cleaned
