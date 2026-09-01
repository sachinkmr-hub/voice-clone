"""Anonymisation helpers.

Phone numbers and claimed identities are personal data. They are useful to keep for
correlation ("this number has called nine customers today") but there is no reason to
store them in the clear to do that, so the default is a salted hash: correlation still
works, re-identification does not.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Optional

from voiceguard.config import Settings, get_settings

PII_KEYS = ("caller_number", "claimed_identity", "callee_number", "customer_id",
            "account_number", "email")


def hash_identifier(value: str, salt: str, *, length: int = 16) -> str:
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()
    return f"h:{digest[:length]}"


def mask_number(value: str) -> str:
    """Keep the shape of a phone number for display without keeping the number."""
    digits = re.sub(r"\D", "", value or "")
    if len(digits) < 4:
        return "***"
    return f"{'*' * max(0, len(digits) - 4)}{digits[-4:]}"


def anonymize_value(key: str, value: Any, settings: Optional[Settings] = None) -> Any:
    settings = settings or get_settings()
    if settings.store_pii or not isinstance(value, str) or not value:
        return value
    if key in PII_KEYS:
        return hash_identifier(value, settings.pii_salt)
    return value


def anonymize_dict(payload: Optional[Dict[str, Any]],
                   settings: Optional[Settings] = None) -> Dict[str, Any]:
    """Return a copy with PII fields hashed, unless ``STORE_PII`` is enabled."""
    if not payload:
        return {}
    settings = settings or get_settings()
    return {k: anonymize_value(k, v, settings) for k, v in payload.items()}


def redact_transcript(text: str, max_chars: int = 0) -> str:
    """Strip digit runs from a transcript before it is logged.

    Live transcripts are where account numbers and OTPs show up. We need the *shape* of
    the language (urgency, secrecy) for the context layer, never the digits themselves,
    so the digits are removed before anything is persisted.
    """
    if not text:
        return ""
    redacted = re.sub(r"\b\d[\d\s-]{3,}\b", "[number]", text)
    redacted = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", "[email]", redacted)
    return redacted[:max_chars] if max_chars else redacted
