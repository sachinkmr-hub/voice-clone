"""Layer 4 — call-context enrichment.

Per PS §6.2, the risk engine enriches the acoustic verdict with call metadata: origin,
known-contact status, transaction context and historical fraud indicators.

This layer is deliberately the weakest voice in the fusion (default weight 0.08). Context
is *correlational* — an unknown number at 2 a.m. asking for a wire transfer is suspicious
whether or not the voice is real — and letting it drive the score would produce a system
that flags legitimate strangers and misses well-socially-engineered clones from known
numbers. It sharpens the acoustic decision; it never replaces it.

Every signal here is explainable in one line to a bank agent, which is a hard requirement:
an agent who cannot explain a block to the customer on the phone will stop using the tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from voiceguard.config import Layer
from voiceguard.models.base import Detector, Factor, LayerResult, ramp, sigmoid

#: Social-engineering pressure language, in the forms that actually appear in Indian
#: vishing transcripts (English, transliterated Hindi, and common code-mixing).
URGENCY_PATTERNS = (
    r"\burgent(?:ly)?\b", r"\bimmediate(?:ly)?\b", r"\bright now\b", r"\bemergency\b",
    r"\bdon'?t tell\b", r"\bdo not tell\b", r"\bkeep this (?:between|confidential)\b",
    r"\bconfidential\b", r"\bbefore (?:the )?(?:market|bank) clos", r"\bin the next \d+ minutes?\b",
    r"\botp\b", r"\bone[- ]time password\b", r"\bshare (?:the )?code\b",
    r"\bturant\b", r"\bjaldi\b", r"\babhi ke abhi\b", r"\bkisi ko mat bata\b",
    r"\bpolice\b.*\bcase\b", r"\barrest\b", r"\bdigital arrest\b",
    r"\bcustoms\b.*\bparcel\b", r"\bblock(?:ed)? (?:your )?account\b",
)

#: Roles whose impersonation is the actual attack in this problem statement.
HIGH_VALUE_ROLES = ("ceo", "cfo", "cto", "coo", "director", "chairman", "managing director",
                    "md", "vp", "head", "manager", "commissioner", "officer", "inspector")


@dataclass
class CallContext:
    """Metadata a caller-facing system can realistically supply."""

    caller_number: Optional[str] = None
    known_contact: Optional[bool] = None
    caller_id_verified: Optional[bool] = None     #: STIR/SHAKEN-style attestation
    origin: str = ""                              #: "pstn" | "voip" | "webrtc" | "international"
    claimed_identity: str = ""
    claimed_role: str = ""
    transaction_amount: float = 0.0
    currency: str = "INR"
    local_hour: Optional[int] = None
    first_contact: Optional[bool] = None
    prior_fraud_reports: int = 0
    transcript: str = ""
    account_age_days: Optional[int] = None
    extra: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "CallContext":
        data = dict(data or {})
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        extra = {k: v for k, v in data.items() if k not in known and isinstance(v, (int, float))}
        return cls(**{k: v for k, v in data.items() if k in known}, extra=extra)

    def as_dict(self) -> dict:
        return {
            "caller_number": self.caller_number,
            "known_contact": self.known_contact,
            "caller_id_verified": self.caller_id_verified,
            "origin": self.origin,
            "claimed_identity": self.claimed_identity,
            "claimed_role": self.claimed_role,
            "transaction_amount": self.transaction_amount,
            "currency": self.currency,
            "local_hour": self.local_hour,
            "first_contact": self.first_contact,
            "prior_fraud_reports": self.prior_fraud_reports,
            "account_age_days": self.account_age_days,
        }


def urgency_hits(transcript: str) -> List[str]:
    """Which pressure phrases appear in the (optional) live transcript."""
    if not transcript:
        return []
    text = transcript.lower()
    return [pattern for pattern in URGENCY_PATTERNS if re.search(pattern, text)]


class ContextDetector(Detector):
    """Scores the situational risk surrounding the call."""

    layer = Layer.CONTEXT.value
    model_id = "context-rules-v1"

    def analyze(self, features: Dict[str, float],
                context: Optional[dict] = None) -> LayerResult:
        call = context.get("call_context") if context else None
        if call is None:
            return LayerResult.unavailable(self.layer, "no call metadata supplied")
        if not isinstance(call, CallContext):
            call = CallContext.from_dict(call)

        factors: List[Factor] = []
        signals: List[tuple] = []   # (strength, weight)

        def add(code: str, label: str, strength: float, weight: float,
                detail: str, value: float = 0.0) -> None:
            signals.append((strength, weight))
            if strength > 0:
                factors.append(Factor(code=code, label=label, contribution=strength,
                                      value=value, detail=detail, layer=self.layer))

        if call.caller_id_verified is False:
            add("caller_id_unverified", "Caller ID not attested", 1.0, 1.2,
                "the network could not attest this caller ID (possible spoofing)")
        elif call.caller_id_verified is True:
            add("caller_id_verified", "Caller ID attested", 0.0, 0.6,
                "caller ID carries a valid network attestation")

        if call.known_contact is False:
            add("unknown_contact", "Number not in the known-contact list", 0.85, 1.0,
                "the number is not registered against this customer or employee")
        elif call.known_contact is True:
            add("known_contact", "Known contact", 0.0, 0.5,
                "number matches a registered contact")

        if call.first_contact:
            add("first_contact", "First-ever contact from this number", 0.6, 0.5,
                "no prior call history with this number")

        if call.prior_fraud_reports > 0:
            add("prior_fraud_reports", "Number previously reported for fraud",
                ramp(float(call.prior_fraud_reports), 0.0, 3.0), 1.4,
                f"{call.prior_fraud_reports} prior fraud report(s) against this number",
                float(call.prior_fraud_reports))

        if call.origin:
            origin_risk = {"international": 0.7, "voip": 0.45, "webrtc": 0.3,
                           "pstn": 0.05, "mobile": 0.05}.get(call.origin.lower(), 0.25)
            add("origin_risk", f"Call origin: {call.origin}", origin_risk, 0.7,
                f"call arrived over {call.origin}")

        if call.transaction_amount > 0:
            # ₹1 lakh is the usual step-up threshold; ₹50 lakh is materially catastrophic.
            strength = ramp(call.transaction_amount, 100_000.0, 5_000_000.0)
            add("transaction_amount", "High-value request", strength, 0.9,
                f"request involves {call.currency} {call.transaction_amount:,.0f}",
                call.transaction_amount)

        if call.local_hour is not None:
            off_hours = call.local_hour < 7 or call.local_hour >= 21
            add("off_hours", "Outside business hours", 0.55 if off_hours else 0.0, 0.5,
                f"call placed at {call.local_hour:02d}:00 local time",
                float(call.local_hour))

        role = (call.claimed_role or call.claimed_identity).lower()
        if role and any(r in role for r in HIGH_VALUE_ROLES):
            add("high_value_role", "Caller claims a senior authority role", 0.6, 0.7,
                f"caller presents as '{call.claimed_role or call.claimed_identity}'")

        hits = urgency_hits(call.transcript)
        if call.transcript:
            add("urgency_language", "Pressure / secrecy language",
                ramp(float(len(hits)), 0.0, 3.0), 1.1,
                f"{len(hits)} social-engineering phrase(s) detected in the live transcript",
                float(len(hits)))

        if call.account_age_days is not None and call.account_age_days < 30:
            add("new_account", "Recently created account", 0.5, 0.4,
                f"destination account is {call.account_age_days} day(s) old",
                float(call.account_age_days))

        if not signals:
            return LayerResult.unavailable(self.layer, "call metadata carried no usable signal")

        total_weight = sum(w for _, w in signals)
        evidence = sum(s * w for s, w in signals) / total_weight
        probability = sigmoid(-1.4 + 3.2 * evidence)

        for factor in factors:
            factor.contribution = float(factor.contribution / max(total_weight, 1e-9))

        # Confidence scales with how much metadata we actually got.
        coverage = min(1.0, total_weight / 5.0)
        return LayerResult(
            layer=self.layer,
            score=float(probability),
            confidence=float(0.3 + 0.5 * coverage),
            factors=factors,
            model_id=self.model_id,
            features_used=len(signals),
        )
