"""Integration endpoints — the "so what" of the risk score.

A number on a dashboard changes nothing. What changes outcomes is a transaction that does
not complete. This module is the reference integration a core-banking or ERP system calls
before releasing funds: it takes a transaction and a live ``session_id`` and returns
``allow`` / ``step_up`` / ``block`` with the reason chain that produced the decision.

The decision logic lives here rather than in the risk engine on purpose: the engine
answers "is this voice synthetic", the integration answers "should this money move", and
those are different questions owned by different teams.
"""

from __future__ import annotations

import time
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from voiceguard.api.deps import AppState, get_state, require_api_key
from voiceguard.api.schemas import ApprovalRequest, ApprovalResponse
from voiceguard.config import RiskBand

router = APIRouter(prefix="/v1/integrations", tags=["integrations"],
                   dependencies=[Depends(require_api_key)])

#: Value above which we never allow a silent pass, whatever the voice score says. A clean
#: voice score is not authorisation — it is the absence of one specific red flag.
ALWAYS_STEP_UP_ABOVE = 1_000_000.0


@router.post("/bank/approval", response_model=ApprovalResponse,
             summary="Gate a transaction on the live voice risk")
def bank_approval(request: ApprovalRequest, state: AppState = Depends(get_state)):
    session = state.sessions.get(request.session_id)
    report = None
    if session is None:
        report = state.repository.get_session(request.session_id)
        if report is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"No voice session {request.session_id!r}. Start voice analysis before "
                f"requesting approval.",
            )

    if session is not None:
        score = session.current_score
        band = session.band
        latest = session.latest
        factors = [f.label for f in latest.explanation.factors[:3]] if latest else []
        windows = session.stats.windows_analyzed
        audio_seconds = session.stats.audio_seconds
    else:
        score = float(report.get("peak_score") or report.get("final_score") or 0.0)
        stored = report.get("report") or {}
        band = stored.get("band", RiskBand.LOW.value)
        factors = [f.get("label", f.get("code", "")) for f in stored.get("top_factors", [])[:3]]
        windows = (stored.get("stats") or {}).get("windows_analyzed", 0)
        audio_seconds = report.get("duration_seconds") or 0.0

    profile = state.profile(request.profile)
    thresholds = {"elevated": profile.elevated, "high": profile.high,
                  "critical": profile.critical}

    reasons: List[str] = []
    verification: List[str] = []

    # Evidence sufficiency comes first: a low score from two seconds of audio is not a
    # clean bill of health, and treating it as one is how this class of tool gets gamed.
    insufficient = windows < 3 or audio_seconds < 3.0
    if insufficient:
        reasons.append(
            f"Only {audio_seconds:.1f} s of speech analysed ({windows} window(s)) — "
            f"not enough evidence to clear this call."
        )

    if score >= thresholds["critical"]:
        decision = "block"
        reasons.append(f"Voice risk {score:.0f}/100 is at or above the "
                       f"{profile.name} block threshold ({thresholds['critical']:.0f}).")
        verification = ["supervisor_escalation", "callback_on_registered_number"]
    elif score >= thresholds["high"]:
        decision = "block" if request.amount >= ALWAYS_STEP_UP_ABOVE else "step_up"
        reasons.append(f"Voice risk {score:.0f}/100 exceeds the "
                       f"{profile.name} review threshold ({thresholds['high']:.0f}).")
        verification = ["callback_on_registered_number", "second_factor"]
    elif score >= thresholds["elevated"] or insufficient:
        decision = "step_up"
        if score >= thresholds["elevated"]:
            reasons.append(f"Voice risk {score:.0f}/100 is elevated.")
        verification = ["knowledge_based_question", "second_factor"]
    elif request.amount >= ALWAYS_STEP_UP_ABOVE:
        decision = "step_up"
        reasons.append(
            f"Voice check passed, but {request.currency} {request.amount:,.0f} is above "
            f"the value ceiling for voice-only authorisation."
        )
        verification = ["second_factor"]
    else:
        decision = "allow"
        reasons.append(f"Voice risk {score:.0f}/100 is within normal range for "
                       f"{profile.name}.")

    if factors and decision != "allow":
        reasons.append("Contributing factors: " + "; ".join(factors) + ".")

    messages = {
        "allow": "Approved. No voice-impersonation indicators above threshold.",
        "step_up": "Additional verification required before this transaction can proceed.",
        "block": "Blocked. Do not act on this call — verify through an independent channel.",
    }

    return ApprovalResponse(
        decision=decision,
        reference=request.reference or f"txn_{int(time.time() * 1000)}",
        session_id=request.session_id,
        risk_score=round(score, 1),
        band=band,
        reasons=reasons,
        required_verification=verification,
        message=messages[decision],
        evaluated_at=time.time(),
    )


@router.get("/bank/policy", summary="Explain the approval policy")
def bank_policy(state: AppState = Depends(get_state)):
    """Exposed so an integrator can show the rules to an auditor without reading code."""
    return {
        "profiles": {name: profile.as_dict() for name, profile in state.profiles.items()},
        "always_step_up_above": ALWAYS_STEP_UP_ABOVE,
        "minimum_evidence": {"windows": 3, "audio_seconds": 3.0},
        "decisions": {
            "allow": "score below the elevated threshold, enough evidence, value under ceiling",
            "step_up": "elevated score, insufficient evidence, or high-value transaction",
            "block": "score at/above the critical threshold, or high score on a large amount",
        },
        "note": (
            "A clean voice score is the absence of one red flag, not an authorisation. "
            "Value ceilings and evidence sufficiency apply independently."
        ),
    }
