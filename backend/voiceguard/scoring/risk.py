"""The risk engine — from layer scores to a decision someone can act on.

Separation of concerns, enforced deliberately:

* the **model** produces a calibrated probability that the audio is synthetic;
* the **policy** (a :class:`~voiceguard.config.RiskProfile`) decides what probability is
  worth interrupting a human for, in this use case;
* **context** (amount, hour, caller reputation) shifts the *thresholds*, never the
  probability.

That last rule is the important one. If a ₹50 lakh transfer nudged the probability upward,
the number on the screen would no longer mean "how likely is this synthetic" and the audit
trail would be indefensible — you could not tell a regulator why the same audio scored
differently on two calls. Instead the probability is untouched and the bar for acting on
it moves, which is exactly how a human analyst reasons and is trivially explainable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from voiceguard.config import (
    DEFAULT_PROFILES,
    RECOMMENDED_ACTIONS,
    RiskBand,
    RiskProfile,
)
from voiceguard.models.base import LayerResult, ramp
from voiceguard.models.context import CallContext
from voiceguard.scoring.explain import Explanation, build_explanation
from voiceguard.scoring.fusion import FusionResult, ScoreFusion


@dataclass
class RiskAssessment:
    """The complete verdict for one analysis window (or one whole call)."""

    score: float                       #: 0–100
    band: str
    action: str
    probability: float                 #: calibrated P(synthetic)
    confidence: float
    explanation: Explanation
    fusion: FusionResult
    profile: str = "default"
    threshold_shift: float = 0.0       #: how far context moved the bands, in points
    window_index: int = 0
    elapsed_seconds: float = 0.0
    speech_detected: bool = True
    provisional: bool = False
    latency_ms: float = 0.0
    created_at: float = field(default_factory=time.time)

    def as_dict(self, *, verbose: bool = False) -> dict:
        payload = {
            "score": round(float(self.score), 1),
            "band": self.band,
            "action": self.action,
            "probability": round(float(self.probability), 4),
            "confidence": round(float(self.confidence), 4),
            "profile": self.profile,
            "threshold_shift": round(float(self.threshold_shift), 2),
            "window_index": self.window_index,
            "elapsed_seconds": round(float(self.elapsed_seconds), 2),
            "speech_detected": self.speech_detected,
            "provisional": self.provisional,
            "latency_ms": round(float(self.latency_ms), 2),
            "headline": self.explanation.headline,
            "factors": [f.as_dict() for f in self.explanation.factors],
            "caveats": list(self.explanation.caveats),
            "created_at": self.created_at,
        }
        if verbose:
            payload["explanation"] = self.explanation.as_dict()
            payload["fusion"] = self.fusion.as_dict()
        else:
            payload["layers"] = self.explanation.layer_summary
        return payload


class RiskEngine:
    """Fuses layer results, applies policy and produces a :class:`RiskAssessment`."""

    #: Context can move the thresholds by at most this many points in either direction.
    #: Without a cap, a large enough transaction would make every call critical.
    MAX_THRESHOLD_SHIFT = 18.0

    def __init__(
        self,
        profile: Optional[RiskProfile] = None,
        fusion: Optional[ScoreFusion] = None,
        profiles: Optional[Dict[str, RiskProfile]] = None,
    ) -> None:
        self.profiles = profiles if profiles is not None else dict(DEFAULT_PROFILES)
        self.profile = profile or self.profiles.get("default", DEFAULT_PROFILES["default"])
        self.fusion = fusion or ScoreFusion()

    # ---------------------------------------------------------------- policy
    def use_profile(self, name: str) -> RiskProfile:
        self.profile = self.profiles.get(name, self.profile)
        return self.profile

    def context_threshold_shift(self, call: Optional[CallContext]) -> float:
        """How many points to lower the alerting bar, given the situation.

        Positive means "act sooner". Each term is small on its own; the cap keeps the
        total from swamping the acoustic evidence.
        """
        if call is None:
            return 0.0
        shift = 0.0
        if call.transaction_amount > 0:
            shift += 10.0 * ramp(call.transaction_amount, 100_000.0, 5_000_000.0)
        if call.prior_fraud_reports > 0:
            shift += 6.0 * ramp(float(call.prior_fraud_reports), 0.0, 3.0)
        if call.known_contact is False:
            shift += 3.0
        if call.caller_id_verified is False:
            shift += 3.0
        elif call.caller_id_verified is True:
            shift -= 2.0
        if call.local_hour is not None and (call.local_hour < 7 or call.local_hour >= 21):
            shift += 2.5
        if call.known_contact is True and call.prior_fraud_reports == 0:
            shift -= 3.0
        return float(max(-self.MAX_THRESHOLD_SHIFT, min(self.MAX_THRESHOLD_SHIFT, shift)))

    def _band(self, score: float, shift: float,
              profile: Optional[RiskProfile] = None) -> RiskBand:
        profile = profile or self.profile
        if score >= max(0.0, profile.critical - shift):
            return RiskBand.CRITICAL
        if score >= max(0.0, profile.high - shift):
            return RiskBand.HIGH
        if score >= max(0.0, profile.elevated - shift):
            return RiskBand.ELEVATED
        return RiskBand.LOW

    # -------------------------------------------------------------- assessment
    def assess(
        self,
        layer_results: Sequence[LayerResult],
        *,
        call_context: Optional[CallContext] = None,
        profile_name: Optional[str] = None,
        window_index: int = 0,
        elapsed_seconds: float = 0.0,
        speech_ratio: float = 1.0,
        speech_detected: bool = True,
        provisional: bool = False,
        latency_ms: float = 0.0,
        score_override: Optional[float] = None,
    ) -> RiskAssessment:
        """Produce the verdict.

        ``score_override`` lets a session supply its smoothed call-level score while still
        getting a per-window explanation — the number the user sees and the reasons under
        it then always refer to the same thing.
        """
        profile = self.profiles.get(profile_name, self.profile) if profile_name else self.profile
        fusion = self.fusion.fuse(layer_results)

        score = float(fusion.score_100 if score_override is None else score_override)
        shift = self.context_threshold_shift(call_context)
        band = self._band(score, shift, profile)

        explanation = build_explanation(
            fusion,
            score=score,
            band=band.value,
            speech_ratio=speech_ratio,
            elapsed_seconds=elapsed_seconds,
            action=RECOMMENDED_ACTIONS.get(band.value, ""),
        )
        if shift > 1.0:
            explanation.caveats.append(
                f"Alerting thresholds lowered by {shift:.0f} points for this call's "
                f"context (value, hour, caller reputation)."
            )

        return RiskAssessment(
            score=score,
            band=band.value,
            action=RECOMMENDED_ACTIONS.get(band.value, ""),
            probability=fusion.probability,
            confidence=fusion.confidence,
            explanation=explanation,
            fusion=fusion,
            profile=profile.name,
            threshold_shift=shift,
            window_index=window_index,
            elapsed_seconds=elapsed_seconds,
            speech_detected=speech_detected,
            provisional=provisional,
            latency_ms=latency_ms,
        )

    # ---------------------------------------------------------------- helpers
    def effective_thresholds(self, call_context: Optional[CallContext] = None,
                             profile_name: Optional[str] = None) -> Dict[str, float]:
        profile = self.profiles.get(profile_name, self.profile) if profile_name else self.profile
        shift = self.context_threshold_shift(call_context)
        return {
            "elevated": max(0.0, profile.elevated - shift),
            "high": max(0.0, profile.high - shift),
            "critical": max(0.0, profile.critical - shift),
            "shift": shift,
        }

    def describe(self) -> dict:
        return {
            "profile": self.profile.as_dict(),
            "available_profiles": sorted(self.profiles),
            "fusion": self.fusion.as_dict(),
        }


class ScoreSmoother:
    """Call-level score smoothing.

    Two behaviours a live gauge needs and a raw per-window score does not have:

    * **EWMA** so the needle does not jitter between adjacent windows;
    * **a sustained-evidence guard** so that a burst of strong synthetic evidence is not
      immediately averaged away by following windows of near-silence. Fraud happens in the
      few seconds someone reads out an account number, and a detector that forgets that
      within two windows is useless.
    """

    def __init__(self, alpha: float = 0.35, memory: int = 8,
                 sustain_quantile: float = 0.8) -> None:
        self.alpha = float(alpha)
        self.memory = int(memory)
        self.sustain_quantile = float(sustain_quantile)
        self.value: Optional[float] = None
        self.history: List[float] = []

    def update(self, score: float) -> float:
        score = float(score)
        self.history.append(score)
        if len(self.history) > self.memory:
            self.history = self.history[-self.memory :]

        if self.value is None:
            self.value = score
        else:
            self.value = self.alpha * score + (1.0 - self.alpha) * self.value

        # Sustained-evidence guard: if several recent windows agree on a higher score,
        # do not let the average drag the reading below what they consistently showed.
        if len(self.history) >= 3:
            ordered = sorted(self.history)
            index = min(len(ordered) - 1, int(self.sustain_quantile * (len(ordered) - 1)))
            sustained = ordered[index]
            if sustained > self.value:
                self.value = self.value + 0.5 * (sustained - self.value)

        return float(self.value)

    def peak(self) -> float:
        return float(max(self.history)) if self.history else 0.0

    def reset(self) -> None:
        self.value = None
        self.history = []
