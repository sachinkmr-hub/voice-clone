"""Threshold-based alerting with de-duplication and escalation.

The hard requirement here is *not* "fire when score > threshold". It is "fire when it
matters, and stay quiet otherwise", because a contact-centre tool that interrupts an agent
on every third call gets switched off in a week — and then detects nothing at all.

Three rules do the work:

* **Cooldown.** At most one alert per session per cooldown window.
* **Escalation-only.** Within a session, re-alert only when the band goes *up*. A call
  sitting at HIGH for two minutes is one alert, not two hundred and forty.
* **Warm-up suppression.** No alerts while the score is still provisional, so the first
  second of a call cannot produce a spurious CRITICAL.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from voiceguard.config import RiskBand, RiskProfile, Settings, get_settings
from voiceguard.alerts.channels import Alert, AlertChannel, ConsoleChannel, deliver
from voiceguard.scoring.risk import RiskAssessment

BAND_ORDER = {
    RiskBand.LOW.value: 0,
    RiskBand.ELEVATED.value: 1,
    RiskBand.HIGH.value: 2,
    RiskBand.CRITICAL.value: 3,
}


class AlertEngine:
    """Decides whether an assessment deserves an alert, then delivers it."""

    #: Bands at or above this raise an alert.
    MIN_ALERT_BAND = RiskBand.HIGH.value

    def __init__(
        self,
        channels: Optional[Dict[str, AlertChannel]] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.channels: Dict[str, AlertChannel] = dict(channels or {})
        self.channels.setdefault("console", ConsoleChannel())
        self._last_alert_at: Dict[str, float] = {}
        self._highest_band: Dict[str, int] = {}
        self.history: List[dict] = []

    # -------------------------------------------------------------------- policy
    def should_alert(self, session_id: str, assessment: RiskAssessment) -> bool:
        if assessment.provisional:
            return False
        band_rank = BAND_ORDER.get(assessment.band, 0)
        if band_rank < BAND_ORDER[self.MIN_ALERT_BAND]:
            return False

        previous_rank = self._highest_band.get(session_id, -1)
        if band_rank <= previous_rank:
            return False           # escalation-only

        last = self._last_alert_at.get(session_id, 0.0)
        if time.time() - last < self.settings.alert_cooldown_seconds:
            return False
        return True

    def channels_for(self, profile: Optional[RiskProfile]) -> List[AlertChannel]:
        if profile is None:
            return [self.channels["console"]]
        selected = [self.channels[name] for name in profile.alert_channels
                    if name in self.channels]
        return selected or [self.channels["console"]]

    # ------------------------------------------------------------------ delivery
    async def maybe_alert(
        self,
        session_id: str,
        assessment: RiskAssessment,
        *,
        profile: Optional[RiskProfile] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[dict]:
        """Evaluate the policy and, if it passes, deliver on every configured channel."""
        if not self.should_alert(session_id, assessment):
            return None

        alert = Alert(
            session_id=session_id,
            score=assessment.score,
            band=assessment.band,
            action=assessment.action,
            headline=assessment.explanation.headline,
            factors=[f.as_dict() for f in assessment.explanation.factors],
            profile=assessment.profile,
            metadata=dict(metadata or {}),
        )
        results = await deliver(self.channels_for(profile), alert)

        self._last_alert_at[session_id] = time.time()
        self._highest_band[session_id] = BAND_ORDER.get(assessment.band, 0)

        record = {"alert": alert.as_dict(), "delivery": results}
        self.history.append(record)
        if len(self.history) > 500:
            self.history = self.history[-500:]
        return record

    # ------------------------------------------------------------------- upkeep
    def register(self, name: str, channel: AlertChannel) -> None:
        self.channels[name] = channel

    def reset_session(self, session_id: str) -> None:
        self._last_alert_at.pop(session_id, None)
        self._highest_band.pop(session_id, None)

    def recent(self, limit: int = 50) -> List[dict]:
        return list(self.history[-limit:][::-1])

    def describe(self) -> dict:
        return {
            "channels": sorted(self.channels),
            "min_alert_band": self.MIN_ALERT_BAND,
            "cooldown_seconds": self.settings.alert_cooldown_seconds,
            "alerts_raised": len(self.history),
        }
