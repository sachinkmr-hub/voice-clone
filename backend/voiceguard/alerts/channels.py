"""Alert delivery channels.

Each channel is a small, independently-failing adapter. A channel that raises must never
prevent the others from firing — an SMTP timeout cannot be allowed to stop the WebSocket
alert that the agent on the call is actually looking at.

The SMS and email channels are structured for a real provider but ship in ``dry_run``
mode: they format and record the message rather than sending it. That is a deliberate
choice for a hackathon deliverable — wiring live credentials into a public repo would be
worse than useless — and swapping in a provider is one method per channel.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger("voiceguard.alerts")


@dataclass
class Alert:
    """One alert, as delivered to every channel."""

    session_id: str
    score: float
    band: str
    action: str
    headline: str
    factors: List[dict] = field(default_factory=list)
    profile: str = "default"
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "score": round(float(self.score), 1),
            "band": self.band,
            "action": self.action,
            "headline": self.headline,
            "factors": self.factors,
            "profile": self.profile,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    def as_text(self) -> str:
        reasons = "; ".join(f["label"] for f in self.factors[:2]) or "no single dominant factor"
        return (
            f"[VoiceGuard {self.band}] Call {self.session_id}: risk {self.score:.0f}/100. "
            f"{reasons}. {self.action}"
        )


class AlertChannel:
    """Base channel."""

    name = "base"

    async def send(self, alert: Alert) -> dict:
        raise NotImplementedError


class WebSocketChannel(AlertChannel):
    """Pushes alerts to connected dashboards via a broadcast callable."""

    name = "websocket"

    def __init__(self, broadcast: Callable[[dict], Awaitable[None]]) -> None:
        self._broadcast = broadcast

    async def send(self, alert: Alert) -> dict:
        await self._broadcast({"type": "alert", "alert": alert.as_dict()})
        return {"channel": self.name, "delivered": True}


class WebhookChannel(AlertChannel):
    """POSTs the alert to a configured URL (e.g. a SIEM or a core-banking hook)."""

    name = "webhook"

    def __init__(self, url: str = "", timeout: float = 4.0) -> None:
        self.url = url
        self.timeout = timeout
        self.sent: List[dict] = []

    async def send(self, alert: Alert) -> dict:
        payload = alert.as_dict()
        if not self.url:
            self.sent.append(payload)
            return {"channel": self.name, "delivered": False, "reason": "no webhook URL configured"}
        try:
            import httpx

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.url, json=payload)
            self.sent.append(payload)
            return {"channel": self.name, "delivered": response.status_code < 400,
                    "status": response.status_code}
        except Exception as exc:
            logger.warning("webhook delivery failed: %s", exc)
            return {"channel": self.name, "delivered": False, "reason": str(exc)}


class EmailChannel(AlertChannel):
    """Formats an email. Dry-run by default; set ``sender`` to deliver for real."""

    name = "email"

    def __init__(self, recipients: Optional[List[str]] = None,
                 sender: Optional[Callable[[str, str, str], None]] = None) -> None:
        self.recipients = recipients or []
        self.sender = sender
        self.outbox: List[dict] = []

    async def send(self, alert: Alert) -> dict:
        subject = f"[VoiceGuard] {alert.band} risk on call {alert.session_id}"
        body = "\n".join([
            alert.headline,
            "",
            f"Risk score: {alert.score:.0f}/100 ({alert.band})",
            f"Recommended action: {alert.action}",
            "",
            "Contributing factors:",
            *[f"  - {f['label']}: {f.get('detail', '')}" for f in alert.factors[:5]],
        ])
        message = {"to": self.recipients, "subject": subject, "body": body}
        self.outbox.append(message)
        if self.sender is None:
            return {"channel": self.name, "delivered": False, "reason": "dry-run",
                    "preview": subject}
        for recipient in self.recipients:
            self.sender(recipient, subject, body)
        return {"channel": self.name, "delivered": True, "recipients": len(self.recipients)}


class SMSChannel(AlertChannel):
    """Formats a 160-character SMS. Dry-run by default."""

    name = "sms"

    def __init__(self, numbers: Optional[List[str]] = None,
                 sender: Optional[Callable[[str, str], None]] = None) -> None:
        self.numbers = numbers or []
        self.sender = sender
        self.outbox: List[dict] = []

    async def send(self, alert: Alert) -> dict:
        text = alert.as_text()[:160]
        self.outbox.append({"to": self.numbers, "text": text})
        if self.sender is None:
            return {"channel": self.name, "delivered": False, "reason": "dry-run",
                    "preview": text}
        for number in self.numbers:
            self.sender(number, text)
        return {"channel": self.name, "delivered": True, "recipients": len(self.numbers)}


class ConsoleChannel(AlertChannel):
    """Always-available fallback so an alert is never silently lost."""

    name = "console"

    def __init__(self) -> None:
        self.records: List[dict] = []

    async def send(self, alert: Alert) -> dict:
        logger.warning("ALERT %s", json.dumps(alert.as_dict(), default=str))
        self.records.append(alert.as_dict())
        return {"channel": self.name, "delivered": True}


async def deliver(channels: List[AlertChannel], alert: Alert) -> List[dict]:
    """Fan out to every channel concurrently, isolating failures."""
    if not channels:
        return []
    results = await asyncio.gather(
        *(channel.send(alert) for channel in channels), return_exceptions=True
    )
    output: List[dict] = []
    for channel, result in zip(channels, results):
        if isinstance(result, Exception):
            logger.warning("channel %s raised: %s", channel.name, result)
            output.append({"channel": channel.name, "delivered": False, "reason": str(result)})
        else:
            output.append(result)
    return output
