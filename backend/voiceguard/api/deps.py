"""Shared application state and request dependencies.

One :class:`AppState` object owns the registry, session manager, alert engine, audit
repository and dashboard hub. It is built once at startup and injected everywhere, which
keeps the routes free of module-level singletons and makes the whole app constructible
inside a test with a temporary database.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Set

from fastapi import Header, HTTPException, Query, Request, WebSocket, status

from voiceguard.alerts.channels import (
    ConsoleChannel,
    EmailChannel,
    SMSChannel,
    WebhookChannel,
    WebSocketChannel,
)
from voiceguard.alerts.engine import AlertEngine
from voiceguard.config import DEFAULT_PROFILES, RiskProfile, Settings, get_settings
from voiceguard.features.extractor import FeatureExtractor
from voiceguard.models.registry import ModelRegistry
from voiceguard.pipeline.manager import SessionManager
from voiceguard.privacy.retention import RetentionPolicy, RetentionSweeper
from voiceguard.storage.db import Database
from voiceguard.storage.repository import AuditRepository

logger = logging.getLogger("voiceguard.api")


class DashboardHub:
    """Fan-out of live events to every connected dashboard.

    A dead socket must never block ingest, so sends are best-effort and failures simply
    drop the subscriber. The alternative — awaiting a stalled client — would stall the
    call it is reporting on.
    """

    def __init__(self, buffer_size: int = 100) -> None:
        self._clients: Set[WebSocket] = set()
        self._buffer: List[dict] = []
        self.buffer_size = buffer_size
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients.add(websocket)
        for event in self._buffer[-25:]:
            try:
                await websocket.send_text(json.dumps(event, default=str))
            except Exception:
                break

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)

    async def broadcast(self, event: dict) -> None:
        self._buffer.append(event)
        if len(self._buffer) > self.buffer_size:
            self._buffer = self._buffer[-self.buffer_size :]

        async with self._lock:
            clients = list(self._clients)
        if not clients:
            return

        payload = json.dumps(event, default=str)
        dead: List[WebSocket] = []
        for client in clients:
            try:
                await client.send_text(payload)
            except Exception:
                dead.append(client)
        if dead:
            async with self._lock:
                for client in dead:
                    self._clients.discard(client)

    @property
    def client_count(self) -> int:
        return len(self._clients)


class AppState:
    """Everything the API needs, constructed once."""

    def __init__(self, settings: Optional[Settings] = None,
                 database: Optional[Database] = None) -> None:
        self.settings = settings or get_settings()
        self.started_at = time.time()

        self.registry = ModelRegistry(self.settings)
        self.extractor = FeatureExtractor()
        self.sessions = SessionManager(self.settings, self.registry, self.extractor)
        self.repository = AuditRepository(
            database or Database(self.settings.database_url), self.settings)
        self.hub = DashboardHub()

        self.alerts = AlertEngine(settings=self.settings)
        self.alerts.register("websocket", WebSocketChannel(self.hub.broadcast))
        self.alerts.register("webhook", WebhookChannel(self.settings.webhook_url))
        self.alerts.register("email", EmailChannel())
        self.alerts.register("sms", SMSChannel())
        self.alerts.register("console", ConsoleChannel())

        self.profiles: Dict[str, RiskProfile] = {
            name: RiskProfile(**profile.as_dict()) for name, profile in DEFAULT_PROFILES.items()
        }
        self.sweeper = RetentionSweeper(self.repository, self.settings)

        self._restore_enrolments()

    def _restore_enrolments(self) -> None:
        """Re-hydrate enrolled speaker profiles from the audit store on startup."""
        try:
            profiles = self.repository.load_enrolments()
        except Exception as exc:
            logger.warning("could not restore enrolments: %s", exc)
            return
        for identity, vectors in profiles.items():
            for vector in vectors:
                self.registry.enrolment.enrol_vector(identity, vector)
        if profiles:
            logger.info("restored %d enrolled identities", len(profiles))

    # ------------------------------------------------------------------ helpers
    def profile(self, name: Optional[str]) -> RiskProfile:
        return self.profiles.get(name or self.settings.default_profile,
                                 self.profiles["default"])

    def retention_policy(self) -> RetentionPolicy:
        return RetentionPolicy.from_settings(self.settings)

    async def publish(self, event: dict) -> None:
        await self.hub.broadcast(event)

    def health(self) -> Dict[str, Any]:
        return {
            "status": "degraded" if self.registry.is_degraded else "ok",
            "version": self.settings.version,
            "environment": self.settings.environment,
            "model_loaded": self.registry.bundle is not None,
            "degraded": list(self.registry.degraded),
            "detectors": {name: d.describe()
                          for name, d in self.registry.detectors().items()},
            "retention": self.retention_policy().as_dict(),
            "sessions": self.sessions.stats(),
            "alerts": self.alerts.describe(),
            "dashboard_clients": self.hub.client_count,
            "uptime_seconds": round(time.time() - self.started_at, 1),
        }

    def shutdown(self) -> None:
        for session_id in [row["session_id"] for row in self.sessions.list_sessions()]:
            try:
                self.sessions.close(session_id)
            except Exception:
                pass


# --------------------------------------------------------------------------------------
# FastAPI dependencies
# --------------------------------------------------------------------------------------

def get_state(request: Request) -> AppState:
    return request.app.state.vg


def get_state_ws(websocket: WebSocket) -> AppState:
    return websocket.app.state.vg


def _valid_key(state: AppState, key: Optional[str]) -> bool:
    return bool(key) and key in state.settings.api_keys


def require_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
) -> None:
    """Authenticate a request when ``VG_AUTH_REQUIRED=true``.

    Accepts either an ``X-API-Key`` header or a bearer token. Auth is off by default so a
    fresh clone demos without setup; ``deploy/`` turns it on, and the health endpoint
    reports which mode is active so nobody ships the open one by accident.
    """
    state: AppState = request.app.state.vg
    if not state.settings.auth_required:
        return

    key = x_api_key
    if not key and authorization and authorization.lower().startswith("bearer "):
        key = authorization[7:].strip()

    if _valid_key(state, key):
        return

    if key:
        try:
            import jwt

            jwt.decode(key, state.settings.jwt_secret, algorithms=["HS256"])
            return
        except Exception:
            pass

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Provide a valid X-API-Key header or bearer token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def check_consent(
    request: Request,
    x_consent: Optional[str] = Header(None, alias="X-Consent"),
) -> None:
    """Enforce the consent header when the deployment requires it."""
    state: AppState = request.app.state.vg
    if not state.settings.require_consent_header:
        return
    if (x_consent or "").lower() not in ("recorded", "analysed", "analyzed", "none-required"):
        raise HTTPException(
            status_code=status.HTTP_428_PRECONDITION_REQUIRED,
            detail=(
                "This deployment requires an X-Consent header "
                "(recorded | analysed | none-required)."
            ),
        )


async def authorize_websocket(
    websocket: WebSocket,
    api_key: Optional[str] = Query(None, alias="api_key"),
) -> bool:
    """WebSocket auth. Browsers cannot set headers on a WS handshake, so the key
    travels as a query parameter — acceptable only because the transport is TLS in any
    real deployment (``deploy/`` terminates it at the proxy)."""
    state: AppState = websocket.app.state.vg
    if not state.settings.auth_required or _valid_key(state, api_key):
        return True
    await websocket.close(code=4401, reason="unauthorized")
    return False
