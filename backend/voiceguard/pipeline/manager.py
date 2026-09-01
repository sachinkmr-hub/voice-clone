"""Session lifecycle management.

Holds live :class:`CallSession` objects, enforces a ceiling on concurrency, and evicts
idle ones. The interface is deliberately narrow (create / get / close / prune) so that
swapping the in-memory dict for Redis — the change needed to drop the sticky-routing
requirement and scale out freely — touches only this file.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional

from voiceguard.config import Settings, get_settings
from voiceguard.features.extractor import FeatureExtractor
from voiceguard.models.context import CallContext
from voiceguard.models.registry import ModelRegistry, get_registry
from voiceguard.pipeline.session import CallSession


class SessionLimitExceeded(RuntimeError):
    """Raised when the node is already at its configured session ceiling."""


class SessionManager:
    """Thread-safe registry of live sessions."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        registry: Optional[ModelRegistry] = None,
        extractor: Optional[FeatureExtractor] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or get_registry()
        # One shared extractor: it is stateless apart from its filterbank caches, and
        # sharing it means those caches are built once per process rather than per call.
        self.extractor = extractor or FeatureExtractor()
        self._sessions: Dict[str, CallSession] = {}
        self._lock = threading.RLock()
        self._closed: List[dict] = []          # small ring of finished-call reports
        self.on_close: Optional[Callable[[CallSession], None]] = None

    # ------------------------------------------------------------------ lifecycle
    def create(
        self,
        *,
        session_id: Optional[str] = None,
        profile: Optional[str] = None,
        language: str = "auto",
        identity: Optional[str] = None,
        call_context: Optional[CallContext] = None,
        metadata: Optional[dict] = None,
    ) -> CallSession:
        with self._lock:
            self.prune()
            if len(self._sessions) >= self.settings.max_active_sessions:
                raise SessionLimitExceeded(
                    f"node is at its ceiling of {self.settings.max_active_sessions} "
                    f"concurrent sessions"
                )
            session = CallSession(
                session_id=session_id,
                profile=profile or self.settings.default_profile,
                language=language,
                identity=identity,
                call_context=call_context,
                registry=self.registry,
                extractor=self.extractor,
                settings=self.settings,
                metadata=metadata,
            )
            self._sessions[session.id] = session
            return session

    def get(self, session_id: str) -> Optional[CallSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def require(self, session_id: str) -> CallSession:
        session = self.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def close(self, session_id: str) -> Optional[dict]:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return None
        session.close()
        report = session.report(include_trail=False)
        self._closed.append(report)
        if len(self._closed) > 200:
            self._closed = self._closed[-200:]
        if self.on_close is not None:
            try:
                self.on_close(session)
            except Exception:
                pass
        return report

    def delete(self, session_id: str) -> bool:
        """Hard removal — used by the right-to-erasure endpoint."""
        with self._lock:
            existed = self._sessions.pop(session_id, None) is not None
            before = len(self._closed)
            self._closed = [r for r in self._closed if r.get("session_id") != session_id]
            return existed or len(self._closed) != before

    def prune(self) -> List[str]:
        """Evict sessions idle longer than the configured timeout."""
        timeout = self.settings.session_idle_timeout_seconds
        now = time.time()
        evicted: List[str] = []
        with self._lock:
            for session_id, session in list(self._sessions.items()):
                if now - session.last_activity > timeout:
                    self._sessions.pop(session_id, None)
                    session.close()
                    self._closed.append(session.report(include_trail=False))
                    evicted.append(session_id)
        return evicted

    # -------------------------------------------------------------------- queries
    def list_sessions(self, *, include_closed: bool = False, limit: int = 100) -> List[dict]:
        with self._lock:
            live = [s.as_dict() for s in self._sessions.values()]
        live.sort(key=lambda row: -row["last_activity"])
        if include_closed:
            live.extend(self._closed[-limit:][::-1])
        return live[:limit]

    def closed_reports(self, limit: int = 50) -> List[dict]:
        return list(self._closed[-limit:][::-1])

    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def stats(self) -> dict:
        with self._lock:
            sessions = list(self._sessions.values())
        latencies = [s.stats.mean_latency_ms for s in sessions if s.stats.windows_analyzed]
        return {
            "active_sessions": len(sessions),
            "closed_sessions": len(self._closed),
            "capacity": self.settings.max_active_sessions,
            "windows_analyzed": sum(s.stats.windows_analyzed for s in sessions),
            "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "alerting_sessions": sum(1 for s in sessions if s.current_score >= 60),
        }
