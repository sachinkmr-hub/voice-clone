"""Retention enforcement.

A retention *policy* that nothing enforces is a paragraph in a slide deck. This module is
the enforcement: a periodic sweeper that hard-deletes expired rows, and helpers that make
the current policy visible to operators and to the console banner.

Deletion is real (``DELETE``, then ``VACUUM`` on demand), not a soft flag — a soft-deleted
voice feature is still a stored voice feature.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

from voiceguard.config import RetentionMode, Settings, get_settings

if TYPE_CHECKING:  # storage imports privacy for anonymisation, so keep this type-only
    from voiceguard.storage.repository import AuditRepository

logger = logging.getLogger("voiceguard.privacy")


@dataclass
class RetentionPolicy:
    """The active policy, in a form that can be displayed and asserted on."""

    mode: str
    ttl_seconds: int
    raw_audio_ttl_seconds: int
    store_pii: bool

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "RetentionPolicy":
        settings = settings or get_settings()
        return cls(
            mode=settings.retention_mode,
            ttl_seconds=settings.retention_ttl_seconds,
            raw_audio_ttl_seconds=settings.raw_audio_ttl_seconds,
            store_pii=settings.store_pii,
        )

    @property
    def keeps_raw_audio(self) -> bool:
        return self.mode == RetentionMode.RAW_AUDIO.value

    @property
    def keeps_features(self) -> bool:
        return self.mode in (RetentionMode.FEATURES_ONLY.value, RetentionMode.RAW_AUDIO.value)

    def banner(self) -> str:
        """The one-line notice the console displays and must not hide."""
        if self.mode == RetentionMode.NONE.value:
            return "Privacy mode: scores only — no audio and no voice features are stored."
        if self.mode == RetentionMode.FEATURES_ONLY.value:
            return (
                f"Privacy mode: features only — no audio is written to disk. "
                f"Feature vectors are deleted after {self.ttl_seconds // 3600} h."
            )
        return (
            f"Privacy mode: RAW AUDIO RETAINED for {self.raw_audio_ttl_seconds // 60} min. "
            f"This is opt-in and must be justified to your DPO."
        )

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "ttl_seconds": self.ttl_seconds,
            "raw_audio_ttl_seconds": self.raw_audio_ttl_seconds,
            "store_pii": self.store_pii,
            "keeps_raw_audio": self.keeps_raw_audio,
            "keeps_features": self.keeps_features,
            "banner": self.banner(),
        }


class RetentionSweeper:
    """Periodically hard-deletes anything past its TTL."""

    def __init__(self, repository: "AuditRepository", settings: Optional[Settings] = None,
                 interval_seconds: float = 300.0) -> None:
        self.repository = repository
        self.settings = settings or get_settings()
        self.interval = interval_seconds
        self._task: Optional[asyncio.Task] = None
        self.last_run: Optional[float] = None
        self.last_result: Dict[str, int] = {}

    def sweep(self) -> Dict[str, int]:
        """One pass. Returns the number of rows removed per table."""
        now = time.time()
        cutoff = now - self.settings.retention_ttl_seconds
        db = self.repository.db

        removed = {
            "assessments": int(db.execute(
                "DELETE FROM assessments WHERE created_at < ?", (cutoff,)).rowcount),
            "alerts": int(db.execute(
                "DELETE FROM alerts WHERE created_at < ?", (cutoff,)).rowcount),
            "sessions": int(db.execute(
                "DELETE FROM sessions WHERE created_at < ? AND closed_at IS NOT NULL",
                (cutoff,)).rowcount),
        }

        # Feature vectors get a shorter life than the score itself: the score is the
        # audit record, the features are only there for dispute resolution.
        feature_cutoff = now - min(self.settings.retention_ttl_seconds,
                                   max(self.settings.raw_audio_ttl_seconds * 4, 3600))
        removed["feature_blobs"] = int(db.execute(
            "UPDATE assessments SET features = NULL "
            "WHERE features IS NOT NULL AND created_at < ?", (feature_cutoff,)).rowcount)

        self.last_run = now
        self.last_result = removed
        if any(removed.values()):
            logger.info("retention sweep removed %s", removed)
        return removed

    def expire_session_audio(self, session) -> int:
        """Drop retained audio buffers once past the raw-audio TTL."""
        if not session.retained_audio:
            return 0
        age = time.time() - session.created_at
        if age > self.settings.raw_audio_ttl_seconds:
            count = len(session.retained_audio)
            session.retained_audio.clear()
            return count
        return 0

    # ------------------------------------------------------------- async runner
    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.interval)
                self.sweep()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # a sweeper crash must not take the API down
                logger.warning("retention sweep failed: %s", exc)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    def describe(self) -> dict:
        return {
            "policy": RetentionPolicy.from_settings(self.settings).as_dict(),
            "interval_seconds": self.interval,
            "last_run": self.last_run,
            "last_result": dict(self.last_result),
            "running": bool(self._task and not self._task.done()),
        }
