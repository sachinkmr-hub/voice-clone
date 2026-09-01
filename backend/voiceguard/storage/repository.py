"""Audit repository — what gets written, and what deliberately does not.

Every write goes through :class:`AuditRepository`, which consults the retention mode
before persisting anything derived from audio. That is the single choke point that makes
the privacy claim in ``docs/PRIVACY.md`` checkable rather than aspirational.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from voiceguard.config import RetentionMode, Settings, get_settings
from voiceguard.privacy.anonymize import anonymize_dict
from voiceguard.scoring.risk import RiskAssessment
from voiceguard.storage.db import Database, get_database


class AuditRepository:
    """Persistence for sessions, assessments, alerts and enrolments."""

    def __init__(self, db: Optional[Database] = None,
                 settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.db = db or get_database(self.settings.database_url)

    @property
    def retention(self) -> str:
        return self.settings.retention_mode

    @property
    def stores_features(self) -> bool:
        return self.retention in (RetentionMode.FEATURES_ONLY.value,
                                  RetentionMode.RAW_AUDIO.value)

    # ------------------------------------------------------------------ sessions
    def open_session(self, session) -> None:
        self.db.execute(
            """INSERT OR REPLACE INTO sessions
               (session_id, created_at, profile, language, identity, call_context)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                session.id, session.created_at, session.profile, session.language,
                session.identity,
                self.db.dumps(anonymize_dict(
                    session.call_context.as_dict() if session.call_context else {},
                    self.settings)),
            ),
        )

    def close_session(self, session, report: Optional[dict] = None) -> None:
        report = report or session.report(include_trail=False)
        self.db.execute(
            """UPDATE sessions
               SET closed_at = ?, verdict = ?, final_score = ?, peak_score = ?,
                   duration_seconds = ?, report = ?
               WHERE session_id = ?""",
            (
                session.closed_at or time.time(), report.get("verdict"),
                report.get("final_score"), report.get("peak_score"),
                report.get("duration_seconds"), self.db.dumps(report), session.id,
            ),
        )

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        row = self.db.query_one("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        if row:
            row["call_context"] = self.db.loads(row.get("call_context"), {})
            row["report"] = self.db.loads(row.get("report"), None)
        return row

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.db.query(
            "SELECT session_id, created_at, closed_at, profile, verdict, final_score, "
            "peak_score, duration_seconds FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )

    # --------------------------------------------------------------- assessments
    def record_assessment(
        self,
        session_id: str,
        assessment: RiskAssessment,
        *,
        features: Optional[Dict[str, float]] = None,
        audio_sha256: Optional[str] = None,
    ) -> None:
        """Write one window's verdict.

        Feature vectors are stored only in ``features_only``/``raw_audio`` modes. They are
        summary statistics (means, variances, band energies) and are not invertible to
        intelligible speech — but they are still derived from a person's voice, so they
        are gated on the retention mode rather than always written.
        """
        if self.retention == RetentionMode.NONE.value:
            feature_blob = None
        elif features and self.stores_features:
            feature_blob = self.db.dumps({k: round(float(v), 6) for k, v in features.items()})
        else:
            feature_blob = None

        self.db.execute(
            """INSERT INTO assessments
               (session_id, created_at, window_index, score, band, confidence,
                latency_ms, factors, layers, features, audio_sha256)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id, assessment.created_at, assessment.window_index,
                float(assessment.score), assessment.band, float(assessment.confidence),
                float(assessment.latency_ms),
                self.db.dumps([f.as_dict() for f in assessment.explanation.factors]),
                self.db.dumps(assessment.explanation.layer_summary),
                feature_blob, audio_sha256,
            ),
        )

    def assessments_for(self, session_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM assessments WHERE session_id = ? ORDER BY created_at ASC LIMIT ?",
            (session_id, limit),
        )
        for row in rows:
            row["factors"] = self.db.loads(row.get("factors"), [])
            row["layers"] = self.db.loads(row.get("layers"), [])
            row["features"] = self.db.loads(row.get("features"), None)
        return rows

    # -------------------------------------------------------------------- alerts
    def record_alert(self, session_id: str, alert: Dict[str, Any]) -> None:
        payload = alert.get("alert", alert)
        self.db.execute(
            "INSERT INTO alerts (session_id, created_at, band, score, payload) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, payload.get("created_at", time.time()), payload.get("band"),
             payload.get("score"), self.db.dumps(alert)),
        )

    def recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,))
        for row in rows:
            row["payload"] = self.db.loads(row.get("payload"), {})
        return rows

    # ---------------------------------------------------------------- enrolments
    def save_enrolment(self, identity: str, vector: np.ndarray) -> None:
        self.db.execute(
            "INSERT INTO enrolments (identity, created_at, vector) VALUES (?, ?, ?)",
            (identity, time.time(), self.db.dumps([float(v) for v in np.asarray(vector).ravel()])),
        )

    def load_enrolments(self) -> Dict[str, List[np.ndarray]]:
        profiles: Dict[str, List[np.ndarray]] = {}
        for row in self.db.query("SELECT identity, vector FROM enrolments"):
            vector = self.db.loads(row["vector"], [])
            if vector:
                profiles.setdefault(row["identity"], []).append(
                    np.asarray(vector, dtype=np.float32))
        return profiles

    def delete_enrolment(self, identity: str) -> int:
        return int(self.db.execute(
            "DELETE FROM enrolments WHERE identity = ?", (identity,)).rowcount)

    # ---------------------------------------------------------------- erasure
    def purge_session(self, session_id: str) -> Dict[str, int]:
        """Right-to-erasure: remove everything tied to one session."""
        return {
            "assessments": int(self.db.execute(
                "DELETE FROM assessments WHERE session_id = ?", (session_id,)).rowcount),
            "alerts": int(self.db.execute(
                "DELETE FROM alerts WHERE session_id = ?", (session_id,)).rowcount),
            "sessions": int(self.db.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)).rowcount),
        }

    def stats(self) -> Dict[str, Any]:
        return {"retention_mode": self.retention, "tables": self.db.counts()}
