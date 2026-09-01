"""SQLite audit store.

Deliberately SQLite: the audit trail must survive a restart and be queryable, but adding
a database server to a demo is a reliability tax with no upside. The schema is plain
enough to port to Postgres unchanged when it needs to be.

What is stored is governed by the retention mode (see ``docs/PRIVACY.md``): in the default
``features_only`` mode the ``audio_sha256`` column holds a hash and nothing else touches
the audio.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    created_at      REAL NOT NULL,
    closed_at       REAL,
    profile         TEXT,
    language        TEXT,
    identity        TEXT,
    verdict         TEXT,
    final_score     REAL,
    peak_score      REAL,
    duration_seconds REAL,
    call_context    TEXT,
    report          TEXT
);

CREATE TABLE IF NOT EXISTS assessments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    created_at      REAL NOT NULL,
    window_index    INTEGER,
    score           REAL,
    band            TEXT,
    confidence      REAL,
    latency_ms      REAL,
    factors         TEXT,
    layers          TEXT,
    features        TEXT,
    audio_sha256    TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    created_at      REAL NOT NULL,
    band            TEXT,
    score           REAL,
    payload         TEXT
);

CREATE TABLE IF NOT EXISTS enrolments (
    identity        TEXT NOT NULL,
    created_at      REAL NOT NULL,
    vector          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_assessments_session ON assessments(session_id);
CREATE INDEX IF NOT EXISTS idx_assessments_created ON assessments(created_at);
CREATE INDEX IF NOT EXISTS idx_alerts_session ON alerts(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at);
"""


class Database:
    """Thin, thread-safe SQLite wrapper."""

    def __init__(self, path: str = "runtime/voiceguard.sqlite3") -> None:
        self.path = path
        self._lock = threading.RLock()
        if path != ":memory:":
            directory = os.path.dirname(os.path.abspath(path))
            os.makedirs(directory, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            # WAL keeps readers (the dashboard) from blocking the ingest writer.
            if path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.commit()

    # -------------------------------------------------------------------- basics
    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cursor = self._conn.execute(sql, params)
            self._conn.commit()
            return cursor

    def query(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def query_one(self, sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def dumps(value: Any) -> str:
        return json.dumps(value, default=str, separators=(",", ":"))

    @staticmethod
    def loads(value: Optional[str], default: Any = None) -> Any:
        if not value:
            return default
        try:
            return json.loads(value)
        except Exception:
            return default

    def vacuum(self) -> None:
        with self._lock:
            self._conn.execute("VACUUM")
            self._conn.commit()

    def counts(self) -> Dict[str, int]:
        return {
            table: int(self.query_one(f"SELECT COUNT(*) AS n FROM {table}")["n"])
            for table in ("sessions", "assessments", "alerts", "enrolments")
        }


_DB: Optional[Database] = None


def get_database(path: Optional[str] = None) -> Database:
    global _DB
    if _DB is None:
        from voiceguard.config import get_settings

        _DB = Database(path or get_settings().database_url)
    return _DB


def reset_database() -> None:
    global _DB
    if _DB is not None:
        _DB.close()
    _DB = None
