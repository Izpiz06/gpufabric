# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Canary token registry and decoy traps for unauthorized access detection.

Provides cryptographic canary generation and tripwire checking persisted in SQLite.
Whenever an adversary or compromised node accesses or supplies a canary token,
a high-priority 'canary_hit' alert is emitted.

Future Gateway Integration Touchpoints:
  1. Job ID Lookup: Call check_canary(job_id, context) in GetJob/QueryStatus handlers.
  2. Result Artifact Fetch: Call check_canary(artifact_id, context) in result download endpoints.
  3. Staging Storage: Seed canary filenames in directory listings; call check_canary
     whenever a client attempts to read staged tensor files.
"""

import os
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.alerts import AlertSink, emit_alert, extract_cert_cn, get_default_alert_sink

DEFAULT_CANARY_DB_PATH = Path.home() / ".local" / "state" / "gpufabric" / "canaries.db"


class CanaryRegistry:
    """Thread-safe SQLite store for active canary tokens and hit tracking."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        alert_sink: Optional[AlertSink] = None,
    ):
        if db_path is None:
            env_path = os.environ.get("GPUFABRIC_CANARY_DB")
            self.db_path = Path(env_path) if env_path else DEFAULT_CANARY_DB_PATH
        else:
            self.db_path = Path(db_path)

        self.alert_sink = alert_sink or get_default_alert_sink()
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS canaries (
                        token TEXT PRIMARY KEY,
                        kind TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        hit_count INTEGER DEFAULT 0,
                        last_hit_at TEXT
                    )
                    """
                )
                conn.commit()

    def mint(self, kind: str = "job_id", prefix: str = "cnry_") -> str:
        """Mint a new opaque cryptographic canary token and persist to registry."""
        raw_token = f"{prefix}{secrets.token_urlsafe(24)}"
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO canaries (token, kind, created_at, hit_count) VALUES (?, ?, ?, 0)",
                    (raw_token, kind, now),
                )
                conn.commit()
        return raw_token

    def check(self, value: str, context: Optional[Any] = None) -> bool:
        """Check if `value` matches a registered canary token.

        If a match is found, increments hit counter and emits a 'canary_hit' alert.
        """
        if not value or not isinstance(value, str):
            return False

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT kind, hit_count FROM canaries WHERE token = ?", (value,)
                )
                row = cursor.fetchone()
                if not row:
                    return False

                kind, hit_count = row[0], row[1]
                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "UPDATE canaries SET hit_count = hit_count + 1, last_hit_at = ? WHERE token = ?",
                    (now, value),
                )
                conn.commit()

        peer = context.peer() if context else ""
        cert_cn = extract_cert_cn(context)
        subject_id = f"user:{cert_cn}" if cert_cn else "user:unknown"

        # Truncate token for safe logging (never log full token or secret)
        safe_tag = f"{value[:8]}..." if len(value) > 8 else value
        detail = f"Canary token '{safe_tag}' triggered (kind={kind}, hits={hit_count + 1})."

        alert = emit_alert(
            kind="canary_hit",
            subject=subject_id,
            peer=peer,
            method="canary_check",
            cert_cn=cert_cn,
            detail=detail,
        )
        if self.alert_sink is not None and self.alert_sink != get_default_alert_sink():
            self.alert_sink.emit(alert)

        return True

    def list_canaries(self) -> List[Dict[str, Any]]:
        """List active canary tokens (tokens truncated for safe inspection)."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT token, kind, created_at, hit_count, last_hit_at FROM canaries ORDER BY created_at DESC"
                )
                rows = cursor.fetchall()

        return [
            {
                "token_prefix": f"{r[0][:10]}...",
                "kind": r[1],
                "created_at": r[2],
                "hit_count": r[3],
                "last_hit_at": r[4],
            }
            for r in rows
        ]


_DEFAULT_REGISTRY: Optional[CanaryRegistry] = None


def get_default_canary_registry() -> CanaryRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = CanaryRegistry()
    return _DEFAULT_REGISTRY


def check_canary(value: str, context: Optional[Any] = None) -> bool:
    """Global tripwire helper for checking incoming IDs against canary registry."""
    return get_default_canary_registry().check(value, context=context)
