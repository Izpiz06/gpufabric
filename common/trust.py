# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Reputation and trust scoring engine using a Beta-reputation model."""

import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("gpufabric.trust")

DEFAULT_TRUST_DB_PATH = Path.home() / ".local" / "state" / "gpufabric" / "trust.db"

# Standard event weights
DEFAULT_WEIGHTS = {
    "VERIFY_PASS": 1.0,
    "VERIFY_FAIL": 3.0,
    "PROBE_FAIL": 5.0,
    "DECOY_CONTACT": 8.0,
    "CANARY_HIT": 8.0,
}

# Thresholds
WARN_THRESHOLD = 0.60
DRAIN_THRESHOLD = 0.35
RECOVERY_THRESHOLD = 0.70
MIN_DRAIN_EVIDENCE = 3  # Minimum evidence count required to trigger DRAIN


class TrustStore:
    """Thread-safe SQLite store implementing a Beta-reputation trust model."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        weights: Optional[Dict[str, float]] = None,
        decay_factor: float = 0.98,
        min_drain_evidence: int = MIN_DRAIN_EVIDENCE,
    ):
        if db_path is None:
            env_path = os.environ.get("GPUFABRIC_TRUST_DB")
            self.db_path = Path(env_path) if env_path else DEFAULT_TRUST_DB_PATH
        else:
            self.db_path = Path(db_path)

        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

        self.decay_factor = float(decay_factor)
        self.min_drain_evidence = int(min_drain_evidence)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS entities (
                        entity_id TEXT PRIMARY KEY,
                        entity_type TEXT NOT NULL,
                        alpha REAL NOT NULL,
                        beta REAL NOT NULL,
                        score REAL NOT NULL,
                        state TEXT NOT NULL,
                        total_events INTEGER DEFAULT 0,
                        fail_count INTEGER DEFAULT 0,
                        consecutive_passes INTEGER DEFAULT 0,
                        last_event_at TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS trust_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        entity TEXT NOT NULL,
                        event TEXT NOT NULL,
                        weight REAL NOT NULL,
                        old_score REAL NOT NULL,
                        new_score REAL NOT NULL,
                        old_state TEXT NOT NULL,
                        new_state TEXT NOT NULL,
                        detail TEXT
                    )
                    """
                )
                conn.commit()

    def record_event(
        self,
        entity: str,
        event: str,
        weight: Optional[float] = None,
        detail: str = "",
    ) -> Tuple[float, str]:
        """Record an audit or verification event, update Beta score, and evaluate state.

        Returns (new_score, new_state).
        """
        now = datetime.now(timezone.utc).isoformat()
        w = float(weight if weight is not None else self.weights.get(event, 1.0))
        event_upper = event.upper()
        is_pass = event_upper == "VERIFY_PASS"

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT entity_type, alpha, beta, score, state, total_events,
                           fail_count, consecutive_passes
                    FROM entities WHERE entity_id = ?
                    """,
                    (entity,),
                )
                row = cursor.fetchone()

                if row is None:
                    entity_type = "user" if entity.startswith("user:") else "node"
                    alpha = 2.0  # Initial uninformative-positive prior
                    beta = 1.0
                    old_score = alpha / (alpha + beta)
                    old_state = "HEALTHY"
                    total_events = 0
                    fail_count = 0
                    consecutive_passes = 0
                else:
                    (
                        entity_type,
                        alpha,
                        beta,
                        old_score,
                        old_state,
                        total_events,
                        fail_count,
                        consecutive_passes,
                    ) = row

                # Apply gradual forgetting/decay to past evidence towards prior (2.0, 1.0)
                alpha = 2.0 + (alpha - 2.0) * self.decay_factor
                beta = 1.0 + (beta - 1.0) * self.decay_factor

                # Accumulate new event evidence
                if is_pass:
                    alpha += w
                    consecutive_passes += 1
                else:
                    beta += w
                    fail_count += 1
                    consecutive_passes = 0

                total_events += 1
                new_score = float(alpha / (alpha + beta))

                # State Transition Logic (Hysteresis & Single-Failure Protection)
                if entity_type == "user":
                    if new_score < DRAIN_THRESHOLD:
                        new_state = "RESTRICTED"
                    elif new_score < WARN_THRESHOLD:
                        new_state = "SUSPECT"
                    else:
                        new_state = "HEALTHY"
                else:
                    # Node evaluation:
                    # CRITICAL: A single failure marks the node SUSPECT, NEVER DRAIN.
                    # DRAIN requires repeated failures crossing the threshold with minimum evidence.
                    if old_state == "DRAIN":
                        # Hysteresis recovery: requires climbing back above RECOVERY_THRESHOLD
                        if new_score >= RECOVERY_THRESHOLD and consecutive_passes >= 3:
                            new_state = "HEALTHY"
                        else:
                            new_state = "DRAIN"
                    else:
                        if (
                            new_score < DRAIN_THRESHOLD
                            and fail_count >= 2
                            and total_events >= self.min_drain_evidence
                        ):
                            new_state = "DRAIN"
                        elif new_score < WARN_THRESHOLD or fail_count == 1:
                            new_state = "SUSPECT"
                        else:
                            new_state = "HEALTHY"

                conn.execute(
                    """
                    INSERT INTO entities (
                        entity_id, entity_type, alpha, beta, score, state,
                        total_events, fail_count, consecutive_passes, last_event_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(entity_id) DO UPDATE SET
                        alpha = excluded.alpha,
                        beta = excluded.beta,
                        score = excluded.score,
                        state = excluded.state,
                        total_events = excluded.total_events,
                        fail_count = excluded.fail_count,
                        consecutive_passes = excluded.consecutive_passes,
                        last_event_at = excluded.last_event_at
                    """,
                    (
                        entity,
                        entity_type,
                        alpha,
                        beta,
                        new_score,
                        new_state,
                        total_events,
                        fail_count,
                        consecutive_passes,
                        now,
                    ),
                )

                conn.execute(
                    """
                    INSERT INTO trust_events (
                        timestamp, entity, event, weight, old_score, new_score,
                        old_state, new_state, detail
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        now,
                        entity,
                        event_upper,
                        w,
                        old_score,
                        new_score,
                        old_state,
                        new_state,
                        detail,
                    ),
                )
                conn.commit()

        return new_score, new_state

    def get_entity(self, entity: str) -> Optional[Dict[str, Any]]:
        """Retrieve entity summary and trust status."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT entity_id, entity_type, alpha, beta, score, state,
                           total_events, fail_count, consecutive_passes, last_event_at
                    FROM entities WHERE entity_id = ?
                    """,
                    (entity,),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                return {
                    "entity_id": row[0],
                    "entity_type": row[1],
                    "alpha": row[2],
                    "beta": row[3],
                    "score": row[4],
                    "state": row[5],
                    "total_events": row[6],
                    "fail_count": row[7],
                    "consecutive_passes": row[8],
                    "last_event_at": row[9],
                }

    def list_entities(self, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all tracked entities, optionally filtered by node or user."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                if entity_type:
                    cursor = conn.execute(
                        "SELECT entity_id, entity_type, alpha, beta, score, state, total_events, last_event_at FROM entities WHERE entity_type = ? ORDER BY score ASC",
                        (entity_type,),
                    )
                else:
                    cursor = conn.execute(
                        "SELECT entity_id, entity_type, alpha, beta, score, state, total_events, last_event_at FROM entities ORDER BY score ASC"
                    )
                rows = cursor.fetchall()

        return [
            {
                "entity_id": r[0],
                "entity_type": r[1],
                "alpha": r[2],
                "beta": r[3],
                "score": r[4],
                "state": r[5],
                "total_events": r[6],
                "last_event_at": r[7],
            }
            for r in rows
        ]

    def get_events(self, entity: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch historical trust score change events."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                if entity:
                    cursor = conn.execute(
                        "SELECT timestamp, entity, event, weight, old_score, new_score, old_state, new_state, detail FROM trust_events WHERE entity = ? ORDER BY id DESC LIMIT ?",
                        (entity, limit),
                    )
                else:
                    cursor = conn.execute(
                        "SELECT timestamp, entity, event, weight, old_score, new_score, old_state, new_state, detail FROM trust_events ORDER BY id DESC LIMIT ?",
                        (limit,),
                    )
                rows = cursor.fetchall()

        return [
            {
                "timestamp": r[0],
                "entity": r[1],
                "event": r[2],
                "weight": r[3],
                "old_score": r[4],
                "new_score": r[5],
                "old_state": r[6],
                "new_state": r[7],
                "detail": r[8],
            }
            for r in rows
        ]

    def reset_entity(self, entity: str) -> bool:
        """Reset an entity back to initial clean state."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM entities WHERE entity_id = ?", (entity,))
                conn.commit()
                return cursor.rowcount > 0


_DEFAULT_TRUST_STORE: Optional[TrustStore] = None


def get_default_trust_store() -> TrustStore:
    global _DEFAULT_TRUST_STORE
    if _DEFAULT_TRUST_STORE is None:
        _DEFAULT_TRUST_STORE = TrustStore()
    return _DEFAULT_TRUST_STORE
