# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Structured security alerts and append-only event sink."""

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("gpufabric.alerts")

DEFAULT_ALERTS_PATH = Path.home() / ".local" / "state" / "gpufabric" / "alerts.jsonl"


@dataclass
class Alert:
    """Security or integrity alert event.

    CRITICAL RULE: NEVER log secret tokens, private keys, or raw tensor payloads.
    """

    timestamp: str  # ISO 8601 UTC
    kind: str  # decoy_contact | canary_hit | verify_fail | probe_fail
    subject: str  # node or user id e.g. "node:worker-01" or "user:izpiz"
    peer: str  # network address string e.g. "ipv4:192.168.1.100:54321"
    method: str  # RPC or verification method name
    cert_cn: str  # authenticated TLS certificate common name (if any)
    detail: str  # human-readable incident description

    @classmethod
    def create(
        cls,
        kind: str,
        subject: str,
        peer: str = "",
        method: str = "",
        cert_cn: str = "",
        detail: str = "",
        timestamp: Optional[str] = None,
    ) -> "Alert":
        """Factory constructor generating UTC ISO timestamp automatically."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        return cls(
            timestamp=timestamp,
            kind=kind,
            subject=subject,
            peer=peer,
            method=method,
            cert_cn=cert_cn,
            detail=detail,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Alert":
        return cls(
            timestamp=data.get("timestamp", ""),
            kind=data.get("kind", ""),
            subject=data.get("subject", ""),
            peer=data.get("peer", ""),
            method=data.get("method", ""),
            cert_cn=data.get("cert_cn", ""),
            detail=data.get("detail", ""),
        )


class AlertSink:
    """Thread-safe append-only sink for security alerts with in-process subscribers."""

    def __init__(self, log_path: Optional[Path] = None):
        if log_path is None:
            env_path = os.environ.get("GPUFABRIC_ALERTS_PATH")
            self.log_path = Path(env_path) if env_path else DEFAULT_ALERTS_PATH
        else:
            self.log_path = Path(log_path)

        self._lock = threading.Lock()
        self._subscribers: List[Callable[[Alert], None]] = []

    def subscribe(self, callback: Callable[[Alert], None]) -> None:
        """Register a callback for real-time alert dispatch."""
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Alert], None]) -> None:
        """Remove a previously registered subscriber callback."""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def emit(self, alert: Alert) -> None:
        """Append alert to JSONL log and notify registered subscribers."""
        line = json.dumps(alert.to_dict()) + "\n"
        with self._lock:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush()
            except Exception as e:
                logger.error(f"Failed to append alert to {self.log_path}: {e}")

            subscribers_snapshot = list(self._subscribers)

        for callback in subscribers_snapshot:
            try:
                callback(alert)
            except Exception as e:
                logger.warning(f"Alert subscriber callback failed: {e}")

    def read_alerts(self, limit: Optional[int] = None, kind: Optional[str] = None) -> List[Alert]:
        """Read recorded alerts from log file with optional filtering."""
        if not self.log_path.exists():
            return []

        alerts: List[Alert] = []
        with self._lock:
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = Alert.from_dict(json.loads(line))
                            if kind is None or item.kind == kind:
                                alerts.append(item)
                        except Exception:
                            continue
            except Exception as e:
                logger.error(f"Failed reading alerts from {self.log_path}: {e}")
                return []

        if limit is not None and len(alerts) > limit:
            alerts = alerts[-limit:]
        return alerts


# Process-wide default AlertSink instance
_DEFAULT_SINK = AlertSink()


def get_default_alert_sink() -> AlertSink:
    return _DEFAULT_SINK


def emit_alert(
    kind: str,
    subject: str,
    peer: str = "",
    method: str = "",
    cert_cn: str = "",
    detail: str = "",
) -> Alert:
    """Convenience helper to emit an alert to the default AlertSink."""
    alert = Alert.create(
        kind=kind,
        subject=subject,
        peer=peer,
        method=method,
        cert_cn=cert_cn,
        detail=detail,
    )
    _DEFAULT_SINK.emit(alert)
    return alert


def extract_cert_cn(context: Any) -> str:
    """Safely extract authenticated client certificate Common Name from gRPC context."""
    if context is None:
        return ""
    try:
        auth_context = context.auth_context()
        if not auth_context:
            return ""
        cn_entries = auth_context.get("x509_common_name", [])
        if cn_entries:
            val = cn_entries[0]
            return val.decode("utf-8") if isinstance(val, bytes) else str(val)
    except Exception:
        pass
    return ""
