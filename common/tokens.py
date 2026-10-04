"""Enrollment token generation, validation, and storage."""

import json
import os
import secrets
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from common.constants import DEFAULT_TOKEN_TTL_SECONDS
from common.tls import default_tls_dir


@dataclass
class EnrollmentToken:
    """Represents a time-limited, single-use client enrollment token."""

    token: str
    client_name: str
    created_at: float
    expires_at: float
    used: bool = False

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def is_valid(self) -> bool:
        return not self.used and not self.is_expired


class TokenStore:
    """Thread-safe persistent store for enrollment tokens."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or (default_tls_dir() / "enrollment_tokens.json")
        self._lock = threading.Lock()
        self._tokens: Dict[str, EnrollmentToken] = {}
        self._load()

    def _load(self) -> None:
        if not self.storage_path.is_file():
            return
        try:
            mtime = self.storage_path.stat().st_mtime
            if hasattr(self, "_last_mtime") and self._last_mtime == mtime:
                return
            raw = self.storage_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            tokens = {}
            for item in data:
                token = EnrollmentToken(**item)
                tokens[token.token] = token
            self._tokens = tokens
            self._last_mtime = mtime
        except Exception:
            # If the file is corrupt or unreadable, keep existing or start fresh
            if not hasattr(self, "_tokens"):
                self._tokens = {}

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure owner-only permissions on directory
        try:
            os.chmod(self.storage_path.parent, 0o700)
        except OSError:
            pass

        data = [asdict(t) for t in self._tokens.values()]
        payload = json.dumps(data, indent=2)

        # Write atomically with 0o600 permissions
        temp_path = self.storage_path.with_suffix(".tmp")
        fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        temp_path.replace(self.storage_path)
        try:
            os.chmod(self.storage_path, 0o600)
        except OSError:
            pass

    def create_token(
        self, client_name: str = "", ttl_seconds: int = DEFAULT_TOKEN_TTL_SECONDS
    ) -> EnrollmentToken:
        """Generate a cryptographically random, short-lived enrollment token."""
        now = time.time()
        # High-entropy random token with prefix
        raw_secret = secrets.token_urlsafe(24)
        token_str = f"gpf_enroll_{raw_secret}"

        token = EnrollmentToken(
            token=token_str,
            client_name=client_name.strip(),
            created_at=now,
            expires_at=now + max(1, ttl_seconds),
            used=False,
        )

        with self._lock:
            self._load()
            self._tokens[token.token] = token
            self._save()

        return token

    def validate_and_consume(self, token_str: str, client_name: str = "") -> bool:
        """Validate the token, ensuring it is unexpired, unused, and matches client constraints.

        Immediately marks the token as used if valid (single-use).
        """
        with self._lock:
            self._load()
            token = self._tokens.get(token_str)
            if not token:
                return False
            if not token.is_valid:
                return False
            if token.client_name and client_name and token.client_name != client_name:
                return False

            # Mark as used (single-use)
            token.used = True
            self._save()
            return True

    def list_tokens(self) -> List[EnrollmentToken]:
        with self._lock:
            self._load()
            return list(self._tokens.values())

    def revoke_token(self, token_str: str) -> bool:
        with self._lock:
            self._load()
            if token_str in self._tokens:
                del self._tokens[token_str]
                self._save()
                return True
            return False

    def clean_expired(self) -> int:
        with self._lock:
            self._load()
            now = time.time()
            to_remove = [k for k, v in self._tokens.items() if v.used or v.expires_at < now]
            for k in to_remove:
                del self._tokens[k]
            if to_remove:
                self._save()
            return len(to_remove)
