# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Cluster policy enforcement, Slurm node draining, and trust monitoring."""

import logging
import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

from common.alerts import Alert, AlertSink, get_default_alert_sink
from common.trust import TrustStore, get_default_trust_store

logger = logging.getLogger("gpufabric.policy")

NODE_NAME_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


class PolicyActions(ABC):
    """Abstract interface for cluster and gateway policy enforcement."""

    @abstractmethod
    def drain_node(self, node: str, reason: str) -> bool:
        """Mark a faulty or suspect worker node as drained in cluster scheduler."""

    @abstractmethod
    def undrain_node(self, node: str) -> bool:
        """Restore a recovered worker node back to active cluster scheduling."""

    @abstractmethod
    def restrict_user(self, user: str, reason: str) -> bool:
        """Add an adversarial user/certificate identity to gateway deny list."""

    @abstractmethod
    def unrestrict_user(self, user: str) -> bool:
        """Remove a user from the gateway deny list."""


class LogOnlyActions(PolicyActions):
    """Safe default policy executor that logs actions without external side effects."""

    def __init__(self):
        self.actions_log: List[dict] = []

    def drain_node(self, node: str, reason: str) -> bool:
        logger.warning(f"[POLICY LOG-ONLY] DRAIN node={node} reason={reason!r}")
        self.actions_log.append({"action": "drain_node", "node": node, "reason": reason})
        return True

    def undrain_node(self, node: str) -> bool:
        logger.info(f"[POLICY LOG-ONLY] UNDRAIN node={node}")
        self.actions_log.append({"action": "undrain_node", "node": node})
        return True

    def restrict_user(self, user: str, reason: str) -> bool:
        logger.warning(f"[POLICY LOG-ONLY] RESTRICT user={user} reason={reason!r}")
        self.actions_log.append({"action": "restrict_user", "user": user, "reason": reason})
        return True

    def unrestrict_user(self, user: str) -> bool:
        logger.info(f"[POLICY LOG-ONLY] UNRESTRICT user={user}")
        self.actions_log.append({"action": "unrestrict_user", "user": user})
        return True


class SlurmActions(PolicyActions):
    """Production policy executor interfacing directly with Slurm scontrol.

    Safety Guarantees:
      - Uses structured argument lists (subprocess list argv, NEVER shell=True).
      - Strictly validates node names against regex to prevent argument injection.
      - Dry-run by default; real execution requires explicit execute=True AND scontrol on PATH.
    """

    def __init__(
        self,
        execute: bool = False,
        deny_list_path: Optional[Path] = None,
    ):
        self.execute = execute
        self.deny_list_path = deny_list_path or (
            Path.home() / ".local" / "state" / "gpufabric" / "restricted_users.txt"
        )
        self.last_command: Optional[List[str]] = None

    def _clean_node_name(self, node: str) -> str:
        clean = node.removeprefix("node:").strip()
        if not NODE_NAME_REGEX.match(clean):
            raise ValueError(f"Invalid node name for cluster action: {node!r}")
        return clean

    def drain_node(self, node: str, reason: str) -> bool:
        clean_node = self._clean_node_name(node)
        safe_reason = reason.replace("\n", " ").strip()[:100]
        argv = [
            "scontrol",
            "update",
            f"NodeName={clean_node}",
            "State=DRAIN",
            f"Reason={safe_reason}",
        ]
        self.last_command = argv

        if not self.execute:
            logger.info(f"[SLURM DRY-RUN] Would execute: {' '.join(argv)}")
            return True

        if not shutil.which("scontrol"):
            logger.warning("scontrol executable not found on PATH; cannot execute Slurm drain")
            return False

        try:
            res = subprocess.run(argv, check=True, capture_output=True, text=True)
            logger.warning(f"Slurm node {clean_node} drained: {res.stdout.strip()}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to drain Slurm node {clean_node}: {e.stderr.strip()}")
            return False

    def undrain_node(self, node: str) -> bool:
        clean_node = self._clean_node_name(node)
        argv = ["scontrol", "update", f"NodeName={clean_node}", "State=RESUME"]
        self.last_command = argv

        if not self.execute:
            logger.info(f"[SLURM DRY-RUN] Would execute: {' '.join(argv)}")
            return True

        if not shutil.which("scontrol"):
            logger.warning("scontrol executable not found on PATH; cannot execute Slurm resume")
            return False

        try:
            res = subprocess.run(argv, check=True, capture_output=True, text=True)
            logger.info(f"Slurm node {clean_node} resumed: {res.stdout.strip()}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to resume Slurm node {clean_node}: {e.stderr.strip()}")
            return False

    def restrict_user(self, user: str, reason: str) -> bool:
        clean_user = user.removeprefix("user:").strip()
        self.deny_list_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing = set()
            if self.deny_list_path.exists():
                with open(self.deny_list_path, "r", encoding="utf-8") as f:
                    existing = {line.strip() for line in f if line.strip()}
            if clean_user not in existing:
                with open(self.deny_list_path, "a", encoding="utf-8") as f:
                    f.write(f"{clean_user}\n")
            logger.warning(f"Restricted user identity {clean_user!r} (reason: {reason})")
            return True
        except Exception as e:
            logger.error(f"Failed updating user deny list {self.deny_list_path}: {e}")
            return False

    def unrestrict_user(self, user: str) -> bool:
        clean_user = user.removeprefix("user:").strip()
        if not self.deny_list_path.exists():
            return True
        try:
            with open(self.deny_list_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip() and line.strip() != clean_user]
            with open(self.deny_list_path, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(f"{line}\n")
            return True
        except Exception as e:
            logger.error(f"Failed removing user from deny list: {e}")
            return False


class TrustMonitor:
    """Subscribes to security AlertSink and verification events to enforce cluster policies."""

    def __init__(
        self,
        trust_store: Optional[TrustStore] = None,
        policy_actions: Optional[PolicyActions] = None,
        alert_sink: Optional[AlertSink] = None,
    ):
        self.trust_store = trust_store or get_default_trust_store()
        self.policy = policy_actions or LogOnlyActions()
        self.alert_sink = alert_sink or get_default_alert_sink()
        self._drained_nodes: set = set()
        self._restricted_users: set = set()
        self.alert_sink.subscribe(self.handle_alert)

    def handle_alert(self, alert: Alert) -> None:
        """Route incoming security alerts into the TrustStore and evaluate policy triggers."""
        if alert.kind in ("decoy_contact", "canary_hit"):
            # Deception events penalize the USER identity, not the worker node
            user_entity = (
                f"user:{alert.cert_cn}"
                if alert.cert_cn
                else (alert.subject if alert.subject.startswith("user:") else "user:unknown")
            )
            score, state = self.trust_store.record_event(
                entity=user_entity,
                event=alert.kind.upper(),
                detail=alert.detail,
            )
            if state == "RESTRICTED" and user_entity not in self._restricted_users:
                self._restricted_users.add(user_entity)
                self.policy.restrict_user(user_entity, alert.detail)
            elif state == "HEALTHY" and user_entity in self._restricted_users:
                self._restricted_users.remove(user_entity)
                self.policy.unrestrict_user(user_entity)

        elif alert.kind == "probe_fail":
            node_entity = (
                alert.subject if alert.subject.startswith("node:") else f"node:{alert.subject}"
            )
            score, state = self.trust_store.record_event(
                entity=node_entity,
                event="PROBE_FAIL",
                detail=alert.detail,
            )
            if state == "DRAIN" and node_entity not in self._drained_nodes:
                self._drained_nodes.add(node_entity)
                self.policy.drain_node(
                    node_entity,
                    f"Trust score dropped to {score:.2f} following probe failure",
                )
            elif state == "HEALTHY" and node_entity in self._drained_nodes:
                self._drained_nodes.remove(node_entity)
                self.policy.undrain_node(node_entity)

        elif alert.kind == "verify_fail":
            node_entity = (
                alert.subject if alert.subject.startswith("node:") else f"node:{alert.subject}"
            )
            score, state = self.trust_store.record_event(
                entity=node_entity,
                event="VERIFY_FAIL",
                detail=alert.detail,
            )
            if state == "DRAIN" and node_entity not in self._drained_nodes:
                self._drained_nodes.add(node_entity)
                self.policy.drain_node(
                    node_entity,
                    f"Trust score dropped to {score:.2f} following verification failure",
                )
            elif state == "HEALTHY" and node_entity in self._drained_nodes:
                self._drained_nodes.remove(node_entity)
                self.policy.undrain_node(node_entity)

    def record_verification(self, node: str, passed: bool, detail: str = "") -> Tuple[float, str]:
        """Record verification outcome and trigger cluster drainage if node crosses threshold."""
        node_entity = node if node.startswith("node:") else f"node:{node}"
        event = "VERIFY_PASS" if passed else "VERIFY_FAIL"
        score, state = self.trust_store.record_event(entity=node_entity, event=event, detail=detail)
        if state == "DRAIN" and node_entity not in self._drained_nodes:
            self._drained_nodes.add(node_entity)
            self.policy.drain_node(
                node_entity,
                f"Trust score dropped to {score:.2f} (repeated verification failures)",
            )
        elif state == "HEALTHY" and node_entity in self._drained_nodes:
            self._drained_nodes.remove(node_entity)
            self.policy.undrain_node(node_entity)
        return score, state
