# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TrustStore, Beta-reputation, and policy actions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from common.alerts import Alert, AlertSink
from common.policy import LogOnlyActions, SlurmActions, TrustMonitor
from common.trust import TrustStore


def test_beta_reputation_math(tmp_path: Path):
    db_path = tmp_path / "trust.db"
    store = TrustStore(db_path=db_path)

    # Initial prior is alpha=2.0, beta=1.0 -> score = 2/3
    score, state = store.record_event("node:worker-test", "VERIFY_PASS")
    assert score > 0.66
    assert state == "HEALTHY"

    # Subsequent passes increase score
    s2, _ = store.record_event("node:worker-test", "VERIFY_PASS")
    assert s2 > score

    # Failures decrease score
    s3, _ = store.record_event("node:worker-test", "VERIFY_FAIL")
    assert s3 < s2


def test_no_drain_on_single_failure(tmp_path: Path):
    """Critical guarantee: A single failure on consumer non-ECC GPU marks SUSPECT, never DRAIN."""
    db_path = tmp_path / "trust.db"
    store = TrustStore(db_path=db_path, min_drain_evidence=3)

    # Starts healthy
    store.record_event("node:lab-pc-01", "VERIFY_PASS")
    store.record_event("node:lab-pc-01", "VERIFY_PASS")

    # One transient verification failure
    score, state = store.record_event(
        "node:lab-pc-01", "VERIFY_FAIL", detail="Single transient bitflip"
    )

    assert state == "SUSPECT", "Single failure must transition node to SUSPECT, not DRAIN"
    assert state != "DRAIN"

    entity = store.get_entity("node:lab-pc-01")
    assert entity["fail_count"] == 1
    assert entity["state"] == "SUSPECT"


def test_drain_after_repeated_failures(tmp_path: Path):
    """Repeated failures crossing the threshold with >= min_evidence trigger DRAIN."""
    db_path = tmp_path / "trust.db"
    store = TrustStore(db_path=db_path, min_drain_evidence=3)

    store.record_event("node:faulty-worker", "VERIFY_FAIL")
    store.record_event("node:faulty-worker", "VERIFY_FAIL")
    score, state = store.record_event("node:faulty-worker", "VERIFY_FAIL")

    assert score < 0.35
    assert state == "DRAIN"

    entity = store.get_entity("node:faulty-worker")
    assert entity["state"] == "DRAIN"


def test_hysteresis_recovery(tmp_path: Path):
    """A drained node does not undrain on a single pass; requires multiple passes (hysteresis)."""
    db_path = tmp_path / "trust.db"
    store = TrustStore(db_path=db_path, min_drain_evidence=3)

    # Force into DRAIN
    for _ in range(3):
        store.record_event("node:recovered-worker", "VERIFY_FAIL", weight=4.0)

    assert store.get_entity("node:recovered-worker")["state"] == "DRAIN"

    # Single pass should NOT immediately clear DRAIN
    _, state1 = store.record_event("node:recovered-worker", "VERIFY_PASS")
    assert state1 == "DRAIN"

    # Sustained passes climb above recovery threshold and restore HEALTHY
    for _ in range(18):
        _, st = store.record_event("node:recovered-worker", "VERIFY_PASS", weight=2.0)

    entity = store.get_entity("node:recovered-worker")
    assert entity["state"] == "HEALTHY"
    assert entity["consecutive_passes"] >= 3


def test_deception_penalizes_user_not_node(tmp_path: Path):
    """Decoy contacts and canary hits must penalize the adversarial USER identity, not the node."""
    alert_sink = AlertSink(log_path=tmp_path / "alerts.jsonl")
    trust_store = TrustStore(db_path=tmp_path / "trust.db")
    policy = LogOnlyActions()
    monitor = TrustMonitor(trust_store=trust_store, policy_actions=policy, alert_sink=alert_sink)

    # Node starts with good trust
    monitor.record_verification("node:worker-node", passed=True)
    initial_node_score = trust_store.get_entity("node:worker-node")["score"]

    # Canary hit triggered with cert_cn="mallory"
    alert = Alert.create(
        kind="canary_hit",
        subject="user:mallory",
        peer="192.168.1.55:4000",
        cert_cn="mallory",
        detail="Accessed job canary",
    )
    alert_sink.emit(alert)

    # User score drops
    user_ent = trust_store.get_entity("user:mallory")
    assert user_ent is not None
    assert user_ent["score"] < 0.35
    assert user_ent["state"] == "RESTRICTED"

    # Node score must remain completely untouched
    node_ent = trust_store.get_entity("node:worker-node")
    assert node_ent["score"] == pytest.approx(initial_node_score)
    assert node_ent["state"] == "HEALTHY"


def test_slurm_actions_exact_argv_and_dry_run():
    actions = SlurmActions(execute=False)

    # Drain dry-run
    ok = actions.drain_node("node:compute-042", "Verification failed on matmul")
    assert ok is True
    assert actions.last_command == [
        "scontrol",
        "update",
        "NodeName=compute-042",
        "State=DRAIN",
        "Reason=Verification failed on matmul",
    ]

    # Resume dry-run
    ok = actions.undrain_node("node:compute-042")
    assert ok is True
    assert actions.last_command == [
        "scontrol",
        "update",
        "NodeName=compute-042",
        "State=RESUME",
    ]


def test_slurm_actions_rejects_malicious_names():
    actions = SlurmActions(execute=False)

    malicious_inputs = [
        "node; rm -rf /",
        "node$(whoami)",
        "node && echo pwned",
        "node|cat",
        "node\nnewline",
    ]
    for bad_name in malicious_inputs:
        with pytest.raises(ValueError, match="Invalid node name"):
            actions.drain_node(bad_name, "reason")


def test_slurm_actions_real_execution(tmp_path: Path):
    actions = SlurmActions(execute=True, deny_list_path=tmp_path / "deny.txt")

    # When scontrol is not on PATH, safely logs and returns False
    with patch("shutil.which", return_value=None):
        assert actions.drain_node("node-1", "reason") is False

    # When scontrol is present, executes subprocess with argument list
    with patch("shutil.which", return_value="/usr/bin/scontrol"):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="Node updated successfully\n")
            assert actions.drain_node("node-1", "bad computation") is True
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args == [
                "scontrol",
                "update",
                "NodeName=node-1",
                "State=DRAIN",
                "Reason=bad computation",
            ]

    # User restriction maintains deny file
    assert actions.restrict_user("user:badactor", "Canary trap triggered") is True
    assert "badactor" in (tmp_path / "deny.txt").read_text()
