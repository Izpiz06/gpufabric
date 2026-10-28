# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for deception layer: decoys, canaries, and known-answer probes."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import grpc
import numpy as np
import pytest

from client.probe import probe_worker
from common.alerts import Alert, AlertSink
from common.canary import CanaryRegistry
from common.gpufabric_pb2 import (
    ComputeRequest,
    ExecuteRequest,
    GPUInfoRequest,
    GPUStatusRequest,
    HealthRequest,
    ListGPUsRequest,
    Operation,
    WorkloadType,
)
from worker.decoy import DecoyServicer
from worker.server import parse_args


def test_alerts_sink(tmp_path: Path):
    sink_path = tmp_path / "alerts.jsonl"
    sink = AlertSink(log_path=sink_path)

    dispatched = []
    sink.subscribe(lambda a: dispatched.append(a))

    alert = Alert.create(
        kind="decoy_contact",
        subject="node:decoy-01",
        peer="127.0.0.1:50000",
        method="GetHealth",
        cert_cn="adversary-cert",
        detail="Test alert",
    )
    sink.emit(alert)

    assert len(dispatched) == 1
    assert dispatched[0].kind == "decoy_contact"

    # Verify JSONL persistence
    read_items = sink.read_alerts()
    assert len(read_items) == 1
    assert read_items[0].subject == "node:decoy-01"
    assert read_items[0].cert_cn == "adversary-cert"


def test_decoy_servicer_reads_and_alerts(tmp_path: Path):
    sink = AlertSink(log_path=tmp_path / "decoy_alerts.jsonl")
    decoy = DecoyServicer(
        worker_id="decoy-test-worker", profile_name="RTX 4060 8GB", alert_sink=sink
    )

    mock_context = MagicMock()
    mock_context.peer.return_value = "ipv4:10.0.0.99:43210"
    mock_context.auth_context.return_value = {"x509_common_name": [b"test-client-cert"]}

    # 1. GetHealth
    h_resp = decoy.GetHealth(HealthRequest(), mock_context)
    assert h_resp.status == "ok"
    assert h_resp.gpu_available is True
    assert h_resp.gpu_count == 1
    assert "RTX 4060" in decoy.profile["name"]

    # 2. GetGPUInfo
    info_resp = decoy.GetGPUInfo(GPUInfoRequest(device_index=0), mock_context)
    assert "RTX 4060" in info_resp.name
    assert info_resp.compute_capability == "8.9"
    assert info_resp.total_vram_bytes == 8_589_934_592

    # 3. GetGPUStatus
    stat_resp = decoy.GetGPUStatus(GPUStatusRequest(device_index=0), mock_context)
    assert stat_resp.temperature_c > 0
    assert stat_resp.gpu_utilization_pct >= 0
    assert stat_resp.available is True

    # 4. ListGPUs
    list_resp = decoy.ListGPUs(ListGPUsRequest(), mock_context)
    assert len(list_resp.gpus) == 1
    assert list_resp.compute_ready is True

    # Verify that each query recorded an alert
    alerts = sink.read_alerts()
    assert len(alerts) >= 4
    methods = [a.method for a in alerts]
    assert "GetHealth" in methods
    assert "GetGPUInfo" in methods
    assert "GetGPUStatus" in methods
    assert "ListGPUs" in methods
    assert alerts[0].cert_cn == "test-client-cert"


def test_decoy_servicer_refuses_computations(tmp_path: Path):
    sink = AlertSink(log_path=tmp_path / "decoy_comp_alerts.jsonl")
    decoy = DecoyServicer(alert_sink=sink)

    mock_context = MagicMock()
    mock_context.peer.return_value = "ipv4:10.0.0.88:12345"
    mock_context.auth_context.return_value = {}

    # Compute
    decoy.Compute(ComputeRequest(op=Operation.OP_MATMUL, device_index=0), mock_context)
    mock_context.abort.assert_called_with(
        grpc.StatusCode.UNAVAILABLE,
        "GPU device is currently busy with high-priority workload.",
    )

    # Execute
    mock_context.reset_mock()
    decoy.Execute(
        ExecuteRequest(workload_type=WorkloadType.VECTOR_ADD, device_index=0),
        mock_context,
    )
    mock_context.abort.assert_called_with(
        grpc.StatusCode.UNAVAILABLE,
        "GPU device is currently busy with high-priority workload.",
    )


def test_canary_registry(tmp_path: Path):
    db_path = tmp_path / "test_canary.db"
    sink = AlertSink(log_path=tmp_path / "canary_alerts.jsonl")
    reg = CanaryRegistry(db_path=db_path, alert_sink=sink)

    # Mint canary
    token = reg.mint(kind="job_id")
    assert token.startswith("cnry_")

    # Regular ID should not trigger hit
    assert reg.check("normal_job_12345") is False
    assert len(sink.read_alerts()) == 0

    # Canary hit
    mock_context = MagicMock()
    mock_context.peer.return_value = "ipv4:172.16.0.5:60000"
    mock_context.auth_context.return_value = {"x509_common_name": [b"bad-actor"]}

    hit = reg.check(token, context=mock_context)
    assert hit is True

    alerts = sink.read_alerts()
    assert len(alerts) == 1
    assert alerts[0].kind == "canary_hit"
    assert alerts[0].cert_cn == "bad-actor"
    assert "cnry_" in alerts[0].detail


def test_probe_worker():
    mock_client = MagicMock()
    mock_client.target = "10.0.0.15:50051"

    # 1. Honest worker passes probe
    def mock_honest_compute(op, a, b, device_index=0):
        res = MagicMock()
        res.result = np.matmul(a, b)
        res.gpu_time_ms = 1.2
        return res

    mock_client.compute.side_effect = mock_honest_compute
    report_pass = probe_worker(mock_client, size=8)
    assert report_pass.passed is True
    assert report_pass.rel_error == pytest.approx(0.0, abs=1e-5)

    # 2. Corrupted/faulty worker fails probe and emits alert
    def mock_faulty_compute(op, a, b, device_index=0):
        res = MagicMock()
        res.result = np.matmul(a, b) + 50.0  # corrupted
        res.gpu_time_ms = 1.2
        return res

    mock_client.compute.side_effect = mock_faulty_compute
    with patch("client.probe.emit_alert") as mock_emit:
        report_fail = probe_worker(mock_client, size=8)
        assert report_fail.passed is False
        assert report_fail.rel_error > 0.1
        assert mock_emit.called
        assert mock_emit.call_args[1]["kind"] == "probe_fail"


def test_server_cli_decoy_args():
    with patch(
        "sys.argv",
        ["gpufabric-worker", "--decoy", "--decoy-profile", "RTX 3060 12GB"],
    ):
        args = parse_args()
        assert args.decoy is True
        assert args.decoy_profile == "RTX 3060 12GB"
