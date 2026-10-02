"""Tests for GPUFabricClient."""

from concurrent import futures
from unittest.mock import patch

import grpc
import numpy as np
import pytest

from client.client import GPUFabricClient, GPUFabricError
from client.commands import collect_inventory, resolve_workers
from common.constants import DEFAULT_MAX_MESSAGE_MB, DEFAULT_PORT
from common.gpufabric_pb2_grpc import add_GPUFabricServiceServicer_to_server
from common.grpc_options import message_size_options
from worker.executor import GPUExecutor
from worker.gpu import GPUManager
from worker.service import GPUFabricServicer
from worker.state import WorkerState


@pytest.fixture(scope="module")
def grpc_server():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=2),
        options=message_size_options(DEFAULT_MAX_MESSAGE_MB),
    )
    state = WorkerState("test-worker-e2e")
    gpu = GPUManager()
    executor = GPUExecutor()
    servicer = GPUFabricServicer(state, gpu, executor)
    add_GPUFabricServiceServicer_to_server(servicer, server)

    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    yield port, servicer, gpu, executor
    server.stop(None)


def test_client_health_and_discover(grpc_server):
    port, _, _, _ = grpc_server
    with GPUFabricClient(host="127.0.0.1", port=port, insecure=True) as client:
        health = client.health()
        assert health.status == "ok"
        assert health.worker_id == "test-worker-e2e"

        disc = client.discover()
        assert disc.worker_id == "test-worker-e2e"


def test_client_gpu_info(grpc_server):
    port, _, gpu, _ = grpc_server
    mock_info = {
        "device_index": 0,
        "name": "NVIDIA A100",
        "total_vram": 80000000000,
        "free_vram": 70000000000,
        "used_vram": 10000000000,
        "compute_capability": "8.0",
        "driver_version": "535.0",
    }
    with (
        patch.object(gpu, "is_available", return_value=True),
        patch.object(gpu, "get_info", return_value=mock_info),
    ):
        with GPUFabricClient(host="127.0.0.1", port=port, insecure=True) as client:
            info = client.get_gpu_info(device_index=0)
            assert info.name == "NVIDIA A100"
            assert info.compute_capability == "8.0"


def test_client_execute_vector_add(grpc_server):
    port, _, _, executor = grpc_server
    with patch.object(executor, "execute_vector_add", return_value=([10.0, 20.0], 0.18, "cupy")):
        with GPUFabricClient(host="127.0.0.1", port=port, insecure=True) as client:
            resp = client.execute_vector_add(a=[1.0, 2.0], b=[9.0, 18.0])
            assert resp.status == "success"
            assert list(resp.result) == [10.0, 20.0]
            assert resp.gpu_backend == "cupy"


def test_client_execute_payload_above_grpc_default_limit(grpc_server):
    # 600k float32 per vector is ~4.8 MB per request, above gRPC's 4 MiB default.
    port, _, _, executor = grpc_server
    n = 600_000
    with patch.object(executor, "execute_vector_add", return_value=([2.0] * n, 1.0, "cupy")):
        with GPUFabricClient(host="127.0.0.1", port=port, insecure=True) as client:
            resp = client.execute_vector_add(a=[1.0] * n, b=[1.0] * n)
            assert resp.result_length == n
            assert len(resp.result) == n


def test_client_compute_roundtrip(grpc_server):
    port, _, _, executor = grpc_server
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.ones((3, 2), dtype=np.float32)
    with patch.object(executor, "compute", return_value=(a @ b, 0.01, 0.2)) as compute:
        with GPUFabricClient(host="127.0.0.1", port=port, insecure=True) as client:
            res = client.matmul(a, b)
    op, inputs, device = compute.call_args.args
    assert op == "matmul" and device == 0
    np.testing.assert_array_equal(inputs[0], a)
    np.testing.assert_array_equal(res.result, a @ b)
    assert res.gpu_time_ms == 0.01
    assert res.round_trip_ms > 0


def test_client_compute_error_is_wrapped(grpc_server):
    port, _, _, _ = grpc_server
    with GPUFabricClient(host="127.0.0.1", port=port, insecure=True) as client:
        with pytest.raises(GPUFabricError, match="inner dimensions"):
            client.matmul(np.ones((2, 3), np.float32), np.ones((2, 3), np.float32))


def test_client_compute_unknown_operation():
    with GPUFabricClient(host="127.0.0.1", port=1, insecure=True) as client:
        with pytest.raises(ValueError, match="Unknown operation"):
            client.compute("transpose", np.ones(2))


def test_collect_inventory_mixes_live_and_dead_workers(grpc_server, monkeypatch):
    port, _, gpu, _ = grpc_server
    with patch.object(gpu, "list_devices", return_value=[]):
        rows = collect_inventory(
            [f"127.0.0.1:{port}", "127.0.0.1:1"], port=DEFAULT_PORT, timeout=2.0, insecure=True
        )
    assert rows[0][0] == f"127.0.0.1:{port}"
    assert rows[0][1].worker_id == "test-worker-e2e"
    assert isinstance(rows[1][1], str) and "127.0.0.1:1" in rows[1][1]


def test_resolve_workers(monkeypatch):
    monkeypatch.setenv("GPUFABRIC_WORKERS", "10.0.0.1, 10.0.0.2:6000,")
    assert resolve_workers([]) == ["10.0.0.1", "10.0.0.2:6000"]
    assert resolve_workers(["10.0.0.9"]) == ["10.0.0.9"]
    monkeypatch.delenv("GPUFABRIC_WORKERS")
    assert resolve_workers([]) == []


def test_client_error_keeps_status_code(grpc_server):
    port, _, _, _ = grpc_server
    with GPUFabricClient(host="127.0.0.1", port=port, insecure=True) as client:
        with pytest.raises(GPUFabricError) as exc:
            client.matmul(np.ones((2, 3), np.float32), np.ones((2, 3), np.float32))
    assert exc.value.code == grpc.StatusCode.INVALID_ARGUMENT
    assert isinstance(exc.value.__cause__, grpc.RpcError)
