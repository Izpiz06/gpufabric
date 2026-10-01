"""Tests for GPUFabricServicer."""

from unittest.mock import MagicMock, patch

from common.gpufabric_pb2 import (
    ExecuteRequest,
    GPUInfoRequest,
    HealthRequest,
    WorkloadType,
)
from worker.executor import GPUExecutor
from worker.gpu import GPUManager
from worker.service import GPUFabricServicer
from worker.state import WorkerState


def test_service_health():
    state = WorkerState("worker-node-1")
    gpu = GPUManager()
    executor = GPUExecutor()
    servicer = GPUFabricServicer(state, gpu, executor)

    mock_context = MagicMock()
    resp = servicer.GetHealth(HealthRequest(), mock_context)
    assert resp.status == "ok"
    assert resp.worker_id == "worker-node-1"


def test_service_gpu_info():
    state = WorkerState("worker-node-1")
    gpu = GPUManager()
    executor = GPUExecutor()
    servicer = GPUFabricServicer(state, gpu, executor)

    mock_info = {
        "device_index": 0,
        "name": "NVIDIA RTX 3080",
        "total_vram": 10000000000,
        "free_vram": 8000000000,
        "used_vram": 2000000000,
        "compute_capability": "8.6",
        "driver_version": "535.0",
    }
    with (
        patch.object(gpu, "is_available", return_value=True),
        patch.object(gpu, "get_info", return_value=mock_info),
    ):
        mock_context = MagicMock()
        resp = servicer.GetGPUInfo(GPUInfoRequest(device_index=0), mock_context)
        assert resp.name == "NVIDIA RTX 3080"
        assert resp.compute_capability == "8.6"


def test_service_execute_vector_add():
    state = WorkerState("worker-node-1")
    gpu = GPUManager()
    executor = GPUExecutor()
    servicer = GPUFabricServicer(state, gpu, executor)

    with patch.object(executor, "execute_vector_add", return_value=([5.0, 7.0], 0.25, "cupy")):
        req = ExecuteRequest(
            workload_type=WorkloadType.VECTOR_ADD,
            a=[1.0, 2.0],
            b=[4.0, 5.0],
            device_index=0,
        )
        mock_context = MagicMock()
        resp = servicer.Execute(req, mock_context)
        assert resp.status == "success"
        assert list(resp.result) == [5.0, 7.0]
        assert resp.gpu_backend == "cupy"
