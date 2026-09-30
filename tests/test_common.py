"""Tests for common models and protocol utilities."""

from common.models import (
    ExecuteRequest,
    ExecuteResponse,
    GPUInfoResponse,
    HealthResponse,
    WorkloadType,
)
from common.protocol import bytes_to_human


def test_bytes_to_human():
    assert bytes_to_human(0) == "0.00 B"
    assert bytes_to_human(1024) == "1.00 KiB"
    assert bytes_to_human(1024 * 1024) == "1.00 MiB"
    assert bytes_to_human(8 * 1024 * 1024 * 1024) == "8.00 GiB"
    assert bytes_to_human(-10) == "0 B"


def test_health_response_model():
    health = HealthResponse(
        status="ok",
        worker_id="test-worker",
        version="0.1.0",
        gpu_available=True,
    )
    assert health.status == "ok"
    assert health.gpu_available is True
    data = health.model_dump()
    assert data["worker_id"] == "test-worker"


def test_gpu_info_model():
    info = GPUInfoResponse(
        device_index=0,
        name="NVIDIA GeForce RTX 4090",
        total_vram_bytes=24576 * 1024 * 1024,
        free_vram_bytes=20000 * 1024 * 1024,
        used_vram_bytes=4576 * 1024 * 1024,
        total_vram_human="24.00 GiB",
        free_vram_human="19.53 GiB",
        used_vram_human="4.47 GiB",
        compute_capability="8.9",
        driver_version="550.54.14",
    )
    assert info.name == "NVIDIA GeForce RTX 4090"
    assert info.compute_capability == "8.9"


def test_execute_request_and_response():
    req = ExecuteRequest(
        workload_type=WorkloadType.VECTOR_ADD,
        a=[1.0, 2.0, 3.0],
        b=[4.0, 5.0, 6.0],
        device_index=0,
    )
    assert len(req.a) == 3
    assert req.workload_type == WorkloadType.VECTOR_ADD

    resp = ExecuteResponse(
        task_id="task-123",
        workload_type="vector_add",
        status="success",
        result=[5.0, 7.0, 9.0],
        result_length=3,
        execution_time_ms=0.45,
        device_index=0,
        gpu_backend="cupy",
    )
    assert resp.status == "success"
    assert resp.result == [5.0, 7.0, 9.0]
