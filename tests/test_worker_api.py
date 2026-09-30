"""Tests for FastAPI worker API routes."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from common.models import GPUInfoResponse, GPUStatusResponse
from worker.app import create_app


@pytest.fixture
def client():
    app = create_app(worker_id="test-worker-01")
    return TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["worker_id"] == "test-worker-01"
    assert "version" in data
    assert "gpu_available" in data


def test_gpu_info_endpoint_mocked():
    app = create_app(worker_id="test-worker-01")
    mock_info = GPUInfoResponse(
        device_index=0,
        name="NVIDIA GeForce RTX 3080",
        total_vram_bytes=10 * 1024 * 1024 * 1024,
        free_vram_bytes=8 * 1024 * 1024 * 1024,
        used_vram_bytes=2 * 1024 * 1024 * 1024,
        total_vram_human="10.00 GiB",
        free_vram_human="8.00 GiB",
        used_vram_human="2.00 GiB",
        compute_capability="8.6",
        driver_version="535.104.05",
    )
    with (
        patch.object(app.state.worker.gpu_manager, "is_available", return_value=True),
        patch.object(app.state.worker.gpu_manager, "get_gpu_info", return_value=mock_info),
    ):
        test_client = TestClient(app)
        resp = test_client.get("/gpu?device_index=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "NVIDIA GeForce RTX 3080"
        assert data["compute_capability"] == "8.6"


def test_gpu_status_endpoint_mocked():
    app = create_app(worker_id="test-worker-01")
    mock_status = GPUStatusResponse(
        device_index=0,
        gpu_utilization_pct=25,
        memory_utilization_pct=10,
        total_vram_bytes=10 * 1024 * 1024 * 1024,
        free_vram_bytes=8 * 1024 * 1024 * 1024,
        used_vram_bytes=2 * 1024 * 1024 * 1024,
        free_vram_human="8.00 GiB",
        used_vram_human="2.00 GiB",
        temperature_c=55,
        active_tasks=0,
    )
    with (
        patch.object(app.state.worker.gpu_manager, "is_available", return_value=True),
        patch.object(app.state.worker.gpu_manager, "get_gpu_status", return_value=mock_status),
    ):
        test_client = TestClient(app)
        resp = test_client.get("/status?device_index=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["gpu_utilization_pct"] == 25
        assert data["temperature_c"] == 55


def test_execute_endpoint_mocked():
    app = create_app(worker_id="test-worker-01")
    with patch.object(
        app.state.worker.gpu_executor,
        "execute_vector_add",
        return_value=([5.0, 7.0, 9.0], 0.32, "cupy"),
    ):
        test_client = TestClient(app)
        payload = {
            "workload_type": "vector_add",
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
            "device_index": 0,
        }
        resp = test_client.post("/execute", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["result"] == [5.0, 7.0, 9.0]
        assert data["gpu_backend"] == "cupy"
        assert data["execution_time_ms"] == 0.32


def test_execute_dimension_mismatch(client):
    payload = {
        "workload_type": "vector_add",
        "a": [1.0, 2.0],
        "b": [4.0],
        "device_index": 0,
    }
    resp = client.post("/execute", json=payload)
    assert resp.status_code == 400
    assert "mismatch" in resp.json()["detail"]
