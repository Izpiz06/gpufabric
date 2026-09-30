"""Tests for GPUFabricClient library."""

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from worker.app import create_app
from common.models import GPUInfoResponse
from client.client import GPUFabricClient


@pytest.fixture
def mock_app():
    app = create_app(worker_id="test-worker-02")
    return app


def test_client_discover_and_health(mock_app):
    test_client = TestClient(mock_app)
    client = GPUFabricClient(base_url="http://testworker:8000")
    client._http_client = test_client

    health = client.discover()
    assert health.status == "ok"
    assert health.worker_id == "test-worker-02"


def test_client_gpu_info(mock_app):
    mock_info = GPUInfoResponse(
        device_index=0,
        name="NVIDIA A100-SXM4-80GB",
        total_vram_bytes=80 * 1024 * 1024 * 1024,
        free_vram_bytes=70 * 1024 * 1024 * 1024,
        used_vram_bytes=10 * 1024 * 1024 * 1024,
        total_vram_human="80.00 GiB",
        free_vram_human="70.00 GiB",
        used_vram_human="10.00 GiB",
        compute_capability="8.0",
        driver_version="535.104.05",
    )
    with patch.object(mock_app.state.worker.gpu_manager, "is_available", return_value=True), \
         patch.object(mock_app.state.worker.gpu_manager, "get_gpu_info", return_value=mock_info):
        test_client = TestClient(mock_app)
        client = GPUFabricClient(base_url="http://testworker:8000")
        client._http_client = test_client

        info = client.get_gpu_info(device_index=0)
        assert info.name == "NVIDIA A100-SXM4-80GB"
        assert info.compute_capability == "8.0"


def test_client_execute_vector_add(mock_app):
    with patch.object(
        mock_app.state.worker.gpu_executor,
        "execute_vector_add",
        return_value=([10.0, 20.0, 30.0], 0.15, "torch_cuda"),
    ):
        test_client = TestClient(mock_app)
        client = GPUFabricClient(base_url="http://testworker:8000")
        client._http_client = test_client

        resp = client.execute_vector_add(a=[1.0, 2.0, 3.0], b=[9.0, 18.0, 27.0])
        assert resp.status == "success"
        assert resp.result == [10.0, 20.0, 30.0]
        assert resp.gpu_backend == "torch_cuda"
