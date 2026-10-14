# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for standalone GPU Fabric Host TUI."""

import asyncio
from unittest.mock import MagicMock, patch

from textual.widgets import Label, Select, Static

from client.cli import build_parser
from worker.diagnostics import CudaDiagnostics
from worker.server import parse_args
from worker.tui.app import (
    GPUFabricHostApp,
    HostHeader,
    get_local_host_info,
)


def test_get_local_host_info():
    """Verify local host info collection without network connectivity."""
    info = get_local_host_info()
    assert "hostname" in info
    assert "platform" in info
    assert "python" in info
    assert "ips" in info
    assert "tls_dir" in info
    assert "tls_desc" in info
    assert len(info["hostname"]) > 0


def test_get_local_host_info_exception_handling():
    """Verify error fallbacks during host info discovery."""
    with patch(
        "worker.tui.app.detect_host_addresses", side_effect=RuntimeError("No network interfaces")
    ):
        with patch("worker.tui.app.default_tls_dir", side_effect=RuntimeError("TLS error")):
            info = get_local_host_info()
            assert info["ips"] == "None detected"
            assert info["tls_desc"] == "Unknown"


def test_host_tui_startup():
    """Test Host TUI launches and mounts all essential panels."""

    async def _test():
        app = GPUFabricHostApp(worker_id="test-worker")
        async with app.run_test() as pilot:
            await pilot.pause()
            header = app.query_one(HostHeader)
            assert header is not None
            assert header.host_name != ""
            assert app.query_one("#host-info-panel") is not None
            assert app.query_one("#gpu-telemetry-panel") is not None
            assert app.query_one("#lbl-status") is not None

    asyncio.run(_test())


def test_host_tui_no_gpus_fallback():
    """Test Host TUI behavior when no NVIDIA GPUs are present or NVML is uninitialized."""

    mock_mgr = MagicMock()
    mock_mgr.device_count.return_value = 0
    mock_mgr.is_available.return_value = False
    mock_mgr.get_status.return_value = None
    mock_mgr.get_info.return_value = {}
    mock_mgr.get_driver_version.return_value = "N/A"
    mock_mgr.get_cuda_version.return_value = "N/A"

    async def _test():
        with patch("worker.tui.app.GPUManager", return_value=mock_mgr):
            app = GPUFabricHostApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                header = app.query_one(HostHeader)
                assert header.gpu_count == 0

                no_gpu_msg = app.query_one("#no-gpu-msg")
                assert no_gpu_msg.display is True

                telemetry_content = app.query_one("#telemetry-content")
                assert telemetry_content.display is False

    asyncio.run(_test())


def test_host_tui_multi_gpu_and_cycling():
    """Test multi-GPU host detection, rendering, and device switching."""

    mock_mgr = MagicMock()
    mock_mgr.device_count.return_value = 2
    mock_mgr.is_available.return_value = True
    mock_mgr.get_driver_version.return_value = "550.54.14"
    mock_mgr.get_cuda_version.return_value = "12.4"
    mock_mgr.peak_memory_bandwidth.return_value = 1008.0

    def mock_get_info(idx):
        if idx == 0:
            return {
                "name": "NVIDIA RTX 4090",
                "compute_capability": "8.9",
                "memory_bus_width": 384,
                "memory_clock_rate": 10501000,
            }
        return {
            "name": "NVIDIA RTX 3090",
            "compute_capability": "8.6",
            "memory_bus_width": 384,
            "memory_clock_rate": 9751000,
        }

    def mock_get_status(idx):
        if idx == 0:
            return {
                "total_vram": 24_000_000_000,
                "free_vram": 16_000_000_000,
                "used_vram": 8_000_000_000,
                "gpu_util": 45,
                "mem_util": 30,
                "temp": 50,
                "power_usage_w": 220,
                "power_limit_w": 450,
            }
        return {
            "total_vram": 24_000_000_000,
            "free_vram": 10_000_000_000,
            "used_vram": 14_000_000_000,
            "gpu_util": 85,
            "mem_util": 65,
            "temp": 75,
            "power_usage_w": 340,
            "power_limit_w": 350,
        }

    mock_mgr.get_info.side_effect = mock_get_info
    mock_mgr.get_status.side_effect = mock_get_status

    async def _test():
        with patch("worker.tui.app.GPUManager", return_value=mock_mgr):
            app = GPUFabricHostApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                header = app.query_one(HostHeader)
                assert header.gpu_count == 2
                assert app.selected_device_index == 0

                # Check Device 0 values
                gpu_name_label = app.query_one("#host-gpu-name", Label)
                assert "RTX 4090" in str(gpu_name_label.render())

                # Cycle device to 1
                app.action_cycle_device()
                await pilot.pause()
                assert app.selected_device_index == 1

                # Check Device 1 values
                gpu_name_label = app.query_one("#host-gpu-name", Label)
                assert "RTX 3090" in str(gpu_name_label.render())

                # Select widget changed
                selector = app.query_one("#host-device-select", Select)
                assert selector.value == 1

                # Test manual refresh
                app.action_refresh()
                await pilot.pause()

    asyncio.run(_test())


def test_host_tui_degraded_diagnostics():
    """Test Host TUI rendering when CUDA diagnostics report DEGRADED/missing cuBLAS."""

    mock_diag = CudaDiagnostics(
        gpu_detected=True,
        cuda_driver_version="12.4",
        driver_version="550.54.14",
        cudart_available=True,
        cudart_version="12.4",
        cupy_available=True,
        cublas_available=False,
        missing_libraries=["libcublas.so.12"],
        warnings=["cuBLAS runtime shared library missing."],
        remediation_hints=[
            "Install CUDA Toolkit or libcublas: sudo apt-get install -y libcublas-12-4"
        ],
    )

    async def _test():
        with patch("worker.tui.app.probe_cuda_environment", return_value=mock_diag):
            app = GPUFabricHostApp()
            async with app.run_test() as pilot:
                await pilot.pause()

                # Status should be DEGRADED
                header = app.query_one(HostHeader)
                assert header.host_status == "DEGRADED"

                cublas_label = app.query_one("#lbl-cublas", Label)
                assert "MISSING" in str(cublas_label.render())

                hints_text = app.query_one("#alerts-text", Static)
                assert "libcublas-12-4" in str(hints_text.render())

    asyncio.run(_test())


def test_cli_host_subcommand_parser():
    """Test that client CLI parses host subcommand with options."""
    parser = build_parser()
    args = parser.parse_args(["host", "--worker-id", "custom-node", "--refresh", "3.5"])
    assert args.command == "host"
    assert args.worker_id == "custom-node"
    assert args.refresh == 3.5


def test_worker_tui_flag_parser():
    """Test that worker CLI parses --tui option."""
    with patch("sys.argv", ["gpufabric-worker", "--tui", "--worker-id", "w-tui"]):
        args = parse_args()
        assert args.tui is True
        assert args.worker_id == "w-tui"
