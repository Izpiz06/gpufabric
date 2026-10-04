# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Tests for GPUManager."""

from unittest.mock import MagicMock, patch

import pytest

from worker.gpu import GPUManager


def test_gpu_manager_fallback():
    with patch("worker.gpu.HAS_PYNVML", False):
        mgr = GPUManager()
        assert mgr.is_available() is False
        with pytest.raises(RuntimeError):
            mgr.get_info(0)


def test_gpu_manager_mocked_nvml():
    with patch("worker.gpu.HAS_PYNVML", True), patch("worker.gpu.pynvml") as mock_nvml:
        mock_nvml.nvmlDeviceGetCount.return_value = 1
        mock_handle = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_nvml.nvmlDeviceGetName.return_value = "NVIDIA RTX A6000"

        mem = MagicMock()
        mem.total = 48000000000
        mem.free = 40000000000
        mem.used = 8000000000
        mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mem
        mock_nvml.nvmlDeviceGetCudaComputeCapability.return_value = (8, 6)
        mock_nvml.nvmlSystemGetDriverVersion.return_value = "550.0"

        rates = MagicMock()
        rates.gpu = 20
        rates.memory = 10
        mock_nvml.nvmlDeviceGetUtilizationRates.return_value = rates
        mock_nvml.nvmlDeviceGetTemperature.return_value = 52

        mgr = GPUManager()
        assert mgr.is_available() is True
        info = mgr.get_info(0)
        assert info["name"] == "NVIDIA RTX A6000"
        assert info["compute_capability"] == "8.6"

        status = mgr.get_status(0)
        assert status["gpu_util"] == 20
        assert status["temp"] == 52


def test_list_devices_merges_info_and_status():
    with patch("worker.gpu.HAS_PYNVML", True), patch("worker.gpu.pynvml") as mock_nvml:
        mock_nvml.nvmlDeviceGetCount.return_value = 2
        mock_nvml.nvmlDeviceGetName.return_value = "NVIDIA RTX 3090"
        mem = MagicMock(total=24, free=20, used=4)
        mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mem
        mock_nvml.nvmlDeviceGetCudaComputeCapability.return_value = (8, 6)
        mock_nvml.nvmlSystemGetDriverVersion.return_value = "550.0"
        mock_nvml.nvmlDeviceGetUtilizationRates.return_value = MagicMock(gpu=7, memory=1)
        mock_nvml.nvmlDeviceGetTemperature.return_value = 40

        devices = GPUManager().list_devices()

    assert [d["device_index"] for d in devices] == [0, 1]
    assert devices[1]["name"] == "NVIDIA RTX 3090"
    assert devices[1]["gpu_util"] == 7
    assert devices[1]["total_vram"] == 24


def test_list_devices_without_nvml():
    with patch("worker.gpu.HAS_PYNVML", False):
        assert GPUManager().list_devices() == []


def test_peak_memory_bandwidth():
    with patch("worker.gpu.HAS_PYNVML", True), patch("worker.gpu.pynvml") as mock_nvml:
        mock_nvml.nvmlDeviceGetCount.return_value = 1
        # RTX 3050 Laptop: 5871 MHz, 128-bit bus
        mock_nvml.nvmlDeviceGetMaxClockInfo.return_value = 5871
        mock_nvml.nvmlDeviceGetMemoryBusWidth.return_value = 128
        assert GPUManager().peak_memory_bandwidth(0) == pytest.approx(187.872)

        mock_nvml.nvmlDeviceGetMemoryBusWidth.side_effect = RuntimeError("not supported")
        assert GPUManager().peak_memory_bandwidth(0) == 0.0


def test_peak_memory_bandwidth_without_nvml():
    with patch("worker.gpu.HAS_PYNVML", False):
        assert GPUManager().peak_memory_bandwidth(0) == 0.0
