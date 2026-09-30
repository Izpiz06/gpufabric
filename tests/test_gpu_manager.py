"""Tests for GPUManager with both real and mocked NVML."""

from unittest.mock import MagicMock, patch
import pytest
from worker.gpu import GPUManager


def test_gpu_manager_fallback_when_nvml_missing():
    with patch("worker.gpu.HAS_PYNVML", False):
        manager = GPUManager()
        assert manager.is_available() is False
        assert manager.get_device_count() == 0
        with pytest.raises(RuntimeError, match="No GPU available"):
            manager.get_gpu_info(0)


def test_gpu_manager_with_mocked_nvml():
    with patch("worker.gpu.HAS_PYNVML", True), patch("worker.gpu.pynvml") as mock_nvml:
        mock_nvml.nvmlDeviceGetCount.return_value = 1
        mock_handle = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_nvml.nvmlDeviceGetName.return_value = "NVIDIA RTX A6000"
        
        mock_mem = MagicMock()
        mock_mem.total = 48 * 1024 * 1024 * 1024
        mock_mem.free = 40 * 1024 * 1024 * 1024
        mock_mem.used = 8 * 1024 * 1024 * 1024
        mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem
        mock_nvml.nvmlDeviceGetCudaComputeCapability.return_value = (8, 6)
        mock_nvml.nvmlSystemGetDriverVersion.return_value = "535.104.05"

        manager = GPUManager()
        assert manager.is_available() is True
        assert manager.get_device_count() == 1

        info = manager.get_gpu_info(device_index=0)
        assert info.name == "NVIDIA RTX A6000"
        assert info.compute_capability == "8.6"
        assert "48.00 GiB" in info.total_vram_human

        # Test status
        mock_util = MagicMock()
        mock_util.gpu = 42
        mock_util.memory = 15
        mock_nvml.nvmlDeviceGetUtilizationRates.return_value = mock_util
        mock_nvml.nvmlDeviceGetTemperature.return_value = 65

        status = manager.get_gpu_status(device_index=0, active_tasks=2)
        assert status.gpu_utilization_pct == 42
        assert status.memory_utilization_pct == 15
        assert status.temperature_c == 65
        assert status.active_tasks == 2
