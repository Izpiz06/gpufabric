"""NVIDIA GPU discovery and monitoring via NVML."""

import logging
import warnings
from typing import Any, Dict, List

logger = logging.getLogger("gpufabric.worker.gpu")

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        try:
            import nvidia_ml_py as pynvml
        except ImportError:
            import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False
    pynvml = None


class GPUManager:
    """Manages NVML initialization and GPU hardware queries."""

    def __init__(self):
        self._initialized = False
        self._init_nvml()

    def _init_nvml(self) -> bool:
        if not HAS_PYNVML:
            return False
        try:
            pynvml.nvmlInit()
            self._initialized = True
            return True
        except Exception as e:
            logger.debug(f"NVML init failed: {e}")
            self._initialized = False
            return False

    def is_available(self) -> bool:
        if not self._initialized:
            self._init_nvml()
        if not self._initialized:
            return False
        try:
            return pynvml.nvmlDeviceGetCount() > 0
        except Exception:
            return False

    def device_count(self) -> int:
        """Number of GPUs NVML can see (0 if NVML is unavailable)."""
        if not self.is_available():
            return 0
        return pynvml.nvmlDeviceGetCount()

    def list_devices(self) -> List[Dict[str, Any]]:
        """Specs and live status for every GPU, merged into one dict per device."""
        return [{**self.get_info(i), **self.get_status(i)} for i in range(self.device_count())]

    def peak_memory_bandwidth(self, device_index: int = 0) -> float:
        """Theoretical peak memory bandwidth in GB/s (1e9 bytes), or 0.0 if unknown.

        max memory clock (MHz) * 2 (double data rate) * bus width (bits) / 8.
        Matches published specs for GDDR6 (RTX 3050: 188 GB/s), GDDR6X and HBM.
        """
        if not self.is_available():
            return 0.0
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            bus_bits = pynvml.nvmlDeviceGetMemoryBusWidth(handle)
        except Exception as e:
            logger.debug(f"Peak bandwidth unavailable for GPU {device_index}: {e}")
            return 0.0
        return clock_mhz * 1e6 * 2 * bus_bits / 8 / 1e9

    def get_cuda_version(self) -> str:
        """System CUDA driver version formatted as X.Y (e.g. '12.2'), or empty string."""
        if not self.is_available():
            return ""
        try:
            raw = pynvml.nvmlSystemGetCudaDriverVersion()
            return f"{raw // 1000}.{(raw % 1000) // 10}"
        except Exception:
            return ""

    def get_driver_version(self) -> str:
        """NVIDIA driver version string, or empty string."""
        if not self.is_available():
            return ""
        try:
            drv = pynvml.nvmlSystemGetDriverVersion()
            return drv.decode("utf-8") if isinstance(drv, bytes) else str(drv)
        except Exception:
            return ""

    def get_info(self, device_index: int = 0) -> Dict[str, Any]:
        """Fetch static GPU device specs."""
        if not self.is_available():
            raise RuntimeError("GPU or NVML driver is unavailable.")

        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        name_raw = pynvml.nvmlDeviceGetName(handle)
        name = name_raw.decode("utf-8") if isinstance(name_raw, bytes) else str(name_raw)

        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

        compute_cap = ""
        try:
            maj, min_ = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            compute_cap = f"{maj}.{min_}"
        except Exception:
            pass

        driver = self.get_driver_version()

        return {
            "device_index": device_index,
            "name": name,
            "total_vram": mem.total,
            "free_vram": mem.free,
            "used_vram": mem.used,
            "compute_capability": compute_cap,
            "driver_version": driver,
        }

    def get_status(self, device_index: int = 0) -> Dict[str, Any]:
        """Fetch dynamic GPU load, temperature, and power telemetry."""
        if not self.is_available():
            raise RuntimeError("GPU or NVML driver is unavailable.")

        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

        gpu_util, mem_util = 0, 0
        try:
            rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util, mem_util = rates.gpu, rates.memory
        except Exception:
            pass

        temp = 0
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            pass

        power_w = 0
        try:
            power_w = int(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000)
        except Exception:
            pass

        power_limit_w = 0
        try:
            limit_mw = 0
            try:
                limit_mw = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle)
            except Exception:
                limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            power_limit_w = int(limit_mw / 1000)
        except Exception:
            pass

        return {
            "device_index": device_index,
            "gpu_util": gpu_util,
            "mem_util": mem_util,
            "total_vram": mem.total,
            "free_vram": mem.free,
            "used_vram": mem.used,
            "temp": temp,
            "power_usage_w": power_w,
            "power_limit_w": power_limit_w,
            "available": True,
        }

    def shutdown(self):
        if self._initialized and HAS_PYNVML:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._initialized = False
