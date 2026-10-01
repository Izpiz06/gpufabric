"""NVIDIA GPU discovery and monitoring via NVML."""

import logging
import warnings
from typing import Any, Dict

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

        driver = ""
        try:
            drv = pynvml.nvmlSystemGetDriverVersion()
            driver = drv.decode("utf-8") if isinstance(drv, bytes) else str(drv)
        except Exception:
            pass

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
        """Fetch dynamic GPU load and temperature."""
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

        return {
            "device_index": device_index,
            "gpu_util": gpu_util,
            "mem_util": mem_util,
            "total_vram": mem.total,
            "free_vram": mem.free,
            "used_vram": mem.used,
            "temp": temp,
        }

    def shutdown(self):
        if self._initialized and HAS_PYNVML:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._initialized = False
