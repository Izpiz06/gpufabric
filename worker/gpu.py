"""GPU management and monitoring module using NVML."""

import logging
from typing import Optional

from common.models import GPUInfoResponse, GPUStatusResponse
from common.protocol import bytes_to_human

import warnings

logger = logging.getLogger("gpufabric.worker.gpu")

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        try:
            import pynvml
        except ImportError:
            import pynvml3 as pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False
    pynvml = None


class GPUManager:
    """Manages NVIDIA GPU discovery and status reporting via pynvml."""

    def __init__(self):
        self._nvml_initialized = False
        self._init_nvml()

    def _init_nvml(self) -> bool:
        if not HAS_PYNVML:
            logger.warning("pynvml module not installed. GPU monitoring unavailable.")
            return False
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            count = pynvml.nvmlDeviceGetCount()
            logger.info(f"NVML initialized successfully. Found {count} GPU device(s).")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize NVML: {e}")
            self._nvml_initialized = False
            return False

    def is_available(self) -> bool:
        """Return True if NVML is initialized and at least 1 GPU is detected."""
        if not self._nvml_initialized:
            self._init_nvml()
        if not self._nvml_initialized:
            return False
        try:
            return pynvml.nvmlDeviceGetCount() > 0
        except Exception:
            return False

    def get_device_count(self) -> int:
        """Return the number of available GPU devices."""
        if not self.is_available():
            return 0
        try:
            return pynvml.nvmlDeviceGetCount()
        except Exception:
            return 0

    def get_gpu_info(self, device_index: int = 0) -> GPUInfoResponse:
        """Fetch static and current memory information for a specific GPU device."""
        if not self.is_available():
            raise RuntimeError("No GPU available or NVML driver is uninitialized.")

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as e:
            raise ValueError(f"Invalid device index {device_index}: {e}")

        # GPU Name
        name_raw = pynvml.nvmlDeviceGetName(handle)
        name = name_raw.decode("utf-8") if isinstance(name_raw, bytes) else str(name_raw)

        # Memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_bytes = mem_info.total
        free_bytes = mem_info.free
        used_bytes = mem_info.used

        # Compute capability
        compute_cap: Optional[str] = None
        try:
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            compute_cap = f"{major}.{minor}"
        except Exception:
            compute_cap = None

        # Driver version
        driver_ver: Optional[str] = None
        try:
            driver_ver_raw = pynvml.nvmlSystemGetDriverVersion()
            driver_ver = (
                driver_ver_raw.decode("utf-8")
                if isinstance(driver_ver_raw, bytes)
                else str(driver_ver_raw)
            )
        except Exception:
            driver_ver = None

        return GPUInfoResponse(
            device_index=device_index,
            name=name,
            total_vram_bytes=total_bytes,
            free_vram_bytes=free_bytes,
            used_vram_bytes=used_bytes,
            total_vram_human=bytes_to_human(total_bytes),
            free_vram_human=bytes_to_human(free_bytes),
            used_vram_human=bytes_to_human(used_bytes),
            compute_capability=compute_cap,
            driver_version=driver_ver,
        )

    def get_gpu_status(self, device_index: int = 0, active_tasks: int = 0) -> GPUStatusResponse:
        """Fetch dynamic status (utilization, memory, temperature) for a GPU."""
        if not self.is_available():
            raise RuntimeError("No GPU available or NVML driver is uninitialized.")

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as e:
            raise ValueError(f"Invalid device index {device_index}: {e}")

        # Memory
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_bytes = mem_info.total
        free_bytes = mem_info.free
        used_bytes = mem_info.used

        # Utilization
        gpu_util = None
        mem_util = None
        try:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu
            mem_util = utilization.memory
        except Exception:
            pass

        # Temperature
        temp = None
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            pass

        return GPUStatusResponse(
            device_index=device_index,
            gpu_utilization_pct=gpu_util,
            memory_utilization_pct=mem_util,
            total_vram_bytes=total_bytes,
            free_vram_bytes=free_bytes,
            used_vram_bytes=used_bytes,
            free_vram_human=bytes_to_human(free_bytes),
            used_vram_human=bytes_to_human(used_bytes),
            temperature_c=temp,
            active_tasks=active_tasks,
        )

    def shutdown(self):
        """Shutdown NVML if initialized."""
        if self._nvml_initialized and HAS_PYNVML:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False
