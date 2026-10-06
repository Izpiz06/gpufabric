# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""CUDA runtime environment and library diagnostics."""

import ctypes.util
import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger("gpufabric.worker.diagnostics")

_LIB_EXT_RE = re.compile(r"([a-zA-Z0-9_\-\.]+\.(?:so(?:\.\d+)*|dll))", re.IGNORECASE)


@dataclass
class CudaDiagnostics:
    """Diagnostic state of host GPU driver, CUDA runtime, CuPy, and math libraries."""

    gpu_detected: bool = False
    gpu_name: str = ""
    gpu_count: int = 0
    driver_version: str = ""
    cuda_driver_version: str = ""
    nvml_available: bool = False
    cupy_available: bool = False
    cudart_available: bool = False
    cudart_version: str = ""
    cublas_available: bool = False
    cublas_version: str = ""
    missing_libraries: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    remediation_hints: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """Overall health status: HEALTHY, DEGRADED, or UNAVAILABLE."""
        if not self.gpu_detected and not self.cupy_available:
            return "UNAVAILABLE"
        if not self.is_healthy:
            return "DEGRADED"
        return "HEALTHY"

    @property
    def is_healthy(self) -> bool:
        """Returns True if all hardware, runtime, and math library dependencies are intact."""
        return (
            self.gpu_detected
            and self.cupy_available
            and self.cublas_available
            and not self.missing_libraries
        )

    def summary(self) -> str:
        """Structured multi-line summary of host CUDA runtime diagnostics."""
        gpu_str = (
            f"{self.gpu_name} (Count: {self.gpu_count})"
            if self.gpu_name
            else (f"{self.gpu_count} GPU(s)" if self.gpu_detected else "None detected")
        )
        if self.cuda_driver_version or self.driver_version:
            gpu_str += f" [Driver: {self.driver_version or 'unknown'}, CUDA Driver: {self.cuda_driver_version or 'unknown'}]"

        cuda_rt_str = (
            f"AVAILABLE ({self.cudart_version})"
            if self.cudart_available and self.cudart_version
            else (
                "AVAILABLE"
                if self.cudart_available
                else ("MISSING" if self.gpu_detected else "N/A")
            )
        )

        cupy_str = "AVAILABLE" if self.cupy_available else "MISSING"

        if self.cublas_available:
            cublas_str = (
                f"AVAILABLE ({self.cublas_version})" if self.cublas_version else "AVAILABLE"
            )
        else:
            missing_tag = (
                f" ({', '.join(self.missing_libraries)} not found)"
                if self.missing_libraries
                else ""
            )
            cublas_str = f"MISSING{missing_tag}"

        lines = [
            f"Worker Status: {self.status}",
            f"GPU detected: {gpu_str}",
            f"CUDA runtime: {cuda_rt_str}",
            f"CuPy: {cupy_str}",
            f"cuBLAS: {cublas_str}",
        ]
        for w in self.warnings:
            lines.append(f"Warning: {w}")
        for h in self.remediation_hints:
            lines.append(f"Hint: {h}")

        return "\n".join(lines)


def _extract_missing_library_names(error_str: str) -> List[str]:
    """Extract library names from error strings (e.g. 'libcublas.so.12')."""
    return [match for match in _LIB_EXT_RE.findall(error_str) if match]


def probe_cuda_environment(
    gpu_manager: Optional[Any] = None, executor: Optional[Any] = None
) -> CudaDiagnostics:
    """Proactively probe GPU hardware, NVML, CUDA runtime, CuPy backend, and cuBLAS library."""
    diag = CudaDiagnostics()

    # 1. Probe NVML and GPU hardware
    if gpu_manager is not None:
        try:
            if gpu_manager.is_available():
                diag.nvml_available = True
                diag.gpu_count = gpu_manager.device_count()
                diag.gpu_detected = diag.gpu_count > 0
                diag.driver_version = gpu_manager.get_driver_version()
                diag.cuda_driver_version = gpu_manager.get_cuda_version()
                if diag.gpu_count > 0:
                    try:
                        info = gpu_manager.get_info(0)
                        diag.gpu_name = info.get("name", "")
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"NVML probe encountered error: {e}")

    # 2. Probe CuPy and CUDA Runtime
    try:
        import cupy as cp

        diag.cupy_available = True
        try:
            dev_count = cp.cuda.runtime.getDeviceCount()
            if dev_count > 0:
                diag.gpu_detected = True
                diag.gpu_count = max(diag.gpu_count, dev_count)
            raw_v = cp.cuda.runtime.runtimeGetVersion()
            major = raw_v // 1000
            minor = (raw_v % 1000) // 10
            diag.cudart_version = f"{major}.{minor}"
            diag.cudart_available = True
        except Exception as e:
            logger.debug(f"CuPy runtime version probe error: {e}")
            cudart_lib = ctypes.util.find_library("cudart")
            if cudart_lib:
                diag.cudart_available = True
                diag.cudart_version = cudart_lib
            elif diag.gpu_detected:
                diag.missing_libraries.append("libcudart.so")
    except ImportError:
        diag.cupy_available = False
        if "cupy" not in diag.missing_libraries:
            diag.missing_libraries.append("cupy")
        diag.remediation_hints.append(
            "Install CuPy matching your CUDA version (e.g. pip install cupy-cuda12x)."
        )
    except Exception as e:
        diag.cupy_available = False
        err_msg = str(e)
        logger.debug(f"CuPy import error: {err_msg}")
        for lib in _extract_missing_library_names(err_msg):
            if lib not in diag.missing_libraries:
                diag.missing_libraries.append(lib)

    # 3. Probe cuBLAS (libcublas)
    if diag.cupy_available and diag.gpu_detected:
        try:
            import cupy as cp

            with cp.cuda.Device(0):
                cublas = getattr(cp.cuda, "cublas", None)
                if cublas is None:
                    import cupy.cuda.cublas as cublas

                handle = cublas.create()
                try:
                    v = cublas.getVersion(handle)
                    diag.cublas_version = str(v)
                    diag.cublas_available = True
                finally:
                    cublas.destroy(handle)
        except Exception as e:
            diag.cublas_available = False
            err_msg = str(e)
            logger.debug(f"cuBLAS probe failed: {err_msg}")
            libs = _extract_missing_library_names(err_msg)
            if libs:
                for lib in libs:
                    if lib not in diag.missing_libraries:
                        diag.missing_libraries.append(lib)
            else:
                cublas_lib = ctypes.util.find_library("cublas")
                if not cublas_lib and "libcublas.so" not in diag.missing_libraries:
                    diag.missing_libraries.append("libcublas.so")
            diag.warnings.append("Some operations (such as matmul) may be unavailable.")
            diag.remediation_hints.append(
                "Install NVIDIA cuBLAS runtime (e.g. pip install nvidia-cublas-cu12 or CUDA Toolkit)."
            )
    elif diag.gpu_detected and not diag.cupy_available:
        cublas_lib = ctypes.util.find_library("cublas")
        if not cublas_lib and "libcublas.so" not in diag.missing_libraries:
            diag.missing_libraries.append("libcublas.so")

    # If cuBLAS is unavailable on a detected GPU, ensure remediation advice is added
    if diag.gpu_detected and not diag.cublas_available:
        if "Some operations (such as matmul) may be unavailable." not in diag.warnings:
            diag.warnings.append("Some operations (such as matmul) may be unavailable.")
        hint = (
            "Install NVIDIA cuBLAS runtime (e.g. pip install nvidia-cublas-cu12 or CUDA Toolkit)."
        )
        if hint not in diag.remediation_hints:
            diag.remediation_hints.append(hint)

    return diag
