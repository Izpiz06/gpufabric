"""GPU execution engine for workloads."""

import logging
import time
from typing import List, Tuple

logger = logging.getLogger("gpufabric.worker.executor")


class GPUExecutionError(Exception):
    """Raised when GPU execution fails or GPU is unavailable."""

    pass


class GPUExecutor:
    """Executes computational workloads on physical GPUs."""

    def __init__(self):
        self.backend = self._detect_backend()
        logger.info(f"GPUExecutor initialized with backend: {self.backend}")

    def _detect_backend(self) -> str:
        # Check CuPy
        try:
            import cupy as cp

            # Verify CUDA runtime / device count
            if cp.cuda.runtime.getDeviceCount() > 0:
                return "cupy"
        except Exception:
            pass

        # Check PyCUDA
        try:
            import pycuda.driver as cuda

            cuda.init()
            if cuda.Device.count() > 0:
                return "pycuda"
        except Exception:
            pass

        # Check PyTorch with CUDA
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                return "torch_cuda"
        except Exception:
            pass

        return "none"

    def is_gpu_ready(self) -> bool:
        """Check if a real GPU execution backend is available."""
        if self.backend == "none":
            self.backend = self._detect_backend()
        return self.backend != "none"

    def execute_vector_add(
        self, a: List[float], b: List[float], device_index: int = 0
    ) -> Tuple[List[float], float, str]:
        """
        Execute vector addition C = A + B on the worker's GPU.

        Returns:
            Tuple of (result_list, execution_time_ms, backend_name)

        Raises:
            ValueError: If vector lengths mismatch.
            GPUExecutionError: If no GPU is available or CUDA execution fails.
        """
        if len(a) != len(b):
            raise ValueError(f"Vector dimensions mismatch: len(A)={len(a)} != len(B)={len(b)}")

        if not self.is_gpu_ready():
            raise GPUExecutionError(
                "No CUDA-capable GPU execution backend is available on this worker. "
                "GPU Fabric requires a real GPU (CuPy, PyCUDA, or PyTorch CUDA)."
            )

        if self.backend == "cupy":
            return self._execute_cupy(a, b, device_index)
        elif self.backend == "pycuda":
            return self._execute_pycuda(a, b, device_index)
        elif self.backend == "torch_cuda":
            return self._execute_torch_cuda(a, b, device_index)
        else:
            raise GPUExecutionError(f"Unsupported GPU backend: {self.backend}")

    def _execute_cupy(
        self, a: List[float], b: List[float], device_index: int
    ) -> Tuple[List[float], float, str]:
        import cupy as cp

        try:
            with cp.cuda.Device(device_index):
                # Transfer host data to GPU
                t0 = time.perf_counter()
                gpu_a = cp.asarray(a, dtype=cp.float32)
                gpu_b = cp.asarray(b, dtype=cp.float32)

                # Execute kernel on GPU
                gpu_c = gpu_a + gpu_b

                # Synchronize stream to ensure GPU computation completes
                cp.cuda.Stream.null.synchronize()
                t1 = time.perf_counter()

                # Transfer back to host
                result = gpu_c.tolist()
                execution_time_ms = (t1 - t0) * 1000.0
                return result, execution_time_ms, "cupy"
        except Exception as e:
            raise GPUExecutionError(f"CuPy execution failed on device {device_index}: {e}")

    def _execute_pycuda(
        self, a: List[float], b: List[float], device_index: int
    ) -> Tuple[List[float], float, str]:
        import numpy as np
        import pycuda.driver as cuda
        import pycuda.gpuarray as gpuarray

        try:
            cuda.init()
            dev = cuda.Device(device_index)
            ctx = dev.make_context()
            try:
                np_a = np.array(a, dtype=np.float32)
                np_b = np.array(b, dtype=np.float32)

                t0 = time.perf_counter()
                gpu_a = gpuarray.to_gpu(np_a)
                gpu_b = gpuarray.to_gpu(np_b)

                gpu_c = gpu_a + gpu_b
                ctx.synchronize()
                t1 = time.perf_counter()

                result = gpu_c.get().tolist()
                execution_time_ms = (t1 - t0) * 1000.0
                return result, execution_time_ms, "pycuda"
            finally:
                ctx.pop()
        except Exception as e:
            raise GPUExecutionError(f"PyCUDA execution failed on device {device_index}: {e}")

    def _execute_torch_cuda(
        self, a: List[float], b: List[float], device_index: int
    ) -> Tuple[List[float], float, str]:
        import torch

        try:
            device = torch.device(f"cuda:{device_index}")
            t0 = time.perf_counter()

            # Transfer to GPU device
            tensor_a = torch.tensor(a, dtype=torch.float32, device=device)
            tensor_b = torch.tensor(b, dtype=torch.float32, device=device)

            # Execute on GPU
            tensor_c = tensor_a + tensor_b

            # Synchronize CUDA device
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()

            # Copy back to host
            result = tensor_c.cpu().tolist()
            execution_time_ms = (t1 - t0) * 1000.0
            return result, execution_time_ms, "torch_cuda"
        except Exception as e:
            raise GPUExecutionError(f"PyTorch CUDA execution failed on device {device_index}: {e}")
