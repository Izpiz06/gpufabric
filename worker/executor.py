"""GPU kernel execution engine."""

import logging
import time
from typing import List, Tuple

logger = logging.getLogger("gpufabric.worker.executor")


class GPUExecutionError(Exception):
    """Raised when GPU execution fails or GPU runtime is absent."""

    pass


class GPUExecutor:
    """Executes computational workloads directly on NVIDIA GPUs."""

    def __init__(self):
        self.device_count = 0
        self.backend = self._detect_backend()

    def _detect_backend(self) -> str:
        try:
            import cupy as cp

            count = cp.cuda.runtime.getDeviceCount()
            if count > 0:
                self.device_count = count
                return "cupy"
        except Exception:
            pass

        try:
            import pycuda.driver as cuda

            cuda.init()
            count = cuda.Device.count()
            if count > 0:
                self.device_count = count
                return "pycuda"
        except Exception:
            pass

        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                self.device_count = torch.cuda.device_count()
                return "torch_cuda"
        except Exception:
            pass

        self.device_count = 0
        return "none"

    def is_gpu_ready(self) -> bool:
        if self.backend == "none":
            self.backend = self._detect_backend()
        return self.backend != "none"

    def warmup(self) -> None:
        """Run a tiny kernel on every GPU to absorb the one-time cold start.

        The first GPU call pays for CUDA context creation and kernel
        compilation (~250 ms on an RTX 3050). Doing it at startup keeps that
        cost out of the first client request. Failures are logged, never raised.
        """
        if not self.is_gpu_ready():
            logger.info("GPU warm-up skipped: no GPU backend available")
            return
        for device_index in range(self.device_count):
            try:
                _, elapsed_ms, _ = self.execute_vector_add([1.0], [1.0], device_index)
                logger.info(f"GPU {device_index} warm-up done ({elapsed_ms:.1f} ms)")
            except Exception as e:
                logger.warning(f"GPU {device_index} warm-up failed: {e}")

    def execute_vector_add(
        self, a: List[float], b: List[float], device_index: int = 0
    ) -> Tuple[List[float], float, str]:
        if len(a) != len(b):
            raise ValueError(f"Vector size mismatch: len(A)={len(a)} != len(B)={len(b)}")

        if not self.is_gpu_ready():
            raise GPUExecutionError("No CUDA-capable GPU found. GPU execution required.")

        if self.backend == "cupy":
            return self._exec_cupy(a, b, device_index)
        elif self.backend == "pycuda":
            return self._exec_pycuda(a, b, device_index)
        elif self.backend == "torch_cuda":
            return self._exec_torch(a, b, device_index)
        else:
            raise GPUExecutionError(f"Unsupported GPU backend: {self.backend}")

    def _exec_cupy(self, a: List[float], b: List[float], device_index: int):
        import cupy as cp

        with cp.cuda.Device(device_index):
            t0 = time.perf_counter()
            ga, gb = cp.asarray(a, dtype=cp.float32), cp.asarray(b, dtype=cp.float32)
            gc = ga + gb
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter()
            return gc.tolist(), (t1 - t0) * 1000.0, "cupy"

    def _exec_pycuda(self, a: List[float], b: List[float], device_index: int):
        import numpy as np
        import pycuda.driver as cuda
        import pycuda.gpuarray as gpuarray

        cuda.init()
        ctx = cuda.Device(device_index).make_context()
        try:
            t0 = time.perf_counter()
            ga = gpuarray.to_gpu(np.array(a, dtype=np.float32))
            gb = gpuarray.to_gpu(np.array(b, dtype=np.float32))
            gc = ga + gb
            ctx.synchronize()
            t1 = time.perf_counter()
            return gc.get().tolist(), (t1 - t0) * 1000.0, "pycuda"
        finally:
            ctx.pop()

    def _exec_torch(self, a: List[float], b: List[float], device_index: int):
        import torch

        dev = torch.device(f"cuda:{device_index}")
        t0 = time.perf_counter()
        ta = torch.tensor(a, dtype=torch.float32, device=dev)
        tb = torch.tensor(b, dtype=torch.float32, device=dev)
        tc = ta + tb
        torch.cuda.synchronize(dev)
        t1 = time.perf_counter()
        return tc.cpu().tolist(), (t1 - t0) * 1000.0, "torch_cuda"
