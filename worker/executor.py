"""GPU kernel execution engine (CuPy backend)."""

import logging
import time
from typing import List, Tuple

logger = logging.getLogger("gpufabric.worker.executor")


class GPUExecutionError(Exception):
    """Raised when GPU execution fails or GPU runtime is absent."""


class GPUExecutor:
    """Executes computational workloads directly on NVIDIA GPUs via CuPy."""

    backend_name = "cupy"

    def __init__(self):
        self.device_count = 0
        self._detect()

    def _detect(self) -> None:
        try:
            import cupy as cp

            self.device_count = cp.cuda.runtime.getDeviceCount()
        except Exception as e:
            logger.debug(f"CuPy GPU detection failed: {e}")
            self.device_count = 0

    def is_gpu_ready(self) -> bool:
        if self.device_count == 0:
            self._detect()
        return self.device_count > 0

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

        import cupy as cp

        with cp.cuda.Device(device_index):
            t0 = time.perf_counter()
            ga, gb = cp.asarray(a, dtype=cp.float32), cp.asarray(b, dtype=cp.float32)
            gc = ga + gb
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter()
            return gc.tolist(), (t1 - t0) * 1000.0, self.backend_name
