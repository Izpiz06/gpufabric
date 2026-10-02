"""GPU kernel execution engine (CuPy backend)."""

import logging
import time
from typing import List, Sequence, Tuple

import numpy as np

logger = logging.getLogger("gpufabric.worker.executor")

# Operation name -> CuPy function name.
OPERATIONS = {
    "vector_add": "add",
    "vector_mul": "multiply",
    "vector_dot": "dot",
    "matrix_add": "add",
    "matmul": "matmul",
}


def sample_inputs(op: str, dtype) -> List[np.ndarray]:
    """Tiny valid operands for `op`, used to compile its kernel during warm-up."""
    if op.startswith("vector"):
        return [np.ones(2, dtype=dtype), np.ones(2, dtype=dtype)]
    return [np.ones((2, 2), dtype=dtype), np.ones((2, 2), dtype=dtype)]


class GPUExecutionError(Exception):
    """Raised when GPU execution fails or GPU runtime is absent."""


class GPUOutOfMemoryError(GPUExecutionError):
    """Raised when the inputs or result do not fit in GPU memory."""


def validate_inputs(op: str, inputs: Sequence[np.ndarray]) -> None:
    """Check operand count, dtypes and shapes for an operation.

    Raises ValueError describing the first problem found.
    """
    if op not in OPERATIONS:
        raise ValueError(f"Unsupported operation: {op}")
    if len(inputs) != 2:
        raise ValueError(f"{op} takes 2 inputs, got {len(inputs)}")
    a, b = inputs
    if a.dtype != b.dtype:
        raise ValueError(f"{op} inputs must share a dtype, got {a.dtype} and {b.dtype}")

    if op in ("vector_add", "vector_mul", "vector_dot"):
        if a.ndim != 1 or b.ndim != 1:
            raise ValueError(f"{op} takes 1-D vectors, got shapes {a.shape} and {b.shape}")
        if a.shape != b.shape:
            raise ValueError(f"Vector size mismatch: {a.shape[0]} != {b.shape[0]}")
    elif op == "matrix_add":
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"{op} takes 2-D matrices, got shapes {a.shape} and {b.shape}")
        if a.shape != b.shape:
            raise ValueError(f"Matrix shape mismatch: {a.shape} != {b.shape}")
    elif op == "matmul":
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"{op} takes 2-D matrices, got shapes {a.shape} and {b.shape}")
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"matmul inner dimensions differ: {a.shape} @ {b.shape}")


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
        """Run every kernel once on every GPU to absorb one-time startup costs.

        The first GPU call pays for CUDA context creation (~250 ms on an
        RTX 3050), and each operation/dtype pair compiles its kernel on first
        use (30-110 ms each). Doing it at startup keeps those costs out of
        client requests. Failures are logged, never raised.
        """
        if not self.is_gpu_ready():
            logger.info("GPU warm-up skipped: no GPU backend available")
            return
        for device_index in range(self.device_count):
            t0 = time.perf_counter()
            try:
                self.execute_vector_add([1.0], [1.0], device_index)
                for dtype in (np.float32, np.float64):
                    for op in OPERATIONS:
                        self.compute(op, sample_inputs(op, dtype), device_index)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
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

    def compute(
        self, op: str, inputs: Sequence[np.ndarray], device_index: int = 0
    ) -> Tuple[np.ndarray, float, float]:
        """Run `op` on the GPU.

        Returns (result, gpu_time_ms, total_time_ms). gpu_time_ms covers only
        the kernel (CUDA events); total_time_ms also includes copying inputs
        to the GPU and the result back.
        """
        validate_inputs(op, inputs)
        if not self.is_gpu_ready():
            raise GPUExecutionError("No CUDA-capable GPU found. GPU execution required.")
        if not 0 <= device_index < self.device_count:
            raise ValueError(
                f"Invalid device_index {device_index}: worker has {self.device_count} GPU(s)"
            )

        import cupy as cp

        fn = getattr(cp, OPERATIONS[op])
        try:
            with cp.cuda.Device(device_index):
                t0 = time.perf_counter()
                gpu_inputs = [cp.asarray(x) for x in inputs]
                start, end = cp.cuda.Event(), cp.cuda.Event()
                start.record()
                out = fn(*gpu_inputs)
                end.record()
                end.synchronize()
                gpu_ms = cp.cuda.get_elapsed_time(start, end)
                result = cp.asnumpy(out)
                total_ms = (time.perf_counter() - t0) * 1000.0
        except cp.cuda.memory.OutOfMemoryError as e:
            cp.get_default_memory_pool().free_all_blocks()
            raise GPUOutOfMemoryError(str(e)) from e
        return result, gpu_ms, total_ms
