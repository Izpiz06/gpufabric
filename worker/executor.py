# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""GPU kernel execution engine (CuPy backend)."""

import logging
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger("gpufabric.worker.executor")

# Operation name -> CuPy function name ("triad" uses a custom fused kernel).
OPERATIONS = {
    "vector_add": "add",
    "vector_mul": "multiply",
    "vector_dot": "dot",
    "matrix_add": "add",
    "matmul": "matmul",
    "triad": None,
}

BENCHMARKS = ("triad", "matmul", "monte_carlo_pi")
MAX_BENCHMARK_REPEATS = 100
# Monte Carlo points are generated in batches to bound GPU memory use.
MONTE_CARLO_BATCH = 10_000_000

_triad_kernel = None


def _get_triad_kernel(cp):
    """Fused a = b + s * c. One pass over memory: read b, read c, write a.

    Plain CuPy `b + s * c` would launch two kernels and write a temporary
    array, moving 5 arrays' worth of data instead of 3.
    """
    global _triad_kernel
    if _triad_kernel is None:
        _triad_kernel = cp.ElementwiseKernel(
            "T b, T c, T s", "T a", "a = b + s * c", "gpufabric_triad"
        )
    return _triad_kernel


def sample_inputs(op: str, dtype) -> List[np.ndarray]:
    """Tiny valid operands for `op`, used to compile its kernel during warm-up."""
    if op == "triad":
        return [np.ones(2, dtype=dtype), np.ones(2, dtype=dtype), np.array(3.0, dtype=dtype)]
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
    arity = 3 if op == "triad" else 2
    if len(inputs) != arity:
        raise ValueError(f"{op} takes {arity} inputs, got {len(inputs)}")
    dtypes = {x.dtype for x in inputs}
    if len(dtypes) != 1:
        raise ValueError(f"{op} inputs must share a dtype, got {sorted(map(str, dtypes))}")
    a, b = inputs[:2]

    if op == "triad":
        if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
            raise ValueError(f"triad takes two equal 1-D vectors, got {a.shape} and {b.shape}")
        if inputs[2].ndim != 0:
            raise ValueError(f"triad scalar must be 0-D, got shape {inputs[2].shape}")
    elif op in ("vector_add", "vector_mul", "vector_dot"):
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

    def _check_device(self, device_index: int) -> None:
        if not self.is_gpu_ready():
            raise GPUExecutionError("No CUDA-capable GPU found. GPU execution required.")
        if not 0 <= device_index < self.device_count:
            raise ValueError(
                f"Invalid device_index {device_index}: worker has {self.device_count} GPU(s)"
            )

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
        self._check_device(device_index)

        import cupy as cp

        fn = _get_triad_kernel(cp) if op == "triad" else getattr(cp, OPERATIONS[op])
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

    def run_benchmark(
        self, name: str, size: int, device_index: int = 0, repeats: int = 5
    ) -> Dict[str, float]:
        """Run a benchmark on data generated on the GPU itself.

        One untimed warm-up run, then `repeats` runs timed with CUDA events.
        Returns best_ms and mean_ms plus the benchmark's metric:
        bandwidth_gb_s (triad), gflops (matmul) or pi_estimate (monte_carlo_pi).
        GPU memory is released afterwards so the worker doesn't hold it.
        """
        if name not in BENCHMARKS:
            raise ValueError(f"Unsupported benchmark: {name}")
        if size < 1:
            raise ValueError(f"Benchmark size must be positive, got {size}")
        if not 1 <= repeats <= MAX_BENCHMARK_REPEATS:
            raise ValueError(f"repeats must be between 1 and {MAX_BENCHMARK_REPEATS}")
        self._check_device(device_index)

        import cupy as cp

        runner = {
            "triad": self._bench_triad,
            "matmul": self._bench_matmul,
            "monte_carlo_pi": self._bench_monte_carlo_pi,
        }[name]
        try:
            with cp.cuda.Device(device_index):
                return runner(cp, size, repeats)
        except cp.cuda.memory.OutOfMemoryError as e:
            raise GPUOutOfMemoryError(str(e)) from e
        finally:
            cp.get_default_memory_pool().free_all_blocks()

    @staticmethod
    def _timed_runs(cp, fn, repeats: int):
        """Call fn() `repeats` times. Returns (times_ms, return values)."""
        times, values = [], []
        start, end = cp.cuda.Event(), cp.cuda.Event()
        for _ in range(repeats):
            start.record()
            values.append(fn())
            end.record()
            end.synchronize()
            times.append(cp.cuda.get_elapsed_time(start, end))
        return times, values

    @staticmethod
    def _summary(times: List[float]) -> Dict[str, float]:
        return {"best_ms": min(times), "mean_ms": sum(times) / len(times)}

    def _bench_triad(self, cp, n: int, repeats: int) -> Dict[str, float]:
        kernel = _get_triad_kernel(cp)
        b = cp.random.random(n, dtype=cp.float32)
        c = cp.random.random(n, dtype=cp.float32)
        a = cp.empty_like(b)
        s = cp.asarray(3.0, dtype=cp.float32)
        kernel(b, c, s, a)
        times, _ = self._timed_runs(cp, lambda: kernel(b, c, s, a), repeats)
        result = self._summary(times)
        bytes_moved = 3 * n * b.itemsize
        result["bandwidth_gb_s"] = bytes_moved / (result["best_ms"] / 1000.0) / 1e9
        return result

    def _bench_matmul(self, cp, n: int, repeats: int) -> Dict[str, float]:
        a = cp.random.random((n, n), dtype=cp.float32)
        b = cp.random.random((n, n), dtype=cp.float32)
        out = cp.empty((n, n), dtype=cp.float32)
        cp.matmul(a, b, out=out)
        times, _ = self._timed_runs(cp, lambda: cp.matmul(a, b, out=out), repeats)
        result = self._summary(times)
        result["gflops"] = 2.0 * n**3 / (result["best_ms"] / 1000.0) / 1e9
        return result

    def _bench_monte_carlo_pi(self, cp, n: int, repeats: int) -> Dict[str, float]:
        def count_inside(points: int):
            inside = cp.zeros((), dtype=cp.int64)
            remaining = points
            while remaining:
                m = min(MONTE_CARLO_BATCH, remaining)
                x = cp.random.random(m, dtype=cp.float32)
                y = cp.random.random(m, dtype=cp.float32)
                inside += cp.count_nonzero(x * x + y * y <= 1.0)
                remaining -= m
            return inside

        count_inside(min(n, MONTE_CARLO_BATCH))
        times, counts = self._timed_runs(cp, lambda: count_inside(n), repeats)
        result = self._summary(times)
        total_inside = sum(int(c) for c in counts)
        result["pi_estimate"] = 4.0 * total_inside / (n * repeats)
        return result
