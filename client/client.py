"""gRPC client SDK for GPU Fabric."""

import time
from dataclasses import dataclass
from typing import List, Optional

import grpc
import numpy as np

from common.constants import DEFAULT_MAX_MESSAGE_MB, DEFAULT_PORT
from common.gpufabric_pb2 import (
    Benchmark,
    BenchmarkRequest,
    BenchmarkResponse,
    ComputeRequest,
    ExecuteRequest,
    ExecuteResponse,
    GPUInfoRequest,
    GPUInfoResponse,
    GPUStatusRequest,
    GPUStatusResponse,
    HealthRequest,
    HealthResponse,
    ListGPUsRequest,
    ListGPUsResponse,
    Operation,
    WorkloadType,
)
from common.gpufabric_pb2_grpc import GPUFabricServiceStub
from common.grpc_options import message_size_options
from common.tensor import from_tensor, to_tensor

OPERATIONS = {
    "vector_add": Operation.OP_VECTOR_ADD,
    "vector_mul": Operation.OP_VECTOR_MUL,
    "vector_dot": Operation.OP_VECTOR_DOT,
    "matrix_add": Operation.OP_MATRIX_ADD,
    "matmul": Operation.OP_MATMUL,
    "triad": Operation.OP_TRIAD,
}

BENCHMARKS = {
    "triad": Benchmark.BENCH_TRIAD,
    "matmul": Benchmark.BENCH_MATMUL,
    "monte_carlo_pi": Benchmark.BENCH_MONTE_CARLO_PI,
}


class GPUFabricError(Exception):
    """Base exception for GPU Fabric client errors.

    `code` is the gRPC status code when the error came from an RPC, so
    callers can tell "retry later" (UNAVAILABLE) from "fix the request"
    (INVALID_ARGUMENT) or "too big for the GPU" (RESOURCE_EXHAUSTED).
    """

    def __init__(self, message: str, code: Optional[grpc.StatusCode] = None):
        super().__init__(message)
        self.code = code


def _rpc_error(prefix: str, e: grpc.RpcError) -> GPUFabricError:
    return GPUFabricError(f"{prefix}: {e.details() or e.code()}", code=e.code())


@dataclass
class ComputeResult:
    """Result of a remote Compute call."""

    result: np.ndarray
    task_id: str
    device_index: int
    gpu_time_ms: float  # kernel only, measured on the worker
    total_time_ms: float  # worker: copy in + kernel + copy out
    round_trip_ms: float  # client: serialize + network + worker + deserialize


class GPUFabricClient:
    """Client for querying GPU workers and executing workloads via gRPC."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = DEFAULT_PORT,
        timeout: float = 15.0,
        max_message_mb: int = DEFAULT_MAX_MESSAGE_MB,
    ):
        # Format address
        if ":" in host:
            self.target = host
        else:
            self.target = f"{host}:{port}"

        self.timeout = timeout
        self._channel = grpc.insecure_channel(
            self.target, options=message_size_options(max_message_mb)
        )
        self._stub = GPUFabricServiceStub(self._channel)

    def close(self):
        self._channel.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def health(self) -> HealthResponse:
        try:
            return self._stub.GetHealth(HealthRequest(), timeout=self.timeout)
        except grpc.RpcError as e:
            raise _rpc_error(f"Health check failed on {self.target}", e) from e

    def discover(self) -> HealthResponse:
        return self.health()

    def get_gpu_info(self, device_index: int = 0) -> GPUInfoResponse:
        try:
            req = GPUInfoRequest(device_index=device_index)
            return self._stub.GetGPUInfo(req, timeout=self.timeout)
        except grpc.RpcError as e:
            raise _rpc_error("Failed to fetch GPU info", e) from e

    def get_status(self, device_index: int = 0) -> GPUStatusResponse:
        try:
            req = GPUStatusRequest(device_index=device_index)
            return self._stub.GetGPUStatus(req, timeout=self.timeout)
        except grpc.RpcError as e:
            raise _rpc_error("Failed to fetch GPU status", e) from e

    def list_gpus(self) -> ListGPUsResponse:
        """Every GPU on the worker with specs and live status."""
        try:
            return self._stub.ListGPUs(ListGPUsRequest(), timeout=self.timeout)
        except grpc.RpcError as e:
            raise _rpc_error(f"Failed to list GPUs on {self.target}", e) from e

    def execute_vector_add(
        self, a: List[float], b: List[float], device_index: int = 0
    ) -> ExecuteResponse:
        try:
            req = ExecuteRequest(
                workload_type=WorkloadType.VECTOR_ADD,
                a=a,
                b=b,
                device_index=device_index,
            )
            return self._stub.Execute(req, timeout=self.timeout)
        except grpc.RpcError as e:
            raise _rpc_error(f"Execute failed on {self.target}", e) from e

    def compute(self, op: str, *inputs, device_index: int = 0) -> ComputeResult:
        """Run an operation on the worker GPU.

        `op` is one of OPERATIONS ("vector_add", "matmul", ...). Inputs are
        numpy arrays (or array-likes) of float32 or float64.
        """
        if op not in OPERATIONS:
            raise ValueError(f"Unknown operation {op!r}; choose from {sorted(OPERATIONS)}")
        req = ComputeRequest(
            op=OPERATIONS[op],
            inputs=[to_tensor(x) for x in inputs],
            device_index=device_index,
        )
        t0 = time.perf_counter()
        try:
            resp = self._stub.Compute(req, timeout=self.timeout)
        except grpc.RpcError as e:
            raise _rpc_error(f"Compute {op} failed", e) from e
        round_trip_ms = (time.perf_counter() - t0) * 1000.0
        return ComputeResult(
            result=from_tensor(resp.result),
            task_id=resp.task_id,
            device_index=resp.device_index,
            gpu_time_ms=resp.gpu_time_ms,
            total_time_ms=resp.total_time_ms,
            round_trip_ms=round_trip_ms,
        )

    def run_benchmark(
        self, name: str, size: int, repeats: int = 5, device_index: int = 0
    ) -> BenchmarkResponse:
        """Run a benchmark on data generated on the worker GPU (see BENCHMARKS)."""
        if name not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark {name!r}; choose from {sorted(BENCHMARKS)}")
        req = BenchmarkRequest(
            benchmark=BENCHMARKS[name], size=size, repeats=repeats, device_index=device_index
        )
        try:
            return self._stub.RunBenchmark(req, timeout=self.timeout)
        except grpc.RpcError as e:
            raise _rpc_error(f"Benchmark {name} failed", e) from e

    def triad(self, b, c, s, device_index: int = 0) -> ComputeResult:
        """a = b + s * c on the worker GPU (s is a scalar)."""
        return self.compute(
            "triad", b, c, np.asarray(s, dtype=np.asarray(b).dtype), device_index=device_index
        )

    def vector_add(self, a, b, device_index: int = 0) -> ComputeResult:
        return self.compute("vector_add", a, b, device_index=device_index)

    def vector_mul(self, a, b, device_index: int = 0) -> ComputeResult:
        return self.compute("vector_mul", a, b, device_index=device_index)

    def vector_dot(self, a, b, device_index: int = 0) -> ComputeResult:
        return self.compute("vector_dot", a, b, device_index=device_index)

    def matrix_add(self, a, b, device_index: int = 0) -> ComputeResult:
        return self.compute("matrix_add", a, b, device_index=device_index)

    def matmul(self, a, b, device_index: int = 0) -> ComputeResult:
        return self.compute("matmul", a, b, device_index=device_index)
