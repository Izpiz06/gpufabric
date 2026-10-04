# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""gRPC Service implementation for GPU Fabric Worker."""

import uuid

import grpc

from common.constants import API_VERSION
from common.formatting import bytes_to_human
from common.gpufabric_pb2 import (
    Benchmark,
    BenchmarkResponse,
    ComputeResponse,
    ExecuteResponse,
    GPUDevice,
    GPUInfoResponse,
    GPUStatusResponse,
    HealthResponse,
    HealthState,
    ListGPUsResponse,
    Operation,
    WorkloadType,
)
from common.gpufabric_pb2_grpc import GPUFabricServiceServicer
from common.tensor import TensorError, from_tensor, to_tensor
from worker.executor import GPUExecutionError, GPUExecutor, GPUOutOfMemoryError
from worker.gpu import GPUManager
from worker.state import WorkerState

# Protobuf Operation enum -> executor operation name.
_OPERATIONS = {
    Operation.OP_VECTOR_ADD: "vector_add",
    Operation.OP_VECTOR_MUL: "vector_mul",
    Operation.OP_VECTOR_DOT: "vector_dot",
    Operation.OP_MATRIX_ADD: "matrix_add",
    Operation.OP_MATMUL: "matmul",
    Operation.OP_TRIAD: "triad",
}

_BENCHMARKS = {
    Benchmark.BENCH_TRIAD: "triad",
    Benchmark.BENCH_MATMUL: "matmul",
    Benchmark.BENCH_MONTE_CARLO_PI: "monte_carlo_pi",
}
DEFAULT_BENCHMARK_REPEATS = 5


class GPUFabricServicer(GPUFabricServiceServicer):
    """Handles gRPC client requests for GPU discovery and execution."""

    def __init__(
        self, worker_state: WorkerState, gpu_manager: GPUManager, gpu_executor: GPUExecutor
    ):
        self.state = worker_state
        self.gpu = gpu_manager
        self.executor = gpu_executor

    def _evaluate_health(self) -> tuple[int, str]:
        nvml_avail = self.gpu.is_available()
        cupy_ready = self.executor.is_gpu_ready()

        if not nvml_avail and not cupy_ready:
            return HealthState.UNAVAILABLE, "No GPU hardware or compute backend available."

        if nvml_avail and not cupy_ready:
            return (
                HealthState.DEGRADED,
                "GPU hardware detected by NVML, but CuPy compute backend is unavailable.",
            )

        if not nvml_avail and cupy_ready:
            return HealthState.DEGRADED, "Compute backend ready, but NVML telemetry is unavailable."

        try:
            count = self.gpu.device_count()
            if count == 0:
                return HealthState.UNAVAILABLE, "No GPU devices found."
            for i in range(count):
                status = self.gpu.get_status(i)
                temp = status.get("temp", 0)
                if temp >= 85:
                    return HealthState.DEGRADED, f"GPU {i} temperature high ({temp}°C)."
            return HealthState.HEALTHY, "All GPU systems operational."
        except Exception as e:
            return HealthState.DEGRADED, f"GPU telemetry error: {e}"

    def GetHealth(self, request, context) -> HealthResponse:
        self.state.heartbeat()
        state, details = self._evaluate_health()
        gpu_ready = state != HealthState.UNAVAILABLE
        cuda_ver = self.gpu.get_cuda_version()
        driver_ver = self.gpu.get_driver_version()
        try:
            gpu_count = (
                self.gpu.device_count()
                if self.gpu.is_available()
                else (self.executor.device_count if self.executor.is_gpu_ready() else 0)
            )
        except Exception:
            gpu_count = 0

        return HealthResponse(
            status="ok",
            worker_id=self.state.worker_id,
            version=API_VERSION,
            gpu_available=gpu_ready,
            health_state=state,
            hostname=self.state.hostname,
            api_version=API_VERSION,
            cuda_version=cuda_ver,
            driver_version=driver_ver,
            last_heartbeat=self.state.last_heartbeat,
            health_details=details,
            active_tasks=self.state.active_tasks,
            gpu_count=gpu_count,
        )

    def GetGPUInfo(self, request, context) -> GPUInfoResponse:
        if not self.gpu.is_available():
            context.abort(
                grpc.StatusCode.UNAVAILABLE, "No GPU or NVML driver found on this worker."
            )
        try:
            info = self.gpu.get_info(request.device_index)
            return GPUInfoResponse(
                device_index=info["device_index"],
                name=info["name"],
                total_vram_bytes=info["total_vram"],
                free_vram_bytes=info["free_vram"],
                used_vram_bytes=info["used_vram"],
                total_vram_human=bytes_to_human(info["total_vram"]),
                free_vram_human=bytes_to_human(info["free_vram"]),
                used_vram_human=bytes_to_human(info["used_vram"]),
                compute_capability=info["compute_capability"],
                driver_version=info["driver_version"],
            )
        except Exception as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))

    def GetGPUStatus(self, request, context) -> GPUStatusResponse:
        if not self.gpu.is_available():
            context.abort(
                grpc.StatusCode.UNAVAILABLE, "No GPU or NVML driver found on this worker."
            )
        try:
            self.state.heartbeat()
            s = self.gpu.get_status(request.device_index)
            workload = self.state.get_device_workload(request.device_index)
            return GPUStatusResponse(
                device_index=s["device_index"],
                gpu_utilization_pct=s["gpu_util"],
                memory_utilization_pct=s["mem_util"],
                total_vram_bytes=s["total_vram"],
                free_vram_bytes=s["free_vram"],
                used_vram_bytes=s["used_vram"],
                free_vram_human=bytes_to_human(s["free_vram"]),
                used_vram_human=bytes_to_human(s["used_vram"]),
                temperature_c=s["temp"],
                active_tasks=self.state.active_tasks,
                power_usage_w=s.get("power_usage_w", 0),
                power_limit_w=s.get("power_limit_w", 0),
                available=s.get("available", True),
                current_workload=workload,
            )
        except Exception as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))

    def Execute(self, request, context) -> ExecuteResponse:
        task_id = str(uuid.uuid4())
        if request.workload_type not in (WorkloadType.VECTOR_ADD, 1):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Unsupported workload type: {request.workload_type}",
            )

        if len(request.a) != len(request.b):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Vector size mismatch: len(a)={len(request.a)} != len(b)={len(request.b)}",
            )

        self.state.increment_tasks(request.device_index, "vector_add")
        try:
            res, elapsed_ms, backend = self.executor.execute_vector_add(
                list(request.a), list(request.b), request.device_index
            )
            return ExecuteResponse(
                task_id=task_id,
                workload_type="vector_add",
                status="success",
                result=res,
                result_length=len(res),
                execution_time_ms=elapsed_ms,
                device_index=request.device_index,
                gpu_backend=backend,
            )
        except GPUExecutionError as e:
            context.abort(grpc.StatusCode.UNAVAILABLE, str(e))
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, f"Workload execution failed: {e}")
        finally:
            self.state.decrement_tasks(request.device_index)

    def _run_gpu_task(self, context, label, fn, device_index: int = 0):
        """Run fn() as an active task and map executor errors to gRPC codes."""
        self.state.increment_tasks(device_index, label)
        try:
            return fn()
        except ValueError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except GPUOutOfMemoryError as e:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, f"GPU out of memory: {e}")
        except GPUExecutionError as e:
            context.abort(grpc.StatusCode.UNAVAILABLE, str(e))
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, f"{label} failed: {e}")
        finally:
            self.state.decrement_tasks(device_index)

    def Compute(self, request, context) -> ComputeResponse:
        op = _OPERATIONS.get(request.op)
        if op is None:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Unsupported operation: {request.op}")
        try:
            inputs = [from_tensor(t) for t in request.inputs]
        except TensorError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))

        task_id = str(uuid.uuid4())
        result, gpu_ms, total_ms = self._run_gpu_task(
            context,
            f"Compute:{op}",
            lambda: self.executor.compute(op, inputs, request.device_index),
            device_index=request.device_index,
        )
        return ComputeResponse(
            task_id=task_id,
            result=to_tensor(result),
            gpu_time_ms=gpu_ms,
            total_time_ms=total_ms,
            device_index=request.device_index,
        )

    def ListGPUs(self, request, context) -> ListGPUsResponse:
        self.state.heartbeat()
        try:
            devices = self.gpu.list_devices()
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, f"Failed to query GPUs: {e}")

        state, _ = self._evaluate_health()
        cuda_ver = self.gpu.get_cuda_version()
        driver_ver = devices[0]["driver_version"] if devices else ""

        gpus = []
        for d in devices:
            dev_idx = d["device_index"]
            workload = self.state.get_device_workload(dev_idx)
            gpus.append(
                GPUDevice(
                    device_index=dev_idx,
                    name=d["name"],
                    total_vram_bytes=d["total_vram"],
                    free_vram_bytes=d["free_vram"],
                    compute_capability=d["compute_capability"],
                    gpu_utilization_pct=d["gpu_util"],
                    temperature_c=d["temp"],
                    power_usage_w=d.get("power_usage_w", 0),
                    power_limit_w=d.get("power_limit_w", 0),
                    memory_utilization_pct=d.get("mem_util", 0),
                    available=d.get("available", True),
                    current_workload=workload,
                )
            )

        return ListGPUsResponse(
            worker_id=self.state.worker_id,
            version=API_VERSION,
            driver_version=driver_ver,
            compute_ready=self.executor.is_gpu_ready(),
            gpus=gpus,
            health_state=state,
            cuda_version=cuda_ver,
            hostname=self.state.hostname,
            last_heartbeat=self.state.last_heartbeat,
        )

    def RunBenchmark(self, request, context) -> BenchmarkResponse:
        name = _BENCHMARKS.get(request.benchmark)
        if name is None:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, f"Unsupported benchmark: {request.benchmark}"
            )
        repeats = request.repeats or DEFAULT_BENCHMARK_REPEATS
        r = self._run_gpu_task(
            context,
            f"Benchmark:{name}",
            lambda: self.executor.run_benchmark(name, request.size, request.device_index, repeats),
            device_index=request.device_index,
        )
        peak = self.gpu.peak_memory_bandwidth(request.device_index) if name == "triad" else 0.0
        return BenchmarkResponse(
            benchmark=request.benchmark,
            size=request.size,
            device_index=request.device_index,
            repeats=repeats,
            best_ms=r["best_ms"],
            mean_ms=r["mean_ms"],
            bandwidth_gb_s=r.get("bandwidth_gb_s", 0.0),
            peak_bandwidth_gb_s=peak,
            gflops=r.get("gflops", 0.0),
            pi_estimate=r.get("pi_estimate", 0.0),
        )
