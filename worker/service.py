"""gRPC Service implementation for GPU Fabric Worker."""

import uuid

import grpc

from common.constants import API_VERSION
from common.formatting import bytes_to_human
from common.gpufabric_pb2 import (
    ComputeResponse,
    ExecuteResponse,
    GPUInfoResponse,
    GPUStatusResponse,
    HealthResponse,
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
}


class GPUFabricServicer(GPUFabricServiceServicer):
    """Handles gRPC client requests for GPU discovery and execution."""

    def __init__(
        self, worker_state: WorkerState, gpu_manager: GPUManager, gpu_executor: GPUExecutor
    ):
        self.state = worker_state
        self.gpu = gpu_manager
        self.executor = gpu_executor

    def GetHealth(self, request, context) -> HealthResponse:
        gpu_ready = self.gpu.is_available() or self.executor.is_gpu_ready()
        return HealthResponse(
            status="ok",
            worker_id=self.state.worker_id,
            version=API_VERSION,
            gpu_available=gpu_ready,
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
            s = self.gpu.get_status(request.device_index)
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

        self.state.increment_tasks()
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
            self.state.decrement_tasks()

    def Compute(self, request, context) -> ComputeResponse:
        op = _OPERATIONS.get(request.op)
        if op is None:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Unsupported operation: {request.op}")
        try:
            inputs = [from_tensor(t) for t in request.inputs]
        except TensorError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))

        task_id = str(uuid.uuid4())
        self.state.increment_tasks()
        try:
            result, gpu_ms, total_ms = self.executor.compute(op, inputs, request.device_index)
        except ValueError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except GPUOutOfMemoryError as e:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, f"GPU out of memory: {e}")
        except GPUExecutionError as e:
            context.abort(grpc.StatusCode.UNAVAILABLE, str(e))
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, f"Compute failed: {e}")
        finally:
            self.state.decrement_tasks()

        return ComputeResponse(
            task_id=task_id,
            result=to_tensor(result),
            gpu_time_ms=gpu_ms,
            total_time_ms=total_ms,
            device_index=request.device_index,
        )
