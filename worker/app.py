"""FastAPI worker application for GPU Fabric."""

import logging
import socket
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, status

from common.models import (
    ExecuteRequest,
    ExecuteResponse,
    GPUInfoResponse,
    GPUStatusResponse,
    HealthResponse,
    WorkloadType,
)
from common.protocol import (
    API_VERSION,
    EXECUTE_ENDPOINT,
    GPU_INFO_ENDPOINT,
    HEALTH_ENDPOINT,
    STATUS_ENDPOINT,
)
from worker.gpu import GPUManager
from worker.executor import GPUExecutor, GPUExecutionError

logger = logging.getLogger("gpufabric.worker")


class WorkerState:
    """Holds runtime worker state, GPU manager, and executor."""

    def __init__(self, worker_id: Optional[str] = None):
        self.worker_id = worker_id or f"worker-{socket.gethostname()}"
        self.gpu_manager = GPUManager()
        self.gpu_executor = GPUExecutor()
        self._active_tasks = 0
        self._task_lock = threading.Lock()

    @property
    def active_tasks(self) -> int:
        with self._task_lock:
            return self._active_tasks

    def increment_active_tasks(self):
        with self._task_lock:
            self._active_tasks += 1

    def decrement_active_tasks(self):
        with self._task_lock:
            self._active_tasks = max(0, self._active_tasks - 1)


def create_app(worker_id: Optional[str] = None) -> FastAPI:
    """Factory to create and configure the FastAPI worker app."""
    worker_state = WorkerState(worker_id=worker_id)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(f"Starting GPU Fabric worker '{worker_state.worker_id}' (v{API_VERSION})")
        yield
        logger.info("Shutting down GPU Fabric worker")
        worker_state.gpu_manager.shutdown()

    app = FastAPI(
        title="GPU Fabric Worker",
        version=API_VERSION,
        description="Worker node API for GPU Fabric remote GPU access and execution.",
        lifespan=lifespan,
    )
    app.state.worker = worker_state

    @app.get(
        HEALTH_ENDPOINT,
        response_model=HealthResponse,
        summary="Worker health check",
    )
    async def get_health() -> HealthResponse:
        """Returns the health status and GPU availability of this worker."""
        gpu_ready = worker_state.gpu_manager.is_available() or worker_state.gpu_executor.is_gpu_ready()
        return HealthResponse(
            status="ok",
            worker_id=worker_state.worker_id,
            version=API_VERSION,
            gpu_available=gpu_ready,
        )

    @app.get(
        GPU_INFO_ENDPOINT,
        response_model=GPUInfoResponse,
        summary="Get GPU hardware information",
    )
    async def get_gpu_info(
        device_index: int = Query(0, description="CUDA device index (default: 0)")
    ) -> GPUInfoResponse:
        """Returns GPU specifications (VRAM, compute capability, driver version)."""
        if not worker_state.gpu_manager.is_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GPU monitoring is unavailable or no NVIDIA GPU was detected on this worker.",
            )
        try:
            return worker_state.gpu_manager.get_gpu_info(device_index=device_index)
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @app.get(
        STATUS_ENDPOINT,
        response_model=GPUStatusResponse,
        summary="Get dynamic GPU and worker status",
    )
    async def get_status(
        device_index: int = Query(0, description="CUDA device index (default: 0)")
    ) -> GPUStatusResponse:
        """Returns real-time GPU load, VRAM usage, temperature, and active task count."""
        if not worker_state.gpu_manager.is_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GPU monitoring is unavailable or no NVIDIA GPU was detected on this worker.",
            )
        try:
            return worker_state.gpu_manager.get_gpu_status(
                device_index=device_index,
                active_tasks=worker_state.active_tasks,
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @app.post(
        EXECUTE_ENDPOINT,
        response_model=ExecuteResponse,
        summary="Execute workload on local GPU",
    )
    async def execute_workload(request: ExecuteRequest) -> ExecuteResponse:
        """Submits a computational workload to execute directly on the worker's GPU."""
        task_id = str(uuid.uuid4())

        if request.workload_type != WorkloadType.VECTOR_ADD:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported workload type: {request.workload_type}",
            )

        if len(request.a) != len(request.b):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Vector dimension mismatch: len(a)={len(request.a)} != len(b)={len(request.b)}",
            )

        worker_state.increment_active_tasks()
        try:
            result, exec_time_ms, backend = worker_state.gpu_executor.execute_vector_add(
                a=request.a,
                b=request.b,
                device_index=request.device_index,
            )
            return ExecuteResponse(
                task_id=task_id,
                workload_type=request.workload_type.value,
                status="success",
                result=result,
                result_length=len(result),
                execution_time_ms=round(exec_time_ms, 4),
                device_index=request.device_index,
                gpu_backend=backend,
            )
        except GPUExecutionError as e:
            logger.error(f"Task {task_id} failed on GPU: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e),
            )
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Workload execution failed: {e}",
            )
        finally:
            worker_state.decrement_active_tasks()

    return app


# Default app instance for standard uvicorn worker.app:app runs
app = create_app()
