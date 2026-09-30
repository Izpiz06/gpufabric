"""Data models for GPU Fabric communication."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class WorkloadType(str, Enum):
    VECTOR_ADD = "vector_add"


class HealthResponse(BaseModel):
    """Response returned by GET /health."""

    status: str = Field(default="ok", description="Status of the worker service")
    worker_id: str = Field(..., description="Unique identifier for the worker")
    version: str = Field(..., description="GPU Fabric version")
    gpu_available: bool = Field(..., description="Whether a GPU is detected and available")


class GPUInfoResponse(BaseModel):
    """Response returned by GET /gpu."""

    device_index: int = Field(default=0, description="CUDA device index")
    name: str = Field(..., description="GPU device model name")
    total_vram_bytes: int = Field(..., description="Total VRAM in bytes")
    free_vram_bytes: int = Field(..., description="Available/Free VRAM in bytes")
    used_vram_bytes: int = Field(..., description="Used VRAM in bytes")
    total_vram_human: str = Field(..., description="Human-readable total VRAM")
    free_vram_human: str = Field(..., description="Human-readable free VRAM")
    used_vram_human: str = Field(..., description="Human-readable used VRAM")
    compute_capability: Optional[str] = Field(
        None, description="CUDA compute capability (e.g. '8.9')"
    )
    driver_version: Optional[str] = Field(None, description="NVIDIA driver version")


class GPUStatusResponse(BaseModel):
    """Response returned by GET /status."""

    device_index: int = Field(default=0, description="CUDA device index")
    gpu_utilization_pct: Optional[int] = Field(
        None, description="GPU core utilization percentage (0-100)"
    )
    memory_utilization_pct: Optional[int] = Field(
        None, description="GPU memory utilization percentage (0-100)"
    )
    total_vram_bytes: int = Field(..., description="Total VRAM in bytes")
    free_vram_bytes: int = Field(..., description="Free VRAM in bytes")
    used_vram_bytes: int = Field(..., description="Used VRAM in bytes")
    free_vram_human: str = Field(..., description="Human-readable free VRAM")
    used_vram_human: str = Field(..., description="Human-readable used VRAM")
    temperature_c: Optional[int] = Field(None, description="GPU temperature in Celsius")
    active_tasks: int = Field(default=0, description="Number of currently executing workloads")


class ExecuteRequest(BaseModel):
    """Request payload for POST /execute."""

    workload_type: WorkloadType = Field(
        default=WorkloadType.VECTOR_ADD, description="Type of GPU workload to execute"
    )
    a: List[float] = Field(..., description="Vector A (list of floats)")
    b: List[float] = Field(..., description="Vector B (list of floats)")
    device_index: int = Field(default=0, description="Target GPU device index")


class ExecuteResponse(BaseModel):
    """Response returned by POST /execute."""

    task_id: str = Field(..., description="Unique ID for the executed task")
    workload_type: str = Field(..., description="Type of workload executed")
    status: str = Field(..., description="Execution status: 'success' or 'error'")
    result: Optional[List[float]] = Field(None, description="Result vector (C = A + B)")
    result_length: Optional[int] = Field(None, description="Length of the result vector")
    execution_time_ms: float = Field(..., description="Execution duration on GPU in milliseconds")
    device_index: int = Field(default=0, description="GPU device index used")
    gpu_backend: Optional[str] = Field(
        None, description="Underlying GPU execution library (e.g. cupy/pycuda/torch)"
    )
    error: Optional[str] = Field(None, description="Error message if execution failed")
