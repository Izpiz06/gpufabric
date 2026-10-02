from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class WorkloadType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WORKLOAD_UNSPECIFIED: _ClassVar[WorkloadType]
    VECTOR_ADD: _ClassVar[WorkloadType]
WORKLOAD_UNSPECIFIED: WorkloadType
VECTOR_ADD: WorkloadType

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("status", "worker_id", "version", "gpu_available")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    GPU_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    status: str
    worker_id: str
    version: str
    gpu_available: bool
    def __init__(self, status: _Optional[str] = ..., worker_id: _Optional[str] = ..., version: _Optional[str] = ..., gpu_available: _Optional[bool] = ...) -> None: ...

class GPUInfoRequest(_message.Message):
    __slots__ = ("device_index",)
    DEVICE_INDEX_FIELD_NUMBER: _ClassVar[int]
    device_index: int
    def __init__(self, device_index: _Optional[int] = ...) -> None: ...

class GPUInfoResponse(_message.Message):
    __slots__ = ("device_index", "name", "total_vram_bytes", "free_vram_bytes", "used_vram_bytes", "total_vram_human", "free_vram_human", "used_vram_human", "compute_capability", "driver_version")
    DEVICE_INDEX_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TOTAL_VRAM_BYTES_FIELD_NUMBER: _ClassVar[int]
    FREE_VRAM_BYTES_FIELD_NUMBER: _ClassVar[int]
    USED_VRAM_BYTES_FIELD_NUMBER: _ClassVar[int]
    TOTAL_VRAM_HUMAN_FIELD_NUMBER: _ClassVar[int]
    FREE_VRAM_HUMAN_FIELD_NUMBER: _ClassVar[int]
    USED_VRAM_HUMAN_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_CAPABILITY_FIELD_NUMBER: _ClassVar[int]
    DRIVER_VERSION_FIELD_NUMBER: _ClassVar[int]
    device_index: int
    name: str
    total_vram_bytes: int
    free_vram_bytes: int
    used_vram_bytes: int
    total_vram_human: str
    free_vram_human: str
    used_vram_human: str
    compute_capability: str
    driver_version: str
    def __init__(self, device_index: _Optional[int] = ..., name: _Optional[str] = ..., total_vram_bytes: _Optional[int] = ..., free_vram_bytes: _Optional[int] = ..., used_vram_bytes: _Optional[int] = ..., total_vram_human: _Optional[str] = ..., free_vram_human: _Optional[str] = ..., used_vram_human: _Optional[str] = ..., compute_capability: _Optional[str] = ..., driver_version: _Optional[str] = ...) -> None: ...

class GPUStatusRequest(_message.Message):
    __slots__ = ("device_index",)
    DEVICE_INDEX_FIELD_NUMBER: _ClassVar[int]
    device_index: int
    def __init__(self, device_index: _Optional[int] = ...) -> None: ...

class GPUStatusResponse(_message.Message):
    __slots__ = ("device_index", "gpu_utilization_pct", "memory_utilization_pct", "total_vram_bytes", "free_vram_bytes", "used_vram_bytes", "free_vram_human", "used_vram_human", "temperature_c", "active_tasks")
    DEVICE_INDEX_FIELD_NUMBER: _ClassVar[int]
    GPU_UTILIZATION_PCT_FIELD_NUMBER: _ClassVar[int]
    MEMORY_UTILIZATION_PCT_FIELD_NUMBER: _ClassVar[int]
    TOTAL_VRAM_BYTES_FIELD_NUMBER: _ClassVar[int]
    FREE_VRAM_BYTES_FIELD_NUMBER: _ClassVar[int]
    USED_VRAM_BYTES_FIELD_NUMBER: _ClassVar[int]
    FREE_VRAM_HUMAN_FIELD_NUMBER: _ClassVar[int]
    USED_VRAM_HUMAN_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_C_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_TASKS_FIELD_NUMBER: _ClassVar[int]
    device_index: int
    gpu_utilization_pct: int
    memory_utilization_pct: int
    total_vram_bytes: int
    free_vram_bytes: int
    used_vram_bytes: int
    free_vram_human: str
    used_vram_human: str
    temperature_c: int
    active_tasks: int
    def __init__(self, device_index: _Optional[int] = ..., gpu_utilization_pct: _Optional[int] = ..., memory_utilization_pct: _Optional[int] = ..., total_vram_bytes: _Optional[int] = ..., free_vram_bytes: _Optional[int] = ..., used_vram_bytes: _Optional[int] = ..., free_vram_human: _Optional[str] = ..., used_vram_human: _Optional[str] = ..., temperature_c: _Optional[int] = ..., active_tasks: _Optional[int] = ...) -> None: ...

class ExecuteRequest(_message.Message):
    __slots__ = ("workload_type", "a", "b", "device_index")
    WORKLOAD_TYPE_FIELD_NUMBER: _ClassVar[int]
    A_FIELD_NUMBER: _ClassVar[int]
    B_FIELD_NUMBER: _ClassVar[int]
    DEVICE_INDEX_FIELD_NUMBER: _ClassVar[int]
    workload_type: WorkloadType
    a: _containers.RepeatedScalarFieldContainer[float]
    b: _containers.RepeatedScalarFieldContainer[float]
    device_index: int
    def __init__(self, workload_type: _Optional[_Union[WorkloadType, str]] = ..., a: _Optional[_Iterable[float]] = ..., b: _Optional[_Iterable[float]] = ..., device_index: _Optional[int] = ...) -> None: ...

class ExecuteResponse(_message.Message):
    __slots__ = ("task_id", "workload_type", "status", "result", "result_length", "execution_time_ms", "device_index", "gpu_backend", "error")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    WORKLOAD_TYPE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    RESULT_LENGTH_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    DEVICE_INDEX_FIELD_NUMBER: _ClassVar[int]
    GPU_BACKEND_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    workload_type: str
    status: str
    result: _containers.RepeatedScalarFieldContainer[float]
    result_length: int
    execution_time_ms: float
    device_index: int
    gpu_backend: str
    error: str
    def __init__(self, task_id: _Optional[str] = ..., workload_type: _Optional[str] = ..., status: _Optional[str] = ..., result: _Optional[_Iterable[float]] = ..., result_length: _Optional[int] = ..., execution_time_ms: _Optional[float] = ..., device_index: _Optional[int] = ..., gpu_backend: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...
