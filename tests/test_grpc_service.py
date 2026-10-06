# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Tests for GPUFabricServicer."""

from unittest.mock import MagicMock, patch

import grpc
import numpy as np
import pytest

from common.gpufabric_pb2 import (
    Benchmark,
    BenchmarkRequest,
    ComputeRequest,
    DType,
    ExecuteRequest,
    GPUInfoRequest,
    GPUStatusRequest,
    HealthRequest,
    HealthState,
    ListGPUsRequest,
    Operation,
    Tensor,
    WorkloadType,
)
from common.tensor import from_tensor, to_tensor
from worker.executor import (
    CUDALibraryNotFoundError,
    GPUExecutionError,
    GPUExecutor,
    GPUOutOfMemoryError,
)
from worker.gpu import GPUManager
from worker.service import GPUFabricServicer
from worker.state import WorkerState


def test_service_health():
    state = WorkerState("worker-node-1")
    gpu = GPUManager()
    executor = GPUExecutor()
    servicer = GPUFabricServicer(state, gpu, executor)

    mock_context = MagicMock()
    resp = servicer.GetHealth(HealthRequest(), mock_context)
    assert resp.status == "ok"
    assert resp.worker_id == "worker-node-1"
    assert resp.hostname == state.hostname
    assert resp.last_heartbeat > 0


def test_service_health_healthy():
    state = WorkerState("worker-node-1")
    gpu = GPUManager()
    executor = GPUExecutor()
    executor.diagnostics.cublas_available = True
    servicer = GPUFabricServicer(state, gpu, executor)

    with (
        patch.object(gpu, "is_available", return_value=True),
        patch.object(gpu, "device_count", return_value=1),
        patch.object(gpu, "get_status", return_value={"temp": 55}),
        patch.object(gpu, "get_cuda_version", return_value="12.2"),
        patch.object(gpu, "get_driver_version", return_value="535.104"),
        patch.object(executor, "is_gpu_ready", return_value=True),
    ):
        mock_context = MagicMock()
        resp = servicer.GetHealth(HealthRequest(), mock_context)
        assert resp.status == "ok"
        assert resp.worker_id == "worker-node-1"
        assert resp.health_state == HealthState.HEALTHY
        assert resp.gpu_available is True
        assert resp.cuda_version == "12.2"
        assert resp.driver_version == "535.104"
        assert resp.hostname == state.hostname
        assert resp.last_heartbeat > 0
        assert resp.active_tasks == 0
        assert resp.gpu_count == 1
        assert "operational" in resp.health_details


def test_service_health_degraded_cublas_missing():
    state = WorkerState("worker-node-1")
    gpu = GPUManager()
    executor = GPUExecutor()
    executor.diagnostics.cublas_available = False
    executor.diagnostics.missing_libraries = ["libcublas.so.12"]
    servicer = GPUFabricServicer(state, gpu, executor)

    with (
        patch.object(gpu, "is_available", return_value=True),
        patch.object(gpu, "device_count", return_value=1),
        patch.object(executor, "is_gpu_ready", return_value=True),
        patch.object(executor, "refresh_diagnostics", return_value=executor.diagnostics),
    ):
        mock_context = MagicMock()
        resp = servicer.GetHealth(HealthRequest(), mock_context)
        assert resp.health_state == HealthState.DEGRADED
        assert "cuBLAS library missing" in resp.health_details
        assert "libcublas.so.12" in resp.health_details


def test_service_health_degraded_cublas_missing_and_high_temp_combined():
    state = WorkerState("worker-node-1")
    gpu = GPUManager()
    executor = GPUExecutor()
    executor.diagnostics.cublas_available = False
    executor.diagnostics.missing_libraries = ["libcublas.so.12"]
    servicer = GPUFabricServicer(state, gpu, executor)

    with (
        patch.object(gpu, "is_available", return_value=True),
        patch.object(gpu, "device_count", return_value=1),
        patch.object(gpu, "get_status", return_value={"temp": 92}),
        patch.object(executor, "is_gpu_ready", return_value=True),
        patch.object(executor, "refresh_diagnostics", return_value=executor.diagnostics),
    ):
        mock_context = MagicMock()
        resp = servicer.GetHealth(HealthRequest(), mock_context)
        assert resp.health_state == HealthState.DEGRADED
        assert "cuBLAS library missing" in resp.health_details
        assert "libcublas.so.12" in resp.health_details
        assert "GPU 0 temperature high (92°C)" in resp.health_details


def test_service_health_degraded_cupy_missing():
    state = WorkerState("worker-node-1")
    gpu = GPUManager()
    executor = GPUExecutor()
    servicer = GPUFabricServicer(state, gpu, executor)

    with (
        patch.object(gpu, "is_available", return_value=True),
        patch.object(gpu, "device_count", return_value=1),
        patch.object(executor, "is_gpu_ready", return_value=False),
    ):
        mock_context = MagicMock()
        resp = servicer.GetHealth(HealthRequest(), mock_context)
        assert resp.health_state == HealthState.DEGRADED
        assert "CuPy" in resp.health_details


def test_service_health_degraded_high_temp():
    state = WorkerState("worker-node-1")
    gpu = GPUManager()
    executor = GPUExecutor()
    executor.diagnostics.cublas_available = True
    servicer = GPUFabricServicer(state, gpu, executor)

    with (
        patch.object(gpu, "is_available", return_value=True),
        patch.object(gpu, "device_count", return_value=1),
        patch.object(gpu, "get_status", return_value={"temp": 90}),
        patch.object(executor, "is_gpu_ready", return_value=True),
    ):
        mock_context = MagicMock()
        resp = servicer.GetHealth(HealthRequest(), mock_context)
        assert resp.health_state == HealthState.DEGRADED
        assert "temperature high" in resp.health_details


def test_service_health_unavailable():
    state = WorkerState("worker-node-1")
    gpu = GPUManager()
    executor = GPUExecutor()
    servicer = GPUFabricServicer(state, gpu, executor)

    with (
        patch.object(gpu, "is_available", return_value=False),
        patch.object(executor, "is_gpu_ready", return_value=False),
    ):
        mock_context = MagicMock()
        resp = servicer.GetHealth(HealthRequest(), mock_context)
        assert resp.health_state == HealthState.UNAVAILABLE
        assert resp.gpu_available is False


def test_service_gpu_status_telemetry():
    state = WorkerState("worker-node-1")
    gpu = GPUManager()
    executor = GPUExecutor()
    servicer = GPUFabricServicer(state, gpu, executor)

    mock_status = {
        "device_index": 0,
        "gpu_util": 45,
        "mem_util": 30,
        "total_vram": 8000000000,
        "free_vram": 5000000000,
        "used_vram": 3000000000,
        "temp": 62,
        "power_usage_w": 95,
        "power_limit_w": 220,
        "available": True,
    }
    with (
        patch.object(gpu, "is_available", return_value=True),
        patch.object(gpu, "get_status", return_value=mock_status),
    ):
        mock_context = MagicMock()
        resp = servicer.GetGPUStatus(GPUStatusRequest(device_index=0), mock_context)
        assert resp.device_index == 0
        assert resp.gpu_utilization_pct == 45
        assert resp.temperature_c == 62
        assert resp.power_usage_w == 95
        assert resp.power_limit_w == 220
        assert resp.available is True
        assert resp.current_workload == "IDLE"


def test_worker_state_workload_and_heartbeat():
    state = WorkerState("worker-1")
    assert state.get_device_workload(0) == "IDLE"
    hb1 = state.last_heartbeat

    state.increment_tasks(0, "Compute:matmul")
    assert state.active_tasks == 1
    assert state.get_device_workload(0) == "Compute:matmul"
    assert state.last_heartbeat >= hb1

    state.decrement_tasks(0)
    assert state.active_tasks == 0
    assert state.get_device_workload(0) == "IDLE"


def test_service_gpu_info():
    state = WorkerState("worker-node-1")
    gpu = GPUManager()
    executor = GPUExecutor()
    servicer = GPUFabricServicer(state, gpu, executor)

    mock_info = {
        "device_index": 0,
        "name": "NVIDIA RTX 3080",
        "total_vram": 10000000000,
        "free_vram": 8000000000,
        "used_vram": 2000000000,
        "compute_capability": "8.6",
        "driver_version": "535.0",
    }
    with (
        patch.object(gpu, "is_available", return_value=True),
        patch.object(gpu, "get_info", return_value=mock_info),
    ):
        mock_context = MagicMock()
        resp = servicer.GetGPUInfo(GPUInfoRequest(device_index=0), mock_context)
        assert resp.name == "NVIDIA RTX 3080"
        assert resp.compute_capability == "8.6"


def test_service_execute_vector_add():
    state = WorkerState("worker-node-1")
    gpu = GPUManager()
    executor = GPUExecutor()
    servicer = GPUFabricServicer(state, gpu, executor)

    with patch.object(executor, "execute_vector_add", return_value=([5.0, 7.0], 0.25, "cupy")):
        req = ExecuteRequest(
            workload_type=WorkloadType.VECTOR_ADD,
            a=[1.0, 2.0],
            b=[4.0, 5.0],
            device_index=0,
        )
        mock_context = MagicMock()
        resp = servicer.Execute(req, mock_context)
        assert resp.status == "success"
        assert list(resp.result) == [5.0, 7.0]
        assert resp.gpu_backend == "cupy"


class _Aborted(Exception):
    def __init__(self, code, details):
        super().__init__(details)
        self.code = code


def _context():
    # Real gRPC contexts raise on abort(); a bare MagicMock would not.
    ctx = MagicMock()
    ctx.abort.side_effect = lambda code, details: (_ for _ in ()).throw(_Aborted(code, details))
    return ctx


def _servicer():
    return GPUFabricServicer(WorkerState("worker-node-1"), GPUManager(), GPUExecutor())


def _vector_request(op=Operation.OP_VECTOR_ADD):
    a = np.array([1.0, 2.0], dtype=np.float32)
    return ComputeRequest(op=op, inputs=[to_tensor(a), to_tensor(a)])


def test_service_compute_success():
    servicer = _servicer()
    with patch.object(
        servicer.executor,
        "compute",
        return_value=(np.array([2.0, 4.0], dtype=np.float32), 0.05, 0.5),
    ) as compute:
        resp = servicer.Compute(_vector_request(), _context())
    assert compute.call_args.args[0] == "vector_add"
    np.testing.assert_array_equal(from_tensor(resp.result), [2.0, 4.0])
    assert resp.gpu_time_ms == 0.05
    assert resp.total_time_ms == 0.5


def test_service_compute_unsupported_operation():
    with pytest.raises(_Aborted) as exc:
        _servicer().Compute(_vector_request(op=Operation.OPERATION_UNSPECIFIED), _context())
    assert exc.value.code == grpc.StatusCode.INVALID_ARGUMENT


def test_service_compute_malformed_tensor():
    bad = Tensor(dtype=DType.FLOAT32, shape=[4], data=b"\x00")
    request = ComputeRequest(op=Operation.OP_VECTOR_ADD, inputs=[bad, bad])
    with pytest.raises(_Aborted) as exc:
        _servicer().Compute(request, _context())
    assert exc.value.code == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.parametrize(
    "error, code",
    [
        (ValueError("bad shape"), grpc.StatusCode.INVALID_ARGUMENT),
        (GPUOutOfMemoryError("oom"), grpc.StatusCode.RESOURCE_EXHAUSTED),
        (GPUExecutionError("no gpu"), grpc.StatusCode.UNAVAILABLE),
        (CUDALibraryNotFoundError("libcublas.so missing"), grpc.StatusCode.UNAVAILABLE),
        (RuntimeError("driver"), grpc.StatusCode.INTERNAL),
    ],
)
def test_service_compute_error_mapping(error, code):
    servicer = _servicer()
    with patch.object(servicer.executor, "compute", side_effect=error):
        with pytest.raises(_Aborted) as exc:
            servicer.Compute(_vector_request(), _context())
    assert exc.value.code == code
    assert servicer.state.active_tasks == 0


def test_service_list_gpus():
    servicer = _servicer()
    device = {
        "device_index": 0,
        "name": "NVIDIA RTX 3050",
        "total_vram": 4 * 1024**3,
        "free_vram": 3 * 1024**3,
        "used_vram": 1024**3,
        "compute_capability": "8.6",
        "driver_version": "610.57.04",
        "gpu_util": 5,
        "mem_util": 2,
        "temp": 52,
    }
    with (
        patch.object(servicer.gpu, "is_available", return_value=True),
        patch.object(servicer.gpu, "list_devices", return_value=[device]),
        patch.object(servicer.executor, "is_gpu_ready", return_value=False),
    ):
        resp = servicer.ListGPUs(ListGPUsRequest(), _context())
    assert resp.worker_id == "worker-node-1"
    assert resp.driver_version == "610.57.04"
    assert resp.compute_ready is False
    assert resp.health_state == HealthState.DEGRADED
    assert resp.hostname == servicer.state.hostname
    assert resp.last_heartbeat > 0
    assert len(resp.gpus) == 1
    assert resp.gpus[0].name == "NVIDIA RTX 3050"
    assert resp.gpus[0].free_vram_bytes == 3 * 1024**3
    assert resp.gpus[0].available is True
    assert resp.gpus[0].current_workload == "IDLE"


def test_service_list_gpus_without_gpus():
    servicer = _servicer()
    with patch.object(servicer.gpu, "list_devices", return_value=[]):
        resp = servicer.ListGPUs(ListGPUsRequest(), _context())
    assert list(resp.gpus) == []
    assert resp.driver_version == ""


def test_service_run_benchmark_triad():
    servicer = _servicer()
    result = {"best_ms": 3.3, "mean_ms": 3.4, "bandwidth_gb_s": 180.6}
    with (
        patch.object(servicer.executor, "run_benchmark", return_value=result) as run,
        patch.object(servicer.gpu, "peak_memory_bandwidth", return_value=187.9),
    ):
        resp = servicer.RunBenchmark(
            BenchmarkRequest(benchmark=Benchmark.BENCH_TRIAD, size=1000), _context()
        )
    assert run.call_args.args == ("triad", 1000, 0, 5)
    assert resp.repeats == 5
    assert resp.bandwidth_gb_s == 180.6
    assert resp.peak_bandwidth_gb_s == 187.9


def test_service_run_benchmark_unsupported():
    with pytest.raises(_Aborted) as exc:
        _servicer().RunBenchmark(BenchmarkRequest(size=10), _context())
    assert exc.value.code == grpc.StatusCode.INVALID_ARGUMENT


def test_service_run_benchmark_out_of_memory():
    servicer = _servicer()
    with patch.object(servicer.executor, "run_benchmark", side_effect=GPUOutOfMemoryError("oom")):
        with pytest.raises(_Aborted) as exc:
            servicer.RunBenchmark(
                BenchmarkRequest(benchmark=Benchmark.BENCH_MATMUL, size=10**6), _context()
            )
    assert exc.value.code == grpc.StatusCode.RESOURCE_EXHAUSTED
    assert servicer.state.active_tasks == 0


def test_state_multi_device_task_tracking():
    from worker.state import WorkerState

    state = WorkerState("test-worker")
    state.increment_tasks(device_index=0, workload="vector_add")
    state.increment_tasks(device_index=1, workload="matmul")
    assert state.active_tasks == 2
    assert state.get_device_workload(0) == "vector_add"
    assert state.get_device_workload(1) == "matmul"

    # Device 0 finishes first
    state.decrement_tasks(device_index=0)
    assert state.active_tasks == 1
    assert state.get_device_workload(0) == "IDLE"
    assert state.get_device_workload(1) == "matmul"

    # Device 1 finishes
    state.decrement_tasks(device_index=1)
    assert state.active_tasks == 0
    assert state.get_device_workload(0) == "IDLE"
    assert state.get_device_workload(1) == "IDLE"


def test_service_compute_matmul_missing_cublas_returns_unavailable():
    servicer = _servicer()
    servicer.executor.device_count = 1
    servicer.executor.diagnostics.cublas_available = False
    servicer.executor.diagnostics.missing_libraries = ["libcublas.so.12"]

    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    req = ComputeRequest(op=Operation.OP_MATMUL, inputs=[to_tensor(a), to_tensor(a)])

    with (
        patch.object(servicer.executor, "is_gpu_ready", return_value=True),
        patch.object(
            servicer.executor, "refresh_diagnostics", return_value=servicer.executor.diagnostics
        ),
    ):
        with pytest.raises(_Aborted) as exc:
            servicer.Compute(req, _context())
        assert exc.value.code == grpc.StatusCode.UNAVAILABLE
        assert "libcublas.so.12" in str(exc.value)


def test_service_run_benchmark_matmul_missing_cublas_returns_unavailable():
    servicer = _servicer()
    servicer.executor.device_count = 1
    servicer.executor.diagnostics.cublas_available = False
    servicer.executor.diagnostics.missing_libraries = ["libcublas.so.12"]

    req = BenchmarkRequest(benchmark=Benchmark.BENCH_MATMUL, size=100)

    with (
        patch.object(servicer.executor, "is_gpu_ready", return_value=True),
        patch.object(
            servicer.executor, "refresh_diagnostics", return_value=servicer.executor.diagnostics
        ),
    ):
        with pytest.raises(_Aborted) as exc:
            servicer.RunBenchmark(req, _context())
        assert exc.value.code == grpc.StatusCode.UNAVAILABLE
        assert "libcublas.so.12" in str(exc.value)
