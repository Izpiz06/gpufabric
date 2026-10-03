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
    HealthRequest,
    ListGPUsRequest,
    Operation,
    Tensor,
    WorkloadType,
)
from common.tensor import from_tensor, to_tensor
from worker.executor import GPUExecutionError, GPUExecutor, GPUOutOfMemoryError
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
        patch.object(servicer.gpu, "list_devices", return_value=[device]),
        patch.object(servicer.executor, "is_gpu_ready", return_value=False),
    ):
        resp = servicer.ListGPUs(ListGPUsRequest(), _context())
    assert resp.worker_id == "worker-node-1"
    assert resp.driver_version == "610.57.04"
    assert resp.compute_ready is False
    assert len(resp.gpus) == 1
    assert resp.gpus[0].name == "NVIDIA RTX 3050"
    assert resp.gpus[0].free_vram_bytes == 3 * 1024**3


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
