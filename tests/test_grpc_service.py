"""Tests for GPUFabricServicer."""

from unittest.mock import MagicMock, patch

import grpc
import numpy as np
import pytest

from common.gpufabric_pb2 import (
    ComputeRequest,
    DType,
    ExecuteRequest,
    GPUInfoRequest,
    HealthRequest,
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
