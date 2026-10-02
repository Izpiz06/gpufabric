"""End-to-end tests against a real worker GPU over gRPC.

These run only when GPUFABRIC_WORKER points at a running worker, e.g.

    GPUFABRIC_WORKER=192.168.1.50:50051 pytest -m remote_gpu -v

CI has no GPU, so they are skipped there. Each test sends data from this
machine, the worker computes on its GPU, and the result is checked
against numpy here. A pass shows the remote GPU was used over the network.
"""

import os

import numpy as np
import pytest

from client.client import GPUFabricClient, GPUFabricError
from client.commands import compute_input_shapes
from client.verify import REFERENCE, relative_error, tolerance

WORKER = os.environ.get("GPUFABRIC_WORKER")
DEVICE = int(os.environ.get("GPUFABRIC_DEVICE", "0"))

pytestmark = [
    pytest.mark.remote_gpu,
    pytest.mark.skipif(not WORKER, reason="set GPUFABRIC_WORKER=host:port to run remote GPU tests"),
]

OPS = ["vector_add", "vector_mul", "vector_dot", "matrix_add", "matmul"]
# (vector length, matrix size) pairs: small, then large enough to exceed
# gRPC's old 4 MiB default (1M float32 vector = 4 MB, 1000x1000 = 4 MB).
SIZES = {"small": (1_000, 64), "large": (1_000_000, 1_000)}


@pytest.fixture(scope="module")
def client():
    with GPUFabricClient(host=WORKER, timeout=120.0) as c:
        yield c


def test_worker_reports_gpu(client):
    health = client.health()
    assert health.status == "ok"
    assert health.gpu_available


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scale", SIZES)
@pytest.mark.parametrize("op", OPS)
def test_compute_matches_numpy(client, op, scale, dtype):
    vec_size, mat_size = SIZES[scale]
    size = vec_size if op.startswith("vector") else mat_size
    rng = np.random.default_rng(42)
    inputs = [rng.random(shape, dtype=dtype) for shape in compute_input_shapes(op, size)]

    res = client.compute(op, *inputs, device_index=DEVICE)

    expected = REFERENCE[op](*inputs)
    assert res.result.shape == np.shape(expected)
    assert res.result.dtype == np.dtype(dtype)
    assert relative_error(res.result, expected) <= tolerance(op, dtype)
    assert res.task_id
    assert res.gpu_time_ms > 0
    assert res.total_time_ms >= res.gpu_time_ms


def test_invalid_device_is_rejected(client):
    a = np.ones(4, dtype=np.float32)
    with pytest.raises(GPUFabricError, match="Invalid device_index"):
        client.vector_add(a, a, device_index=999)
