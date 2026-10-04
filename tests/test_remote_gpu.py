# Copyright (c) 2026 Shreyas Mene
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests against a real worker GPU over gRPC.

These run only when GPUFABRIC_WORKER points at a running worker, e.g.

    GPUFABRIC_WORKER=192.168.1.50:50051 pytest -m remote_gpu -v

Certificates are read from GPUFABRIC_TLS_DIR or ~/.config/gpufabric/tls.
Set GPUFABRIC_INSECURE=1 for a worker started with --insecure.

CI has no GPU, so they are skipped there. Each test sends data from this
machine, the worker computes on its GPU, and the result is checked
against numpy here. A pass shows the remote GPU was used over the network.
"""

import os

import grpc
import numpy as np
import pytest

from client import bench
from client.client import GPUFabricClient, GPUFabricError
from client.commands import compute_input_shapes
from client.verify import REFERENCE, relative_error, tolerance

WORKER = os.environ.get("GPUFABRIC_WORKER")
DEVICE = int(os.environ.get("GPUFABRIC_DEVICE", "0"))
# mTLS certificates come from GPUFABRIC_TLS_DIR (or the default directory).
# Set GPUFABRIC_INSECURE=1 to test a worker started with --insecure.
INSECURE = os.environ.get("GPUFABRIC_INSECURE") == "1"

pytestmark = [
    pytest.mark.remote_gpu,
    pytest.mark.skipif(not WORKER, reason="set GPUFABRIC_WORKER=host:port to run remote GPU tests"),
]

OPS = ["vector_add", "vector_mul", "vector_dot", "matrix_add", "matmul", "triad"]
# (vector length, matrix size) pairs: small, then large enough to exceed
# gRPC's old 4 MiB default (1M float32 vector = 4 MB, 1000x1000 = 4 MB).
SIZES = {"small": (1_000, 64), "large": (1_000_000, 1_000)}


@pytest.fixture(scope="module")
def client():
    with GPUFabricClient(host=WORKER, timeout=120.0, insecure=INSECURE) as c:
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
    size = mat_size if op.startswith("mat") else vec_size
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


def test_list_gpus_shows_device(client):
    inv = client.list_gpus()
    assert inv.compute_ready
    assert DEVICE in [g.device_index for g in inv.gpus]
    gpu = next(g for g in inv.gpus if g.device_index == DEVICE)
    assert gpu.name
    assert 0 < gpu.free_vram_bytes <= gpu.total_vram_bytes


def test_triad_operation_matches_numpy(client):
    rng = np.random.default_rng(1)
    b, c = rng.random(1_000_000, dtype=np.float32), rng.random(1_000_000, dtype=np.float32)
    res = client.triad(b, c, 3.0, device_index=DEVICE)
    assert relative_error(res.result, b + np.float32(3.0) * c) <= tolerance("triad", np.float32)


def test_gpu_triad_bandwidth_is_plausible(client):
    r = client.run_benchmark("triad", 10_000_000, repeats=3, device_index=DEVICE)
    assert r.bandwidth_gb_s > 0
    if r.peak_bandwidth_gb_s:
        # Measured bandwidth cannot meaningfully exceed the theoretical peak.
        assert r.bandwidth_gb_s <= 1.05 * r.peak_bandwidth_gb_s


def test_matmul_benchmark_reports_gflops(client):
    r = client.run_benchmark("matmul", 1000, repeats=3, device_index=DEVICE)
    assert r.gflops > 0
    assert r.best_ms <= r.mean_ms


def test_monte_carlo_pi_within_five_sigma(client):
    r = client.run_benchmark("monte_carlo_pi", 10_000_000, repeats=2, device_index=DEVICE)
    assert bench.check_pi(r.pi_estimate, 10_000_000 * r.repeats).status == "PASS"


def test_network_throughput_meets_default_threshold(client):
    median, _, verified = bench.measure_network(client, 1_000_000, repeats=3, device_index=DEVICE)
    assert verified
    assert median >= bench.DEFAULT_MIN_NETWORK_MB_S


def test_oversized_benchmark_reports_out_of_memory(client):
    inv = client.list_gpus()
    gpu = next(g for g in inv.gpus if g.device_index == DEVICE)
    too_big = gpu.total_vram_bytes // 12 + 1  # three float32 arrays won't fit
    with pytest.raises(GPUFabricError) as exc:
        client.run_benchmark("triad", too_big, repeats=1, device_index=DEVICE)
    assert exc.value.code == grpc.StatusCode.RESOURCE_EXHAUSTED
