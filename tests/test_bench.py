# Copyright (c) 2026 Shreyas Mene
# SPDX-License-Identifier: Apache-2.0

"""Tests for benchmark pass/fail logic and the network/sweep helpers."""

import math
from unittest.mock import MagicMock

import grpc
import numpy as np
import pytest

from client import bench
from client.client import ComputeResult, GPUFabricError


def test_auto_triad_size():
    assert bench.auto_triad_size(4 * 1024**3) == bench.DEFAULT_MAX_TRIAD_SIZE
    # 120 MB free -> a quarter (30 MB) / 12 bytes per element
    assert bench.auto_triad_size(120_000_000) == 2_500_000
    assert bench.auto_triad_size(0) == 1


def test_check_gpu_bandwidth():
    assert bench.check_gpu_bandwidth(180.6, 187.9, 50).status == "PASS"
    assert bench.check_gpu_bandwidth(80.0, 187.9, 50).status == "FAIL"
    assert bench.check_gpu_bandwidth(80.0, 0.0, 50).status == "SKIP"


def test_check_network():
    assert bench.check_network(2.5, 2.0, verified=True).status == "PASS"
    assert bench.check_network(1.5, 2.0, verified=True).status == "FAIL"
    assert bench.check_network(500.0, 2.0, verified=False).status == "FAIL"


def test_check_pi_uses_five_sigma():
    n = 10**8
    limit = 5 * bench.PI_SAMPLE_STD / math.sqrt(n)
    assert bench.check_pi(math.pi + 0.9 * limit, n).status == "PASS"
    assert bench.check_pi(math.pi + 1.1 * limit, n).status == "FAIL"


def _fake_triad_client():
    client = MagicMock()

    def triad(b, c, s, device_index=0):
        return ComputeResult(b + s * c, "t", device_index, 0.1, 1.0, round_trip_ms=10.0)

    client.triad.side_effect = triad
    return client


def test_measure_network_counts_bytes_both_ways():
    median, best, verified = bench.measure_network(_fake_triad_client(), 250_000, repeats=3)
    # 3 vectors * 1 MB + 4-byte scalar in 10 ms -> ~300 MB/s
    assert median == pytest.approx(300.0004)
    assert best == median
    assert verified


def test_measure_network_detects_wrong_result():
    client = MagicMock()
    client.triad.side_effect = lambda b, c, s, device_index=0: ComputeResult(
        np.zeros_like(b), "t", 0, 0.1, 1.0, 10.0
    )
    _, _, verified = bench.measure_network(client, 1000, repeats=2)
    assert not verified


def test_sweep_stops_at_out_of_memory():
    client = MagicMock()
    ok = MagicMock(bandwidth_gb_s=170.0)
    oom = GPUFabricError("oom", code=grpc.StatusCode.RESOURCE_EXHAUSTED)
    client.run_benchmark.side_effect = [ok, ok, oom]
    rows = bench.sweep_triad(client, free_vram_bytes=10**12, start=10)
    assert [(n, note) for n, _, note in rows] == [(10, "ok"), (20, "ok"), (40, "out of GPU memory")]


def test_sweep_stops_before_exceeding_free_vram():
    client = MagicMock()
    client.run_benchmark.return_value = MagicMock(bandwidth_gb_s=170.0)
    rows = bench.sweep_triad(client, free_vram_bytes=12 * 100, start=10)
    # 90% of free VRAM fits 90 elements: sizes 10, 20, 40, 80 run
    assert [n for n, _, _ in rows] == [10, 20, 40, 80]


def test_sweep_reraises_other_errors():
    client = MagicMock()
    client.run_benchmark.side_effect = GPUFabricError("down", code=grpc.StatusCode.UNAVAILABLE)
    with pytest.raises(GPUFabricError):
        bench.sweep_triad(client, free_vram_bytes=10**12)


def test_measure_network_returns_history():
    median, best, verified, history = bench.measure_network(
        _fake_triad_client(), 250_000, repeats=3, return_history=True
    )
    assert verified is True
    assert len(history) == 3
    assert history[0]["trial"] == 1
    assert "throughput_mb_s" in history[0]
    assert "round_trip_ms" in history[0]
    assert "network_time_ms" in history[0]


def test_generate_benchmark_plot(tmp_path):
    from client.plot import generate_benchmark_plot

    out_file = tmp_path / "test_bench_plot.png"
    net_history = [
        {"trial": 1, "throughput_mb_s": 25.0, "round_trip_ms": 40.0, "gpu_time_ms": 1.0},
        {"trial": 2, "throughput_mb_s": 30.0, "round_trip_ms": 35.0, "gpu_time_ms": 1.0},
    ]
    triad_data = [
        {"elements": 1_000_000, "vram_mb": 12.0, "bandwidth_gb_s": 120.0, "latency_ms": 0.1},
        {"elements": 2_000_000, "vram_mb": 24.0, "bandwidth_gb_s": 125.0, "latency_ms": 0.19},
    ]
    matmul_data = [
        {"size": 256, "gflops": 500.0, "latency_ms": 0.06},
        {"size": 512, "gflops": 3000.0, "latency_ms": 0.08},
    ]
    mc_data = {"samples": 100_000_000, "estimate": 3.14159, "latency_ms": 10.0}
    latencies = {
        "Network Triad": {"total_ms": 35.0, "gpu_ms": 1.0},
        "GPU Triad": {"total_ms": 0.19, "gpu_ms": 0.19},
    }

    res_path = generate_benchmark_plot(
        target="192.168.1.50:50051",
        gpu_name="NVIDIA RTX 4090",
        driver_version="550.54.14",
        cuda_version="12.4",
        network_history=net_history,
        triad_data=triad_data,
        matmul_data=matmul_data,
        monte_carlo_data=mc_data,
        workload_latencies=latencies,
        output_path=out_file,
        peak_bandwidth_gb_s=1008.0,
    )
    assert res_path is not None
    assert res_path.exists()
    assert res_path.stat().st_size > 0
    # Also verify companion SVG
    svg_file = out_file.with_suffix(".svg")
    assert svg_file.exists()

