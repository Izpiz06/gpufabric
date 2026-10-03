"""Benchmark checks: run benchmarks on a worker and judge them pass/fail."""
import math
import statistics
from dataclasses import dataclass
from typing import List, Optional, Tuple

import grpc
import numpy as np

from client.client import GPUFabricClient, GPUFabricError
from client.verify import REFERENCE, relative_error, tolerance

DEFAULT_MIN_NETWORK_MB_S = 2.0  # effective end-to-end throughput, 1 MB = 1e6 bytes
DEFAULT_MIN_GPU_BW_PCT = 50.0  # triad bandwidth as a % of the NVML theoretical peak
DEFAULT_MAX_TRIAD_SIZE = 50_000_000
# Std dev of one Monte Carlo sample of 4 * [point inside circle]. The estimate
# over N points has std dev PI_SAMPLE_STD / sqrt(N).
PI_SAMPLE_STD = 4 * math.sqrt(math.pi / 4 * (1 - math.pi / 4))
PI_SIGMAS = 5
TRIAD_BYTES_PER_ELEMENT = 3 * 4  # read b, read c, write a; float32


@dataclass
class Check:
    name: str
    value: str
    threshold: str
    status: str  # PASS, FAIL, INFO or SKIP


def auto_triad_size(free_vram_bytes: int) -> int:
    """Triad size that uses at most a quarter of free VRAM, capped at 50M elements."""
    fit = int(free_vram_bytes * 0.25) // TRIAD_BYTES_PER_ELEMENT
    return max(1, min(DEFAULT_MAX_TRIAD_SIZE, fit))


def check_gpu_bandwidth(bandwidth_gb_s: float, peak_gb_s: float, min_pct: float) -> Check:
    value = f"{bandwidth_gb_s:.1f} GB/s"
    if peak_gb_s <= 0:
        return Check("GPU memory bandwidth (triad)", value, "peak unknown", "SKIP")
    pct = 100.0 * bandwidth_gb_s / peak_gb_s
    return Check(
        "GPU memory bandwidth (triad)",
        f"{value} ({pct:.0f}% of {peak_gb_s:.1f} GB/s peak)",
        f">= {min_pct:.0f}% of peak",
        "PASS" if pct >= min_pct else "FAIL",
    )


def check_network(median_mb_s: float, min_mb_s: float, verified: bool) -> Check:
    if not verified:
        return Check("Network triad (end-to-end)", "wrong result", f">= {min_mb_s} MB/s", "FAIL")
    return Check(
        "Network triad (end-to-end)",
        f"{median_mb_s:.1f} MB/s (median)",
        f">= {min_mb_s} MB/s",
        "PASS" if median_mb_s >= min_mb_s else "FAIL",
    )


def check_pi(estimate: float, samples: int) -> Check:
    limit = PI_SIGMAS * PI_SAMPLE_STD / math.sqrt(samples)
    error = abs(estimate - math.pi)
    return Check(
        "Monte Carlo pi",
        f"{estimate:.6f} (error {error:.1e}, {samples:,} points)",
        f"error <= {limit:.1e} ({PI_SIGMAS} sigma)",
        "PASS" if error <= limit else "FAIL",
    )


def matmul_info(gflops: float, size: int) -> Check:
    return Check(f"Matmul {size}x{size} float32", f"{gflops:,.0f} GFLOPS", "-", "INFO")


def measure_network(
    client: GPUFabricClient, size: int, repeats: int, device_index: int = 0
) -> Tuple[float, float, bool]:
    """End-to-end triad: send b and c, get a back, over the real network.

    Returns (median MB/s, best MB/s, result verified). Throughput counts the
    bytes sent plus received divided by the client-side round-trip time,
    so it reflects what a client actually gets.
    """
    rng = np.random.default_rng(0)
    b = rng.random(size, dtype=np.float32)
    c = rng.random(size, dtype=np.float32)
    s = np.float32(3.0)
    client.triad(b[:16], c[:16], s, device_index=device_index)  # untimed: connect + warm
    rates, verified = [], True
    for _ in range(repeats):
        res = client.triad(b, c, s, device_index=device_index)
        moved = b.nbytes + c.nbytes + s.nbytes + res.result.nbytes
        rates.append(moved / (res.round_trip_ms / 1000.0) / 1e6)
        err = relative_error(res.result, REFERENCE["triad"](b, c, s))
        verified = verified and err <= tolerance("triad", np.float32)
    return statistics.median(rates), max(rates), verified


def sweep_triad(
    client: GPUFabricClient,
    free_vram_bytes: int,
    device_index: int = 0,
    repeats: int = 3,
    start: int = 1_000_000,
) -> List[Tuple[int, Optional[float], str]]:
    """Double the triad size until it no longer fits in GPU memory.

    Stops at 90% of free VRAM or the first out-of-memory error. Returns
    (size, bandwidth GB/s or None, note) rows.
    """
    rows = []
    n = start
    while n * TRIAD_BYTES_PER_ELEMENT <= 0.9 * free_vram_bytes:
        try:
            r = client.run_benchmark("triad", n, repeats=repeats, device_index=device_index)
            rows.append((n, r.bandwidth_gb_s, "ok"))
        except GPUFabricError as e:
            if e.code == grpc.StatusCode.RESOURCE_EXHAUSTED:
                rows.append((n, None, "out of GPU memory"))
                break
            raise
        n *= 2
    return rows
