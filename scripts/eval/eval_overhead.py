# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Benchmark and compare verification overhead vs full CPU recompute."""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from client.client import GPUFabricClient
from client.verify import REFERENCE, freivalds_check


def run_overhead_evaluation(
    sizes=(256, 512, 1024, 2048, 3072),
    repeats=5,
    dtype_str="float32",
    simulate=True,
    worker_host=None,
    output_csv="scripts/eval/data/overhead.csv",
):
    dtype = np.dtype(dtype_str)
    records = []
    client = None

    if not simulate and worker_host:
        client = GPUFabricClient(host=worker_host)

    rng = np.random.default_rng(42)

    for n in sizes:
        print(f"Measuring overhead for size n={n} ({repeats} repeats)...")
        gpu_times = []
        rtt_times = []
        full_times = []
        k1_times = []
        k3_times = []
        k5_times = []

        for _ in range(repeats):
            a = rng.uniform(-1.0, 1.0, size=(n, n)).astype(dtype)
            b = rng.uniform(-1.0, 1.0, size=(n, n)).astype(dtype)

            if simulate or client is None:
                # Simulated timing
                t0 = time.perf_counter()
                c = np.matmul(a, b)
                sim_kernel_ms = (time.perf_counter() - t0) * 1000.0
                gpu_ms = sim_kernel_ms
                rtt_ms = sim_kernel_ms + 1.5  # simulated network overhead
            else:
                t0 = time.perf_counter()
                res = client.compute("matmul", a, b)
                rtt_ms = (time.perf_counter() - t0) * 1000.0
                gpu_ms = res.gpu_time_ms
                c = res.result

            # Full CPU recompute
            t0 = time.perf_counter()
            _ = REFERENCE["matmul"](a, b)
            full_ms = (time.perf_counter() - t0) * 1000.0

            # Freivalds k=1
            t0 = time.perf_counter()
            _ = freivalds_check(a, b, c, rounds=1, rng=rng)
            f1_ms = (time.perf_counter() - t0) * 1000.0

            # Freivalds k=3
            t0 = time.perf_counter()
            _ = freivalds_check(a, b, c, rounds=3, rng=rng)
            f3_ms = (time.perf_counter() - t0) * 1000.0

            # Freivalds k=5
            t0 = time.perf_counter()
            _ = freivalds_check(a, b, c, rounds=5, rng=rng)
            f5_ms = (time.perf_counter() - t0) * 1000.0

            gpu_times.append(gpu_ms)
            rtt_times.append(rtt_ms)
            full_times.append(full_ms)
            k1_times.append(f1_ms)
            k3_times.append(f3_ms)
            k5_times.append(f5_ms)

        records.append(
            {
                "n": n,
                "dtype": dtype_str,
                "gpu_time_ms": round(float(np.mean(gpu_times)), 4),
                "round_trip_ms": round(float(np.mean(rtt_times)), 4),
                "full_recompute_ms": round(float(np.mean(full_times)), 4),
                "freivalds_k1_ms": round(float(np.mean(k1_times)), 4),
                "freivalds_k3_ms": round(float(np.mean(k3_times)), 4),
                "freivalds_k5_ms": round(float(np.mean(k5_times)), 4),
                "simulated": "true" if simulate else "false",
            }
        )

    if client is not None:
        client.close()

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "n",
        "dtype",
        "gpu_time_ms",
        "round_trip_ms",
        "full_recompute_ms",
        "freivalds_k1_ms",
        "freivalds_k3_ms",
        "freivalds_k5_ms",
        "simulated",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nWrote overhead records to {out_path}")
    return records


def main():
    parser = argparse.ArgumentParser(description="Evaluate verification runtime overhead")
    parser.add_argument("--worker", type=str, default=None, help="Worker address")
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=True,
        help="Simulate execution locally with NumPy (default: True)",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Repeats per size")
    parser.add_argument(
        "--sizes", type=str, default="256,512,1024,2048", help="Comma-separated matrix sizes"
    )
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument(
        "--output", type=str, default="scripts/eval/data/overhead.csv", help="Output CSV path"
    )
    args = parser.parse_args()

    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    simulate = args.simulate if args.worker is None else False

    run_overhead_evaluation(
        sizes=sizes,
        repeats=args.repeats,
        dtype_str=args.dtype,
        simulate=simulate,
        worker_host=args.worker,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
