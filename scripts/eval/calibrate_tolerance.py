# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Calibrate floating-point tolerance threshold for Freivalds matrix verification.

Measures normalized Freivalds residual distribution across honest matmul runs
(real worker or --simulate) to determine optimal threshold_factor.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from client.client import GPUFabricClient
from client.verify import freivalds_check


def run_calibration(
    sizes=(128, 256, 512, 1024),
    trials_per_size=50,
    dtype_str="float32",
    simulate=True,
    worker_host=None,
    output_csv=None,
):
    dtype = np.dtype(dtype_str)
    records = []
    client = None

    if not simulate and worker_host:
        client = GPUFabricClient(host=worker_host)

    rng = np.random.default_rng(42)

    for n in sizes:
        for trial in range(trials_per_size):
            a = rng.uniform(-5.0, 5.0, size=(n, n)).astype(dtype)
            b = rng.uniform(-5.0, 5.0, size=(n, n)).astype(dtype)

            if simulate or client is None:
                c = np.matmul(a, b)
            else:
                res = client.compute("matmul", a, b)
                c = res.result

            # Run Freivalds check with factor=1.0 to get raw residual & base scale
            chk = freivalds_check(a, b, c, rounds=3, rng=rng, threshold_factor=1.0)
            base_scale = chk.threshold
            max_res = chk.max_residual
            norm_res = max_res / base_scale if base_scale > 0 else 0.0

            records.append(
                {
                    "size": n,
                    "dtype": dtype_str,
                    "trial": trial,
                    "max_residual": max_res,
                    "base_scale": base_scale,
                    "normalized_residual": norm_res,
                    "simulated": "true" if simulate else "false",
                }
            )

    if client is not None:
        client.close()

    if output_csv:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "size",
                    "dtype",
                    "trial",
                    "max_residual",
                    "base_scale",
                    "normalized_residual",
                    "simulated",
                ],
            )
            writer.writeheader()
            writer.writerows(records)
        print(f"Wrote {len(records)} calibration records to {out_path}")

    # Summary
    norm_vals = [r["normalized_residual"] for r in records]
    max_norm = max(norm_vals) if norm_vals else 0.0
    p95 = float(np.percentile(norm_vals, 95)) if norm_vals else 0.0
    rec_factor = float(np.ceil(max_norm * 2.5)) if max_norm > 0 else 10.0

    print("\nCalibration Summary:")
    print(f"  Total trials: {len(records)} (simulated={simulate})")
    print(f"  P95 normalized residual: {p95:.3f}")
    print(f"  Max honest normalized residual: {max_norm:.3f}")
    print(f"  Recommended threshold_factor: {rec_factor} (with ~2.5x safety margin)")

    return records


def main():
    parser = argparse.ArgumentParser(description="Calibrate Freivalds threshold factor")
    parser.add_argument("--worker", type=str, default=None, help="Worker address")
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=True,
        help="Simulate execution locally with NumPy (default: True)",
    )
    parser.add_argument("--trials", type=int, default=50, help="Trials per matrix size")
    parser.add_argument(
        "--sizes", type=str, default="128,256,512,1024", help="Comma-separated matrix sizes"
    )
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument(
        "--output",
        type=str,
        default="scripts/eval/data/calibration.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    simulate = args.simulate if args.worker is None else False
    run_calibration(
        sizes=sizes,
        trials_per_size=args.trials,
        dtype_str=args.dtype,
        simulate=simulate,
        worker_host=args.worker,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
