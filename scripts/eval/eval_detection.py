# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Evaluate verification detection rate across fault modes, rates, magnitudes, and rounds."""

import argparse
import csv
from pathlib import Path

import numpy as np

from client.client import GPUFabricClient
from client.verify import verify_result
from worker.faults import FaultInjector


def run_detection_evaluation(
    sizes=(256, 512),
    rounds_list=(1, 2, 3, 4, 5, 6),
    rates=(0.0001, 0.001, 0.01, 0.05),
    magnitudes=(1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0),
    modes=("random_elements", "bitflip", "zeros", "scale"),
    trials_per_cell=30,
    dtype_str="float32",
    simulate=True,
    worker_host=None,
    output_csv="scripts/eval/data/detection.csv",
):
    dtype = np.dtype(dtype_str)
    records = []
    client = None

    if not simulate and worker_host:
        client = GPUFabricClient(host=worker_host)

    rng = np.random.default_rng(2026)

    total_cells = len(sizes) * len(modes) * len(rates) * len(magnitudes) * len(rounds_list)
    print(f"Running detection evaluation over {total_cells} cells (trials={trials_per_cell})...")
    done = 0

    for n in sizes:
        for mode in modes:
            for rate in rates:
                for mag in magnitudes:
                    # For modes like zeros or bitflip where magnitude or rate is fixed/irrelevant,
                    # we can avoid redundant repetitions if desired, but running provides complete grid.
                    for k in rounds_list:
                        detected = 0
                        false_positives = 0

                        for t in range(trials_per_cell):
                            seed = int(rng.integers(1, 1_000_000_000))
                            trial_rng = np.random.default_rng(seed)
                            a = trial_rng.uniform(-2.0, 2.0, size=(n, n)).astype(dtype)
                            b = trial_rng.uniform(-2.0, 2.0, size=(n, n)).astype(dtype)

                            if simulate or client is None:
                                c_honest = np.matmul(a, b)
                            else:
                                c_honest = client.compute("matmul", a, b).result

                            # Check false positive on honest output
                            rep_honest = verify_result(
                                "matmul",
                                [a, b],
                                c_honest,
                                mode="fast",
                                rounds=k,
                                rng=trial_rng,
                            )
                            if not rep_honest.passed:
                                false_positives += 1

                            # Inject fault
                            injector = FaultInjector(mode=mode, rate=rate, magnitude=mag, seed=seed)
                            c_corrupted = injector.inject(c_honest)

                            # Verify corrupted output
                            rep_fault = verify_result(
                                "matmul",
                                [a, b],
                                c_corrupted,
                                mode="fast",
                                rounds=k,
                                rng=trial_rng,
                            )
                            if not rep_fault.passed:
                                detected += 1

                        detection_rate = detected / trials_per_cell
                        records.append(
                            {
                                "n": n,
                                "dtype": dtype_str,
                                "mode": mode,
                                "rate": rate,
                                "magnitude": mag,
                                "k": k,
                                "trials": trials_per_cell,
                                "detected": detected,
                                "detection_rate": round(detection_rate, 4),
                                "false_positive_trials": false_positives,
                                "simulated": "true" if simulate else "false",
                            }
                        )
                        done += 1
                        if done % 100 == 0 or done == total_cells:
                            print(f"  Completed {done}/{total_cells} cells...")

    if client is not None:
        client.close()

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "n",
        "dtype",
        "mode",
        "rate",
        "magnitude",
        "k",
        "trials",
        "detected",
        "detection_rate",
        "false_positive_trials",
        "simulated",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nWrote {len(records)} detection records to {out_path}")
    return records


def main():
    parser = argparse.ArgumentParser(description="Evaluate verification detection rate")
    parser.add_argument("--worker", type=str, default=None, help="Worker address")
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=True,
        help="Simulate execution locally with NumPy (default: True)",
    )
    parser.add_argument("--trials", type=int, default=20, help="Trials per grid cell")
    parser.add_argument("--sizes", type=str, default="256,512", help="Comma-separated matrix sizes")
    parser.add_argument("--rounds", type=str, default="1,2,3,4,5,6", help="Rounds to sweep")
    parser.add_argument(
        "--rates", type=str, default="0.0001,0.001,0.01", help="Corruption rates to sweep"
    )
    parser.add_argument(
        "--magnitudes", type=str, default="0.001,0.01,0.1,1.0", help="Magnitudes to sweep"
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="random_elements,bitflip,zeros,scale",
        help="Comma-separated fault modes",
    )
    parser.add_argument(
        "--output", type=str, default="scripts/eval/data/detection.csv", help="Output CSV path"
    )
    args = parser.parse_args()

    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    rounds_list = [int(x.strip()) for x in args.rounds.split(",") if x.strip()]
    rates = [float(x.strip()) for x in args.rates.split(",") if x.strip()]
    magnitudes = [float(x.strip()) for x in args.magnitudes.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    simulate = args.simulate if args.worker is None else False

    run_detection_evaluation(
        sizes=sizes,
        rounds_list=rounds_list,
        rates=rates,
        magnitudes=magnitudes,
        modes=modes,
        trials_per_cell=args.trials,
        simulate=simulate,
        worker_host=args.worker,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
