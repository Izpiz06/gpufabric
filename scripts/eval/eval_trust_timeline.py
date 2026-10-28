# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Simulate and record node reputation timelines through TrustStore.

Generates realistic sequences of verification passes, transient faults, and repeated
injected corruptions, outputting an event timeline CSV for evaluation plotting.
"""

import argparse
import csv
from pathlib import Path

from common.policy import LogOnlyActions, TrustMonitor
from common.trust import TrustStore


def run_timeline_simulation(
    steps=60,
    output_csv="scripts/eval/data/trust_timeline.csv",
    db_path=None,
    simulate=True,
):
    if db_path is None:
        db_path = Path("scripts/eval/data/sim_trust.db")
    if db_path.exists():
        db_path.unlink()

    store = TrustStore(db_path=db_path)
    policy = LogOnlyActions()
    monitor = TrustMonitor(trust_store=store, policy_actions=policy)

    # 1. Honest node: consistently passes, has 1 transient glitch around step 15, recovers
    for step in range(steps):
        if step == 15:
            monitor.record_verification(
                "node:honest-4060",
                passed=False,
                detail="Transient soft error (single bit)",
            )
        else:
            monitor.record_verification(
                "node:honest-4060", passed=True, detail="Matmul verification passed"
            )

    # 2. Faulty node: honest until step 20, then experiences persistent injected faults
    for step in range(steps):
        if step < 20:
            monitor.record_verification(
                "node:faulty-4060", passed=True, detail="Matmul verification passed"
            )
        else:
            monitor.record_verification(
                "node:faulty-4060",
                passed=False,
                detail="Corrupted matmul (injected fault)",
            )

    events = store.get_events(limit=steps * 2 + 10)
    # Reverse events to chronological order
    events.reverse()

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "step",
        "timestamp",
        "entity",
        "event",
        "weight",
        "old_score",
        "new_score",
        "old_state",
        "new_state",
        "detail",
        "simulated",
    ]
    records = []
    for idx, ev in enumerate(events):
        records.append(
            {
                "step": idx + 1,
                "timestamp": ev["timestamp"],
                "entity": ev["entity"],
                "event": ev["event"],
                "weight": ev["weight"],
                "old_score": ev["old_score"],
                "new_score": ev["new_score"],
                "old_state": ev["old_state"],
                "new_state": ev["new_state"],
                "detail": ev["detail"],
                "simulated": "true" if simulate else "false",
            }
        )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Recorded {len(records)} timeline events to {out_path}")
    print(f"Policy actions executed: {len(policy.actions_log)}")
    for act in policy.actions_log:
        print(f"  {act}")

    return records


def main():
    parser = argparse.ArgumentParser(description="Generate trust timeline simulation data")
    parser.add_argument(
        "--output",
        type=str,
        default="scripts/eval/data/trust_timeline.csv",
        help="Output CSV path",
    )
    parser.add_argument("--steps", type=int, default=40, help="Simulation steps per node")
    args = parser.parse_args()

    run_timeline_simulation(steps=args.steps, output_csv=args.output)


if __name__ == "__main__":
    main()
