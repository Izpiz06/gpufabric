# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Plot trust score timeline over time with threshold lines and drain marker."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

FIG_SIZE = (5.0, 3.5)
DPI = 300

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.titlesize": 11,
        "figure.autolayout": True,
    }
)


def plot_timeline(csv_path: Path, output_dir: Path):
    if not csv_path.exists():
        print(f"Data file {csv_path} does not exist; run eval_trust_timeline.py first.")
        return

    honest_steps, honest_scores = [], []
    faulty_steps, faulty_scores = [], []
    drain_step = None
    drain_score = None

    h_idx = 0
    f_idx = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entity = row["entity"]
            score = float(row["new_score"])
            state = row["new_state"]

            if "honest" in entity:
                h_idx += 1
                honest_steps.append(h_idx)
                honest_scores.append(score)
            elif "faulty" in entity:
                f_idx += 1
                faulty_steps.append(f_idx)
                faulty_scores.append(score)
                if state == "DRAIN" and drain_step is None:
                    drain_step = f_idx
                    drain_score = score

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(honest_steps, honest_scores, "g-", linewidth=2.0, label="Honest Node (RTX 4060)")
    ax.plot(faulty_steps, faulty_scores, "r-", linewidth=2.0, label="Faulty Node (Injected Faults)")

    # Threshold lines
    ax.axhline(0.60, color="orange", linestyle="--", linewidth=1.2, label="Warn Threshold (0.60)")
    ax.axhline(0.35, color="red", linestyle=":", linewidth=1.5, label="Drain Threshold (0.35)")

    # Mark drain event
    if drain_step is not None:
        ax.plot(drain_step, drain_score, "rX", markersize=10, label="Slurm DRAIN Fired")
        ax.annotate(
            "Slurm DRAIN",
            xy=(drain_step, drain_score),
            xytext=(drain_step - 5, drain_score - 0.15),
            arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=5),
            fontweight="bold",
            color="darkred",
        )

    ax.set_xlabel("Evaluation Event Sequence")
    ax.set_ylabel("Beta Trust Score [0, 1]")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Node Reputation Over Time & Automated Drainage", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left")

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "fig5_trust_timeline.svg"
    png_path = output_dir / "fig5_trust_timeline.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {svg_path} and {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot trust score timeline figure")
    parser.add_argument(
        "--data",
        type=str,
        default="scripts/eval/data/trust_timeline.csv",
        help="Input timeline CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scripts/eval/figures",
        help="Output figures directory",
    )
    args = parser.parse_args()

    plot_timeline(Path(args.data), Path(args.output_dir))


if __name__ == "__main__":
    main()
