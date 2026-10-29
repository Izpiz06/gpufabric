from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# GPUFabric | RTX 3050 Benchmark Analysis
# ============================================================

GPU = "NVIDIA GeForce RTX 3050 6GB Laptop GPU"
OUTPUT = Path("benchmark_graphs")
OUTPUT.mkdir(exist_ok=True)

# Measured benchmark results
network = 6.3                  # MB/s
network_min = 2.0              # MB/s

memory = 125.0                 # GB/s
memory_peak = 168.0            # GB/s
memory_min = memory_peak * 0.50

matmul = 4237                  # GFLOPS

pi_estimate = 3.141534
pi_reference = np.pi
pi_error = abs(pi_estimate - pi_reference)
pi_tolerance = 3.7e-4
mc_points = 500_000_000

# Clean, report-friendly styling
plt.rcParams.update({
    "figure.figsize": (9, 5.5),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "axes.axisbelow": True,
})


def finish(filename):
    """Save and display the current figure."""
    plt.tight_layout()
    plt.savefig(OUTPUT / filename, bbox_inches="tight")
    plt.show()
    plt.close()


def annotate_bars(ax, bars, fmt):
    """Label bars with their values."""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


# ------------------------------------------------------------
# 1. Network throughput
# ------------------------------------------------------------

fig, ax = plt.subplots()

labels = ["Measured", "Minimum requirement"]
values = [network, network_min]

bars = ax.bar(labels, values, width=0.55)
annotate_bars(ax, bars, lambda x: f"{x:.1f} MB/s")

ax.set_title("Network Triad Throughput", loc="left", pad=14)
ax.set_ylabel("Throughput (MB/s)")
ax.set_ylim(0, max(values) * 1.25)

ax.text(
    0.98, 0.94,
    f"Threshold attainment: {network / network_min:.2f}×",
    transform=ax.transAxes,
    ha="right", va="top",
)

finish("01_network_throughput.png")


# ------------------------------------------------------------
# 2. GPU memory bandwidth
# ------------------------------------------------------------

fig, ax = plt.subplots()

labels = ["Measured", "50% acceptance threshold", "Theoretical peak"]
values = [memory, memory_min, memory_peak]

bars = ax.bar(labels, values, width=0.60)
annotate_bars(ax, bars, lambda x: f"{x:.1f} GB/s")

ax.set_title("GPU Memory Bandwidth", loc="left", pad=14)
ax.set_ylabel("Bandwidth (GB/s)")
ax.set_ylim(0, memory_peak * 1.20)

ax.text(
    0.98, 0.94,
    f"Peak utilization: {memory / memory_peak:.1%}",
    transform=ax.transAxes,
    ha="right", va="top",
)

finish("02_memory_bandwidth.png")


# ------------------------------------------------------------
# 3. Matrix multiplication performance
# ------------------------------------------------------------

fig, ax = plt.subplots()

bars = ax.bar(["1000 × 1000 FP32"], [matmul], width=0.45)
annotate_bars(ax, bars, lambda x: f"{x:,.0f} GFLOPS")

ax.set_title("Matrix Multiplication Performance", loc="left", pad=14)
ax.set_ylabel("Performance (GFLOPS)")
ax.set_ylim(0, matmul * 1.22)

ax.text(
    0.98, 0.94,
    f"Measured throughput: {matmul / 1000:.3f} TFLOPS",
    transform=ax.transAxes,
    ha="right", va="top",
)

finish("03_matmul_performance.png")


# ------------------------------------------------------------
# 4. Monte Carlo pi accuracy
# ------------------------------------------------------------

fig, ax = plt.subplots()

labels = ["Observed error", "Allowed error"]
values = [pi_error, pi_tolerance]

bars = ax.bar(labels, values, width=0.50)
annotate_bars(ax, bars, lambda x: f"{x:.2e}")

ax.set_title("Monte Carlo π Numerical Accuracy", loc="left", pad=14)
ax.set_ylabel("Absolute error")
ax.set_ylim(0, pi_tolerance * 1.25)

ax.text(
    0.98, 0.94,
    f"Samples: {mc_points:,}\n"
    f"Tolerance used: {pi_error / pi_tolerance:.1%}",
    transform=ax.transAxes,
    ha="right", va="top",
)

finish("04_monte_carlo_accuracy.png")


# ------------------------------------------------------------
# 5. Analytical compliance profile
# ------------------------------------------------------------
# Normalize the three threshold-based tests so 100% means
# exactly meeting the threshold.
#
# Higher is better for every metric:
#   Network: measured / minimum
#   Memory:  measured / minimum
#   Accuracy: allowed error / observed error
#
# This is a normalized comparison, NOT a time-series graph.

metrics = [
    "Network throughput",
    "Memory bandwidth",
    "Monte Carlo accuracy",
]

compliance = [
    network / network_min * 100,
    memory / memory_min * 100,
    pi_tolerance / pi_error * 100,
]

fig, ax = plt.subplots(figsize=(10, 5.8))

x = np.arange(len(metrics))

ax.plot(
    x,
    compliance,
    marker="o",
    markersize=8,
    linewidth=2,
)

ax.axhline(
    100,
    linestyle="--",
    linewidth=1.4,
    label="Acceptance threshold (100%)",
)

for xi, yi in zip(x, compliance):
    ax.annotate(
        f"{yi:.1f}%",
        (xi, yi),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel("Normalized threshold compliance (%)")
ax.set_title(
    "GPUFabric Benchmark Compliance Profile",
    loc="left",
    pad=14,
)

ax.set_ylim(0, max(compliance) * 1.18)
ax.legend(frameon=False, loc="upper left")

ax.text(
    0.99, 0.94,
    "Higher is better; 100% indicates the threshold.",
    transform=ax.transAxes,
    ha="right", va="top",
    fontsize=9,
)

finish("05_compliance_profile.png")


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\nGPUFabric Benchmark Summary")
print("=" * 58)
print(f"GPU: {GPU}")
print(f"Network throughput : {network:.1f} MB/s — PASS")
print(
    f"Memory bandwidth   : {memory:.1f} GB/s "
    f"({memory / memory_peak:.1%} of peak) — PASS"
)
print(f"Matrix multiplication: {matmul:,} GFLOPS")
print(f"Monte Carlo points : {mc_points:,}")
print(f"Monte Carlo π      : {pi_estimate:.6f}")
print(f"Absolute error     : {pi_error:.3e} — PASS")
print(f"\nSaved graphs in: {OUTPUT.resolve()}")
