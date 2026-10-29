# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Analytical plotting and visualization for GPU Fabric benchmarks."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("gpufabric.client.plot")


def generate_benchmark_plot(
    target: str,
    gpu_name: str,
    driver_version: str,
    cuda_version: str,
    network_history: List[Dict[str, float]],
    triad_data: List[Dict[str, Any]],
    matmul_data: List[Dict[str, Any]],
    monte_carlo_data: Dict[str, Any],
    workload_latencies: Dict[str, Dict[str, float]],
    output_path: Optional[Path] = None,
    peak_bandwidth_gb_s: float = 0.0,
    compliance_data: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Generate analytical benchmark figures and dashboard and save to disk.

    Produces:
      - 2x2 Comprehensive Analytical Dashboard (PNG and SVG)
      - Individual detailed line graphs saved in the output directory
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        logger.warning(f"matplotlib is not available; skipping benchmark plot generation: {e}")
        return None

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_target = target.replace(":", "-").replace(".", "-")

    if output_path is None:
        out_dir = Path("benchmark_graphs")
        out_dir.mkdir(parents=True, exist_ok=True)
        dashboard_path = out_dir / f"benchmark_dashboard_{safe_target}_{timestamp}.png"
    else:
        dashboard_path = Path(output_path)
        out_dir = dashboard_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Multi-Panel Comprehensive Analytical Dashboard
    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.5), dpi=300)
    fig_title = f"GPU Fabric Benchmark Analysis — {gpu_name} ({target})"
    sub_title = (
        f"Driver: {driver_version or 'N/A'} | CUDA: {cuda_version or 'N/A'} | "
        f"Peak Memory BW: {peak_bandwidth_gb_s:.1f} GB/s | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    fig.suptitle(fig_title, fontsize=13, fontweight="bold", y=0.98)
    fig.text(0.5, 0.95, sub_title, ha="center", fontsize=9, color="#64748b")

    # Panel 1: Network Throughput & Latency Timeline
    ax1 = axs[0, 0]
    if network_history:
        trials = [d["trial"] for d in network_history]
        mb_s = [d["throughput_mb_s"] for d in network_history]
        rtt_ms = [d["round_trip_ms"] for d in network_history]
        gpu_ms = [d.get("gpu_time_ms", 0.0) for d in network_history]

        l1 = ax1.plot(
            trials, mb_s, "o-", color="#10b981", linewidth=2.2, markersize=6, label="Throughput (MB/s)"
        )
        med_val = float(np.median(mb_s))
        l1_med = ax1.axhline(
            med_val, color="#059669", linestyle="--", alpha=0.8, label=f"Median ({med_val:.1f} MB/s)"
        )
        ax1.set_xlabel("Trial Iteration #")
        ax1.set_ylabel("Network Throughput (MB/s)", color="#10b981", fontweight="bold")
        ax1.set_title("1. Network Throughput & Latency Timeline", fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.6)

        ax1_twin = ax1.twinx()
        l2 = ax1_twin.plot(
            trials, rtt_ms, "s-", color="#f59e0b", linewidth=2.0, markersize=5, label="Round-Trip (ms)"
        )
        l3 = ax1_twin.plot(
            trials, gpu_ms, "^-", color="#6366f1", linewidth=1.8, markersize=5, label="Remote GPU Kernel (ms)"
        )
        ax1_twin.set_ylabel("Latency (ms)", color="#f59e0b", fontweight="bold")

        lines1 = l1 + [l1_med] + l2 + l3
        labels1 = [line.get_label() for line in lines1]
        ax1.legend(lines1, labels1, loc="best", fontsize=8)
    else:
        ax1.text(0.5, 0.5, "No network trial history recorded", ha="center", va="center")
        ax1.set_title("1. Network Throughput & Latency Timeline", fontweight="bold")

    # Panel 2: GPU Memory Bandwidth & Latency vs Data Size (STREAM Triad)
    ax2 = axs[0, 1]
    if triad_data:
        mb_sizes = [d["vram_mb"] for d in triad_data]
        bw_vals = [d["bandwidth_gb_s"] for d in triad_data]
        lat_vals = [d["latency_ms"] for d in triad_data]

        l_bw = ax2.plot(
            mb_sizes,
            bw_vals,
            "o-",
            color="#3b82f6",
            linewidth=2.2,
            markersize=6,
            label="Measured Bandwidth (GB/s)",
        )
        lines2 = list(l_bw)
        if peak_bandwidth_gb_s > 0:
            l_peak = ax2.axhline(
                peak_bandwidth_gb_s,
                color="#94a3b8",
                linestyle="--",
                linewidth=1.8,
                label=f"Theoretical Peak ({peak_bandwidth_gb_s:.1f} GB/s)",
            )
            lines2.append(l_peak)

        ax2.set_xlabel("VRAM Footprint (MB)")
        ax2.set_ylabel("Triad Bandwidth (GB/s)", color="#3b82f6", fontweight="bold")
        ax2.set_title("2. GPU Memory Bandwidth & Latency Scaling (STREAM Triad)", fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.6)

        ax2_twin = ax2.twinx()
        l_lat = ax2_twin.plot(
            mb_sizes,
            lat_vals,
            "d-",
            color="#ef4444",
            linewidth=1.8,
            markersize=5,
            label="Kernel Latency (ms)",
        )
        ax2_twin.set_ylabel("Kernel Execution Time (ms)", color="#ef4444", fontweight="bold")

        lines2.extend(l_lat)
        labels2 = [line.get_label() for line in lines2]
        ax2.legend(lines2, labels2, loc="best", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No triad bandwidth data available", ha="center", va="center")
        ax2.set_title("2. GPU Memory Bandwidth Scaling", fontweight="bold")

    # Panel 3: Compute Throughput & Scaling (SGEMM Matmul)
    ax3 = axs[1, 0]
    if matmul_data:
        dim_sizes = [d["size"] for d in matmul_data]
        gflops_vals = [d["gflops"] for d in matmul_data]
        mat_lat_vals = [d["latency_ms"] for d in matmul_data]

        l_gflops = ax3.plot(
            dim_sizes,
            gflops_vals,
            "D-",
            color="#8b5cf6",
            linewidth=2.2,
            markersize=6,
            label="Throughput (GFLOPS)",
        )
        ax3.set_xlabel("Matrix Dimension N (N x N float32)")
        ax3.set_ylabel("Compute Throughput (GFLOPS)", color="#8b5cf6", fontweight="bold")
        ax3.set_title("3. Compute Performance: GFLOPS & Scaling (SGEMM)", fontweight="bold")
        ax3.grid(True, linestyle=":", alpha=0.6)

        ax3_twin = ax3.twinx()
        l_mlat = ax3_twin.plot(
            dim_sizes,
            mat_lat_vals,
            "o-",
            color="#ec4899",
            linewidth=1.8,
            markersize=5,
            label="Kernel Latency (ms)",
        )
        ax3_twin.set_ylabel("Kernel Latency (ms)", color="#ec4899", fontweight="bold")

        lines3 = l_gflops + l_mlat
        labels3 = [line.get_label() for line in lines3]
        ax3.legend(lines3, labels3, loc="best", fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No matmul benchmark data available", ha="center", va="center")
        ax3.set_title("3. Compute Performance: GFLOPS & Scaling", fontweight="bold")

    # Panel 4: Workload Latency Comparison & Timeline Breakdown
    ax4 = axs[1, 1]
    if workload_latencies:
        workloads = list(workload_latencies.keys())
        tot_ms = [workload_latencies[w].get("total_ms", 0.0) for w in workloads]
        gpu_ms = [workload_latencies[w].get("gpu_ms", 0.0) for w in workloads]

        x = np.arange(len(workloads))
        w = 0.35
        bars_tot = ax4.bar(
            x - w / 2, tot_ms, w, label="Total Round-Trip (ms)", color="#f59e0b", alpha=0.85
        )
        bars_gpu = ax4.bar(
            x + w / 2, gpu_ms, w, label="GPU Kernel Only (ms)", color="#3b82f6", alpha=0.85
        )

        ax4.set_xticks(x)
        ax4.set_xticklabels(workloads, rotation=12, ha="right", fontsize=9)
        ax4.set_ylabel("Execution Latency (ms, log scale)", fontweight="bold")
        ax4.set_yscale("log")
        ax4.set_title("4. Workload Latency: Network Round-Trip vs GPU Execution", fontweight="bold")
        ax4.grid(True, linestyle=":", alpha=0.6, which="both")
        ax4.legend(loc="upper right", fontsize=8)

        # Bar value annotations
        for bar in bars_tot:
            h = bar.get_height()
            if h > 0:
                ax4.text(
                    bar.get_x() + bar.get_width() / 2,
                    h * 1.15,
                    f"{h:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#b45309",
                )
        for bar in bars_gpu:
            h = bar.get_height()
            if h > 0:
                ax4.text(
                    bar.get_x() + bar.get_width() / 2,
                    h * 1.15,
                    f"{h:.2f}" if h < 1 else f"{h:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#1d4ed8",
                )
    else:
        ax4.text(0.5, 0.5, "No workload latency comparison available", ha="center", va="center")
        ax4.set_title("4. Workload Latency Comparison", fontweight="bold")

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(dashboard_path, format="png", dpi=300, bbox_inches="tight")
    svg_path = dashboard_path.with_suffix(".svg")
    try:
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 2. Individual Analytical Line Graph: Compliance Profile
    # -------------------------------------------------------------------------
    if compliance_data:
        try:
            fig_comp, ax_comp = plt.subplots(figsize=(9, 5.5), dpi=300)
            metrics = list(compliance_data.keys())
            compliance_vals = [compliance_data[m]["pct"] for m in metrics]

            x_comp = np.arange(len(metrics))
            ax_comp.plot(
                x_comp,
                compliance_vals,
                marker="o",
                markersize=8,
                linewidth=2.2,
                color="#2563eb",
            )
            ax_comp.axhline(
                100.0,
                linestyle="--",
                linewidth=1.5,
                color="#dc2626",
                label="Acceptance Threshold (100%)",
            )
            for xi, yi in zip(x_comp, compliance_vals):
                ax_comp.annotate(
                    f"{yi:.1f}%",
                    (xi, yi),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                )
            ax_comp.set_xticks(x_comp)
            ax_comp.set_xticklabels(metrics, fontsize=10)
            ax_comp.set_ylabel("Normalized Threshold Compliance (%)", fontweight="bold")
            ax_comp.set_title(
                f"GPUFabric Benchmark Compliance Profile — {gpu_name}", loc="left", pad=14, fontweight="bold"
            )
            ax_comp.set_ylim(0, max(max(compliance_vals) * 1.25, 120.0))
            ax_comp.grid(True, linestyle=":", alpha=0.5)
            ax_comp.legend(frameon=False, loc="upper left")

            comp_file = out_dir / f"05_compliance_profile_{safe_target}.png"
            plt.tight_layout()
            fig_comp.savefig(comp_file, bbox_inches="tight")
            plt.close(fig_comp)
        except Exception as e:
            logger.debug(f"Error plotting compliance profile: {e}")

    logger.info(f"Generated benchmark dashboard: {dashboard_path}")
    return dashboard_path
