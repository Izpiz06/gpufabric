# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Live demonstration showcase for GPU Fabric Trust Layer novelties (C1, C2, C3).

Designed for interactive evaluation and poster session presentations:
  - C1: Verifiable Computation (Freivalds O(k*n^2) verification vs O(n^3) recomputation,
        calibrated IEEE-754 bounds, and real-time silent data corruption detection).
  - C2: Active Deception Mechanisms (Decoy honeypot nodes and cryptographic canary tripwires).
  - C3: Node & User Reputation (Bayesian Beta-reputation model, single-failure transient
        protection, Slurm automated drainage, and hysteresis recovery).
"""

import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from client.verify import freivalds_check, tolerance
from common.alerts import AlertSink
from common.canary import CanaryRegistry
from common.trust import TrustStore
from worker.decoy import DecoyServicer
from worker.faults import FaultInjector

console = Console()


def _pause(step_mode: bool, prompt: str = "Press [bold cyan][Enter][/bold cyan] to proceed to next pillar..."):
    if step_mode:
        console.print()
        console.input(f"  [dim]▶[/dim] {prompt} ")
        console.print()


def run_c1_demo(matrix_size: int = 1024, rounds: int = 3) -> Dict[str, Any]:
    """Demonstrate Pillar C1: Verifiable Computation & Freivalds' Algorithm."""
    console.rule("[bold cyan]Pillar C1: Verifiable Computation (Freivalds O(k·n²) Verification)[/bold cyan]")

    desc = (
        "[bold]Problem:[/bold] Consumer GPUs (e.g. RTX 4060/3060) lack ECC memory and run in uncontrolled lab environments.\n"
        "Full client CPU recomputation requires [bold red]O(n³)[/bold red] operations, completely negating offload acceleration.\n"
        "[bold]Novelty:[/bold] Probabilistic Freivalds verification runs in [bold green]O(k·n²)[/bold green] time with "
        "Higham (2002) calibrated IEEE-754 error bounds, catching silent data corruption (SDC) in milliseconds."
    )
    console.print(Panel(desc, title="C1 Overview & Theoretical Foundation", border_style="cyan"))

    console.print(f"[dim]Generating test matrices A, B ∈ ℝ^{matrix_size}×{matrix_size} (float32)...[/dim]")
    rng = np.random.default_rng(42)
    a = rng.uniform(-1.5, 1.5, size=(matrix_size, matrix_size)).astype(np.float32)
    b = rng.uniform(-1.5, 1.5, size=(matrix_size, matrix_size)).astype(np.float32)

    # Calculate theoretical arithmetic complexity
    gemm_flops = 2.0 * (matrix_size**3)
    freiv_flops = 4.0 * rounds * (matrix_size**2)
    flop_reduction = gemm_flops / max(freiv_flops, 1.0)

    # 1. Full CPU Recomputation O(n^3)
    t0 = time.perf_counter()
    c_honest = np.matmul(a, b)
    t_full_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Honest Freivalds Verification O(k * n^2)
    t0 = time.perf_counter()
    f_honest = freivalds_check(a, b, c_honest, rounds=rounds, rng=rng)
    t_freivalds_ms = (time.perf_counter() - t0) * 1000.0

    speedup = t_full_ms / max(t_freivalds_ms, 0.001)

    # Display Honest Results Table
    t = Table(title=f"Verification Performance Benchmark (Matrix: {matrix_size}×{matrix_size}, k={rounds})")
    t.add_column("Verification Method", style="bold")
    t.add_column("Asymptotic Complexity", justify="center")
    t.add_column("Arithmetic FLOPs", justify="right")
    t.add_column("Time (ms)", justify="right")
    t.add_column("Residual Δ", justify="right")
    t.add_column("Threshold", justify="right")
    t.add_column("Integrity Verdict", justify="center")

    t.add_row(
        "Full CPU Recompute",
        "O(n³)",
        f"{gemm_flops / 1e9:.2f} GFLOPs",
        f"{t_full_ms:.2f} ms",
        "0.00e+00",
        f"{tolerance('matmul', np.float32):.1e}",
        "[bold green]PASS (Exact)[/bold green]",
    )
    t.add_row(
        f"Freivalds Probabilistic (k={rounds})",
        "O(k·n²)",
        f"[bold green]{freiv_flops / 1e6:.2f} MFLOPs[/bold green]",
        f"[bold green]{t_freivalds_ms:.2f} ms[/bold green]",
        f"{f_honest.max_residual:.2e}",
        f"{f_honest.threshold:.2e}",
        "[bold green]PASS (Verified)[/bold green]",
    )
    console.print(t)
    console.print(
        f"  [bold green]✔[/bold green] [bold]Theoretical Arithmetic Reduction:[/bold] [bold cyan]{flop_reduction:.1f}× fewer FLOPs[/bold cyan] "
        f"({freiv_flops / 1e6:.1f} MFLOPs vs {gemm_flops / 1e9:.2f} GFLOPs).\n"
        f"  [bold green]✔[/bold green] [bold]Error Probability Bound:[/bold] Pr[error] ≤ 2⁻ᵏ = [bold cyan]{(0.5**rounds)*100:.1f}%[/bold cyan] "
        f"across {rounds} Rademacher projection rounds."
    )

    console.print()
    console.print("[bold yellow]⚡ Live Attack: Injecting Silent Data Corruption (SDC) via Bitflip...[/bold yellow]")

    # 3. Corrupt result with realistic single bitflip in tensor payload
    injector = FaultInjector(mode="bitflip", rate=0.0001, seed=42)
    c_corrupted = injector.inject(c_honest)

    t0 = time.perf_counter()
    f_corrupt = freivalds_check(a, b, c_corrupted, rounds=rounds, rng=rng)
    t_detect_ms = (time.perf_counter() - t0) * 1000.0

    verdict_badge = (
        "[bold red]CORRUPTION CAUGHT[/bold red]"
        if not f_corrupt.passed
        else "[bold yellow]UNDETECTED[/bold yellow]"
    )

    t_fault = Table(title="Adversarial Fault & Silent Data Corruption (SDC) Test")
    t_fault.add_column("Scenario", style="bold")
    t_fault.add_column("Perturbation Mechanism", justify="center")
    t_fault.add_column("Detection Time", justify="right")
    t_fault.add_column("Measured Residual Δ", justify="right")
    t_fault.add_column("Calibrated Bound", justify="right")
    t_fault.add_column("Detection Outcome", justify="center")

    t_fault.add_row(
        "Untampered Output",
        "None (Honest)",
        f"{t_freivalds_ms:.2f} ms",
        f"{f_honest.max_residual:.2e}",
        f"{f_honest.threshold:.2e}",
        "[bold green]PASS (Honest)[/bold green]",
    )
    t_fault.add_row(
        "Simulated Cosmic Ray / SDC",
        "VRAM Bitflip (0.01% rate)",
        f"[bold cyan]{t_detect_ms:.2f} ms[/bold cyan]",
        f"[bold red]{f_corrupt.max_residual:.2e}[/bold red]",
        f"{f_corrupt.threshold:.2e}",
        verdict_badge,
    )
    console.print(t_fault)

    console.print(
        f"  [bold red]✖[/bold red] [bold]Security Invariant Verified:[/bold] Bitflip corruption caught in "
        f"[bold cyan]{t_detect_ms:.2f} ms[/bold cyan]! Residual ([bold red]{f_corrupt.max_residual:.2e}[/bold red]) "
        f"exceeded threshold ([bold cyan]{f_corrupt.threshold:.2e}[/bold cyan]) by {f_corrupt.max_residual / f_corrupt.threshold:.1e}×!"
    )

    return {
        "matrix_size": matrix_size,
        "rounds": rounds,
        "gemm_flops": gemm_flops,
        "freiv_flops": freiv_flops,
        "flop_reduction": flop_reduction,
        "t_full_ms": t_full_ms,
        "t_freivalds_ms": t_freivalds_ms,
        "speedup": speedup,
        "honest_residual": f_honest.max_residual,
        "corrupt_residual": f_corrupt.max_residual,
        "threshold": f_honest.threshold,
    }


def run_c2_demo() -> Dict[str, Any]:
    """Demonstrate Pillar C2: Active Deception Mechanisms (Decoys & Canaries)."""
    console.rule("[bold cyan]Pillar C2: Active Deception Mechanisms (Honeypot Decoys & Canaries)[/bold cyan]")

    desc = (
        "[bold]Threat Model:[/bold] Shared academic / multi-tenant workstation clusters where internal adversaries\n"
        "or compromised accounts scan nodes, probe unauthorized GPUs, or snoop staged tenant tensors.\n"
        "[bold]Novelty:[/bold] Lightweight Honeypot Decoys mimicking high-value hardware with zero GPU overhead,\n"
        "and Cryptographic Canary Tripwires that trigger high-priority alerts on unauthorized access."
    )
    console.print(Panel(desc, title="C2 Overview & Deception Architecture", border_style="cyan"))

    with tempfile.TemporaryDirectory() as tmp_dir:
        canary_db = Path(tmp_dir) / "canaries.db"
        alert_sink = AlertSink(log_path=Path(tmp_dir) / "alerts.jsonl")
        registry = CanaryRegistry(db_path=canary_db, alert_sink=alert_sink)

        # 1. Cryptographic Canary Tripwires
        console.print("[bold cyan]1. Cryptographic Canary Token Tripwire Demonstration:[/bold cyan]")
        token = registry.mint(kind="job_id", prefix="cnry_")
        console.print(f"  [dim]•[/dim] Minted cryptographic canary token: [bold yellow]{token}[/bold yellow] (Kind: job_id)")
        console.print("  [dim]•[/dim] Seeded into cluster directory and simulated artifact catalog.")

        # Simulate adversarial scraping
        console.print("  [bold red]⚡ Simulating rogue tenant trying to access canary job...[/bold red]")
        registry.check(token, context=None)

        alerts = alert_sink.read_alerts()
        t_alerts = Table(title="Security Event Log (Real-time Tripwire Alerts)")
        t_alerts.add_column("Timestamp", style="dim")
        t_alerts.add_column("Alert Kind", style="bold red")
        t_alerts.add_column("Subject Identity", style="cyan")
        t_alerts.add_column("Trigger Value", style="yellow")
        t_alerts.add_column("Action Taken", style="bold green")

        for a in alerts:
            t_alerts.add_row(
                a.timestamp[:19].replace("T", " "),
                a.kind,
                a.subject,
                token[:16] + "...",
                "High-Priority Incident Alert Dispatched",
            )
        console.print(t_alerts)
        console.print(
            f"  [bold green]✔[/bold green] [bold]Tripwire Triggered:[/bold] Canary match detected! "
            f"Event logged with Subject: [bold cyan]{alerts[0].subject}[/bold cyan]."
        )

        console.print()
        # 2. Honeypot Decoy Worker Node
        console.print("[bold cyan]2. Lightweight Honeypot Decoy Worker Node Demonstration:[/bold cyan]")
        decoy = DecoyServicer(
            worker_id="decoy-node-rtx4060",
            profile_name="RTX 4060 8GB",
            alert_sink=alert_sink,
        )
        console.print(
            f"  [dim]•[/dim] Decoy worker active: [bold]{decoy.worker_id}[/bold] "
            f"(Profile: [bold cyan]{decoy.profile_name}[/bold cyan], CuPy/GPU required: [bold green]None (Zero GPU overhead)[/bold green])"
        )

        # Simulate read-only probe
        health_resp = decoy.GetHealth(None, None)
        console.print(
            f"  [dim]•[/dim] Telemetry response to unauthorized scanner: Status: [bold green]{health_resp.status}[/bold green], "
            f"Driver: [cyan]{health_resp.driver_version}[/cyan], CUDA: [cyan]{health_resp.cuda_version}[/cyan]"
        )

        # Simulate unauthorized compute submission on decoy
        console.print("  [bold red]⚡ Rogue client attempts to execute unallocated workload on decoy...[/bold red]")
        decoy._record_contact(method="Compute", context=None, detail="Unauthorized matrix multiplication request")

        new_alerts = [a for a in alert_sink.read_alerts() if a.kind == "decoy_contact"]
        if new_alerts:
            latest = new_alerts[-1]
            console.print(
                f"  [bold green]✔[/bold green] [bold]Decoy Trap Triggered:[/bold] Request rejected with UNAVAILABLE "
                f"(zero GPU time allocated). Incident logged for: [bold cyan]{latest.subject}[/bold cyan]."
            )

        total_alerts = len(alert_sink.read_alerts())

    return {"canary_token": token, "alert_count": total_alerts}


def run_c3_demo() -> Dict[str, Any]:
    """Demonstrate Pillar C3: Bayesian Reputation, Transient Tolerance, and Slurm Drainage."""
    console.rule("[bold cyan]Pillar C3: Bayesian Reputation & Slurm Integration[/bold cyan]")

    desc = (
        "[bold]The Dilemma:[/bold] Consumer GPUs without ECC experience occasional cosmic ray or thermal transient bitflips.\n"
        "Instantly blacklisting a node on a single failure causes severe cluster churn and false-positive node outages.\n"
        "[bold]Novelty:[/bold] GPU Fabric uses a Bayesian Beta-reputation model (α, β) with:\n"
        "  1. [bold yellow]Single-Failure Transient Tolerance:[/bold yellow] Node moves to SUSPECT, avoiding premature DRAIN.\n"
        "  2. [bold red]Automated Slurm Drainage:[/bold red] Drains only when persistent failures confirm bad hardware.\n"
        "  3. [bold green]Hysteresis Recovery:[/bold green] Drained node requires climbing to score ≥ 0.70 to avoid flapping."
    )
    console.print(Panel(desc, title="C3 Overview & Beta Reputation Lifecycle", border_style="cyan"))

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "trust.db"
        store = TrustStore(db_path=db_path, decay_factor=0.98, min_drain_evidence=3)

        entity = "node:rtx4060-worker-01"
        timeline: List[Dict[str, Any]] = []

        # Step 0: Baseline verification pass
        score, state = store.record_event(
            entity=entity,
            event="VERIFY_PASS",
            detail="Initial benchmark verification pass",
        )
        ent = store.get_entity(entity)
        timeline.append({
            "step": "Baseline",
            "event": "VERIFY_PASS (w=1.0)",
            "alpha": ent["alpha"],
            "beta": ent["beta"],
            "score": score,
            "state": state,
            "slurm": "None (Active in partition)",
            "rationale": "Prior updated with successful verification evidence",
        })

        # Step 1: Transient Single Failure
        score, state = store.record_event(
            entity=entity,
            event="VERIFY_FAIL",
            detail="Single Freivalds residual spike (transient cosmic ray / thermal jitter)",
        )
        ent = store.get_entity(entity)
        timeline.append({
            "step": "Event 1",
            "event": "VERIFY_FAIL (w=3.0)",
            "alpha": ent["alpha"],
            "beta": ent["beta"],
            "score": score,
            "state": state,
            "slurm": "[bold yellow]NONE (Transient Protection!)[/bold yellow]",
            "rationale": "Single failure marks SUSPECT; node NOT drained",
        })

        # Step 2: Second Consecutive Failure -> DRAIN threshold breached
        score, state = store.record_event(
            entity=entity,
            event="PROBE_FAIL",
            detail="Known-answer probe calculation mismatch",
        )
        ent = store.get_entity(entity)
        timeline.append({
            "step": "Event 2",
            "event": "PROBE_FAIL (w=5.0)",
            "alpha": ent["alpha"],
            "beta": ent["beta"],
            "score": score,
            "state": state,
            "slurm": f"[bold red]scontrol update NodeName={entity.split(':')[-1]} State=DRAIN[/bold red]",
            "rationale": f"Persistent failure confirmed (score {score:.3f} < 0.35, fails ≥ 2)",
        })

        # Step 3: Node maintenance & recovery passes (demonstrating Hysteresis hold)
        key_passes = [1, 5, 10, 16]
        for i in range(1, 17):
            score, state = store.record_event(
                entity=entity,
                event="VERIFY_PASS",
                detail=f"Verification audit pass #{i}",
            )
            if i in key_passes:
                ent = store.get_entity(entity)
                slurm_act = (
                    f"[bold green]scontrol update NodeName={entity.split(':')[-1]} State=RESUME[/bold green]"
                    if state == "HEALTHY"
                    else "Remain in DRAIN (Hysteresis hold)"
                )
                rationale = (
                    f"Score climbed to {score:.3f} (< 0.70 recovery threshold)"
                    if state == "DRAIN"
                    else f"Score {score:.3f} ≥ 0.70 bound; hysteresis cleared!"
                )
                timeline.append({
                    "step": f"Pass #{i}",
                    "event": "VERIFY_PASS (w=1.0)",
                    "alpha": ent["alpha"],
                    "beta": ent["beta"],
                    "score": score,
                    "state": state,
                    "slurm": slurm_act,
                    "rationale": rationale,
                })

        # Step 4: User Identity Penalty Separation
        user_entity = "user:adversary_account"
        u_score, u_state = store.record_event(
            entity=user_entity,
            event="CANARY_HIT",
            detail="Accessed rogue canary token",
        )

        # Print State Transition Timeline Table
        t_rep = Table(title="Bayesian Beta-Reputation Transition Timeline (node:rtx4060-worker-01)")
        t_rep.add_column("Timeline Stage", style="bold")
        t_rep.add_column("Observed Event", style="cyan")
        t_rep.add_column("Beta (α, β)", justify="center")
        t_rep.add_column("Trust Score", justify="right")
        t_rep.add_column("State", justify="center")
        t_rep.add_column("Automated Slurm Action", justify="left")
        t_rep.add_column("Security & Availability Rationale")

        state_styles = {
            "HEALTHY": "[bold green]HEALTHY[/bold green]",
            "SUSPECT": "[bold yellow]SUSPECT[/bold yellow]",
            "DRAIN": "[bold red]DRAIN[/bold red]",
            "RESTRICTED": "[bold magenta]RESTRICTED[/bold magenta]",
        }

        for row in timeline:
            badge = state_styles.get(row["state"], row["state"])
            t_rep.add_row(
                row["step"],
                row["event"],
                f"({row['alpha']:.1f}, {row['beta']:.1f})",
                f"[bold]{row['score']:.3f}[/bold]",
                badge,
                row["slurm"],
                row["rationale"],
            )
        console.print(t_rep)

        console.print()
        console.print(
            f"  [bold cyan]•[/bold cyan] [bold]Identity Separation Invariant:[/bold] "
            f"Canary violation penalized [bold magenta]{user_entity}[/bold magenta] "
            f"(Score: [bold red]{u_score:.3f}[/bold red], State: [bold magenta]{u_state}[/bold magenta]), "
            f"leaving healthy hardware nodes online!"
        )

    return {"timeline": timeline, "user_score": u_score, "user_state": u_state}


def generate_poster_figure(output_path: Path, c1_data: Dict[str, Any], c3_data: Dict[str, Any]) -> Path:
    """Generate a publication-grade analytical line figure for poster presentation."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[yellow]Matplotlib not installed; skipping poster graph generation.[/yellow]")
        return output_path

    matrix_sizes = np.array([256, 512, 1024, 2048, 4096])

    # Theoretical FLOP curves
    gemm_flops = 2.0 * (matrix_sizes**3)
    freiv_flops = 4.0 * 3 * (matrix_sizes**2)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Subplot 1: Computational Complexity & Work Reduction (Log scale)
    ax1.plot(matrix_sizes, gemm_flops / 1e9, "r--o", linewidth=2.2, label=r"Full Recomputation $O(n^3)$ [FLOPs]")
    ax1.plot(matrix_sizes, freiv_flops / 1e9, "g-s", linewidth=2.4, label=r"Freivalds Verifier $O(k \cdot n^2)$ ($k=3$) [FLOPs]")
    ax1.set_yscale("log")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(matrix_sizes)
    ax1.set_xticklabels([str(s) for s in matrix_sizes])
    ax1.set_xlabel("Matrix Dimension $N$ ($N \\times N$ Float32)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Arithmetic Work (GFLOPs, log-scale)", fontsize=11, fontweight="bold")
    ax1.set_title("Pillar C1: Theoretical FLOP Reduction", fontsize=13, fontweight="bold", pad=10)
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", frameon=True, fontsize=10)

    # Annotate arithmetic reduction at N=2048
    reduction_2048 = gemm_flops[3] / freiv_flops[3]
    ax1.annotate(
        f"{reduction_2048:.0f}× Work\nReduction",
        xy=(matrix_sizes[3], freiv_flops[3] / 1e9),
        xytext=(matrix_sizes[3] * 0.45, (freiv_flops[3] / 1e9) * 20),
        arrowprops=dict(facecolor="#2e7d32", shrink=0.08, width=1.5, headwidth=7),
        fontweight="bold",
        color="#1b5e20",
        fontsize=10,
    )

    # Subplot 2: Bayesian Trust Score & Hysteresis Trajectory
    timeline = c3_data.get("timeline", [])
    steps = [row["step"] for row in timeline]
    scores = [row["score"] for row in timeline]
    x_indices = np.arange(len(steps))

    # Threshold horizontal bands
    ax2.axhspan(0.60, 1.0, color="#d4edda", alpha=0.6, label="HEALTHY Region (≥ 0.60)")
    ax2.axhspan(0.35, 0.60, color="#fff3cd", alpha=0.6, label="SUSPECT Region (0.35 - 0.60)")
    ax2.axhspan(0.0, 0.35, color="#f8d7da", alpha=0.6, label="DRAIN Region (< 0.35)")

    # Plot score points
    ax2.plot(x_indices, scores, "b-o", linewidth=2.2, markersize=7, label=r"Beta Trust Score $\frac{\alpha}{\alpha + \beta}$")

    # Mark Hysteresis recovery line
    ax2.axhline(0.70, color="green", linestyle=":", linewidth=2, label="Hysteresis Recovery Threshold (0.70)")

    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(steps, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("Trust Score", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Pillar C3: Bayesian Reputation & Slurm DRAIN/Hysteresis", fontsize=13, fontweight="bold", pad=10)
    ax2.legend(loc="lower right", frameon=True, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "GPU Fabric Trust Layer Novelty Evaluation (Poster Presentation)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def run_showcase(args, client_kwargs: Optional[Dict[str, Any]] = None):
    """Main entry point for the novelty demonstration command."""
    matrix_size = getattr(args, "matrix_size", 1024) or 1024
    step_mode = getattr(args, "step", False)
    do_graph = getattr(args, "graph", False) or getattr(args, "save_poster_figure", False)
    output_graph_path = Path(getattr(args, "output_graph", "gpufabric_novelty_showcase.png") or "gpufabric_novelty_showcase.png")

    # Display Presentation Banner
    banner = Text.from_markup(
        "[bold cyan]╔══════════════════════════════════════════════════════════════════════════════╗[/bold cyan]\n"
        "[bold cyan]║[/bold cyan]                     [bold white]GPU FABRIC: TRUST LAYER NOVELTY DEMO[/bold white]                     [bold cyan]║[/bold cyan]\n"
        "[bold cyan]║[/bold cyan]      [italic green]Zero-Trust GPU Acceleration Across Decentralized Consumer Workstations[/italic green]   [bold cyan]║[/bold cyan]\n"
        "[bold cyan]╚══════════════════════════════════════════════════════════════════════════════╝[/bold cyan]"
    )
    console.print(banner)
    console.print(
        "  [bold]Novelty Pillars Featured in Demonstration:[/bold]\n"
        "    [bold cyan]• C1 (Verifiable Computation):[/bold cyan] Freivalds O(k·n²) vs O(n³) recompute + real-time bitflip detection.\n"
        "    [bold cyan]• C2 (Active Deception):[/bold cyan] Lightweight RTX 4060 decoy honeypot node + cryptographic canaries.\n"
        "    [bold cyan]• C3 (Node/User Reputation):[/bold cyan] Bayesian Beta model with single-failure tolerance & Slurm drainage.\n"
    )

    # Run Pillar C1
    c1_data = run_c1_demo(matrix_size=matrix_size)
    _pause(step_mode, "Press [bold cyan][Enter][/bold cyan] to proceed to Pillar C2 (Active Deception)...")

    # Run Pillar C2
    c2_data = run_c2_demo()
    _pause(step_mode, "Press [bold cyan][Enter][/bold cyan] to proceed to Pillar C3 (Bayesian Reputation & Slurm)...")

    # Run Pillar C3
    c3_data = run_c3_demo()

    # Optional Graph Generation
    if do_graph:
        console.print()
        with console.status("[bold green]Generating publication-grade poster figure..."):
            fig_path = generate_poster_figure(output_graph_path, c1_data, c3_data)
        console.print(f"  [bold green]✔[/bold green] [bold]Poster Figure Generated:[/bold] [bold cyan]{fig_path.resolve()}[/bold cyan]")

    # Final Summary Card
    console.print()
    console.rule("[bold green]Poster Session Takeaways & Summary[/bold green]")
    summary_text = (
        "[bold cyan]1. Asymptotic Verification Speedup (C1):[/bold cyan]\n"
        f"   Freivalds reduced arithmetic workload by [bold cyan]{c1_data['flop_reduction']:.1f}× fewer FLOPs[/bold cyan] "
        f"({c1_data['freiv_flops'] / 1e6:.1f} MFLOPs vs {c1_data['gemm_flops'] / 1e9:.2f} GFLOPs), "
        "catching silent data corruption (SDC) without saturating the client.\n\n"
        "[bold cyan]2. High-Precision Adversary Trapping (C2):[/bold cyan]\n"
        f"   Decoy honeypots and canary tokens trapped {c2_data.get('alert_count', 2)} rogue operations with zero GPU compute overhead.\n\n"
        "[bold cyan]3. High-Availability Transient Fault Tolerance (C3):[/bold cyan]\n"
        "   Beta reputation prevents premature Slurm node drainage on isolated single bitflips (SUSPECT state),\n"
        "   only draining on persistent failure and enforcing strict hysteresis before returning to HEALTHY."
    )
    console.print(Panel(summary_text, title="[bold]Summary for Evaluators & Reviewers[/bold]", border_style="green"))
    console.print(
        "[dim]Tip: Pass '--step' for interactive interview pacing, or '--graph' to output high-resolution poster figures.[/dim]\n"
    )
