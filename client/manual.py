# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Interactive CLI user manual, reference guides, and first-time setup assistant."""

import platform
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from common.tls import default_tls_dir

console = Console()

# -----------------------------------------------------------------------------
# Manual Chapters Content (Markdown)
# -----------------------------------------------------------------------------

MANUAL_OVERVIEW = """
# ⚡ GPU Fabric: User Manual & Getting Started

**GPU Fabric** is a lightweight, high-performance distributed GPU runtime for local networks.
It connects machines with NVIDIA GPUs into an on-demand compute fabric via gRPC, protected by
mutual TLS (mTLS) and verified with an autonomous Trust Layer.

### 📚 Manual Sections Available
You can read any section directly using `gpufabric manual <section>`:

- **`setup`** — Complete step-by-step 1st time setup guide (Worker & Client)
- **`worker`** — Setting up and operating GPU worker nodes
- **`client`** — Using the client CLI, Python SDK, and environment variables
- **`commands`** — Comprehensive command-by-command reference and examples
- **`mtls`** — Mutual TLS certificates, network SANs, and client enrollment
- **`trust`** — Verifiable results (Freivalds), decoys, canaries, and reputation
- **`troubleshoot`** — Diagnosing common network, CUDA, and certificate errors
- **`interactive`** — Automated environment diagnostic & interactive setup wizard
"""

MANUAL_SETUP = """
# 🚀 First-Time Setup Guide

Follow this guide to get your first GPU Fabric cluster up and running in minutes.

---

### Step 0: Prerequisites
- **Python**: 3.10 or newer on all machines (`python3 --version`).
- **Worker Machine (GPU)**: Linux or Windows with NVIDIA GPU, modern drivers (CUDA 12+), and CuPy:
  ```bash
  pip install -r requirements.txt
  pip install -e .
  pip install cupy-cuda12x   # Or cupy-cuda13x depending on CUDA driver
  ```
- **Client Machine (Submitter)**: Any OS (Linux, macOS, Windows) with Python:
  ```bash
  pip install -r requirements.txt
  pip install -e .
  ```

---

### Step 1: Initialize Mutual TLS Security (Recommended)
GPU Fabric encrypts all gRPC traffic and restricts compute access using mutual TLS (mTLS).

1. **On the Worker Machine**, create your private Certificate Authority (CA) and server certificate:
   ```bash
   gpufabric-certs init
   ```
   *Tip: This automatically detects your local LAN IP addresses (e.g. `192.168.1.50`) and adds them as Subject Alternative Names (SANs).*

2. **Generate a Certificate for Your Client Machine**:
   ```bash
   gpufabric-certs add-client my-laptop
   ```
   This generates a bundle in `~/.config/gpufabric/tls/clients/my-laptop/`.

3. **Copy the Client Certificate Bundle to Your Client**:
   Copy the 3 files (`ca.crt`, `client.crt`, `client.key`) into `~/.config/gpufabric/tls/` on your client laptop.

*(For quick local testing on a single machine without TLS, you can pass `--insecure` to worker and client).*

---

### Step 2: Start the GPU Worker Node
On the machine with the NVIDIA GPU:

```bash
# Start in the foreground:
gpufabric-worker

# Or specify custom host/port:
gpufabric-worker --host 0.0.0.0 --port 50051

# Or run in the background:
nohup gpufabric-worker > worker.log 2>&1 &
```

You can verify the worker is healthy locally by launching the Host TUI:
```bash
gpufabric host
```

---

### Step 3: Discover & Inspect from the Client Machine
On your client machine:

1. **Discover workers on the local LAN**:
   ```bash
   gpufabric discover
   ```

2. **Check worker health and GPU specifications**:
   ```bash
   gpufabric health 192.168.1.50
   gpufabric gpu 192.168.1.50
   gpufabric status 192.168.1.50
   ```

---

### Step 4: Run Your First Remote GPU Workload
Execute a verified matrix multiplication on the remote GPU:

```bash
gpufabric compute 192.168.1.50 matmul --size 2000 --verify fast
```

This sends two 2000x2000 float32 matrices to the remote GPU, executes the matrix multiplication
directly on physical VRAM via CuPy, returns the result, and verifies it in $O(k n^2)$ time
using Freivalds' randomized algorithm!

---

### Step 5: Launch the Interactive Client TUI
To monitor and manage all cluster nodes in an interactive full-screen dashboard:

```bash
gpufabric
```

From the dashboard:
- Press **b** to open the remote GPU Benchmark modal.
- Press **Enter** on any worker to view full hardware telemetry.
- Press **d** to scan the LAN for new worker nodes.
"""

MANUAL_WORKER = """
# 🖥️ GPU Worker Guide

The worker is a gRPC service that exposes NVIDIA GPUs for remote execution. Workloads run directly
on physical GPU memory through CuPy and NVML.

### Starting the Worker
```bash
# Standard startup (reads certs from ~/.config/gpufabric/tls)
gpufabric-worker

# Custom bind address and port:
gpufabric-worker --host 0.0.0.0 --port 50051

# Insecure mode (No TLS, for isolated local testing only):
gpufabric-worker --insecure

# Run with interactive Host monitoring dashboard:
gpufabric-worker --tui
```

### Trust Layer & Security Flags
```bash
# Launch a decoy worker node on a secondary port (traps unauthorized probes):
gpufabric-worker --decoy-port 50052

# Fault injection testing (simulates non-ECC bit flips to test client detection):
gpufabric-worker --fault-rate 0.05 --fault-magnitude 0.1
```

### Running as a Systemd Service (Linux)
Create `/etc/systemd/system/gpufabric-worker.service`:
```ini
[Unit]
Description=GPU Fabric Worker Daemon
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/youruser
ExecStart=/home/youruser/.local/bin/gpufabric-worker --host 0.0.0.0 --port 50051
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```
Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now gpufabric-worker
```
"""

MANUAL_CLIENT = """
# 💻 Client CLI & SDK Guide

The client communicates with workers to query hardware, stream telemetry, dispatch tensor workloads,
and run benchmarks.

### Environment Variables
- `GPUFABRIC_WORKERS`: Comma-separated list of default workers (e.g. `192.168.1.50:50051,192.168.1.51:50051`).
  When set, `gpufabric ls` automatically queries all configured nodes without typing addresses!
- `GPUFABRIC_TLS_DIR`: Custom directory holding `ca.crt`, `client.crt`, `client.key`.
  Default: `~/.config/gpufabric/tls/`.

### Python SDK Quickstart
You can integrate GPU Fabric into your own Python scripts:

```python
import numpy as np
from client.client import GPUFabricClient

# Connect to worker over mTLS
with GPUFabricClient(host="192.168.1.50", port=50051) as client:
    # 1. Inspect hardware
    info = client.get_gpu_info(device_index=0)
    print(f"Connected to {info.name} ({info.free_vram_human} free)")

    # 2. Dispatch matrix multiplication
    a = np.random.randn(1024, 1024).astype(np.float32)
    b = np.random.randn(1024, 1024).astype(np.float32)

    result = client.compute("matmul", a, b, device_index=0)
    print(f"Computed in {result.execution_time_ms:.2f} ms on remote GPU!")

    # 3. Verify computation with Freivalds check
    from client.verify import verify_result
    report = verify_result("matmul", [a, b], result.data, mode="fast")
    print(f"Trust Layer check: {report.status} (residual: {report.residual:.2e})")
```
"""

MANUAL_COMMANDS = """
# 📖 CLI Command Reference

| Command | Description | Example |
|---|---|---|
| `gpufabric` | Launch interactive multi-node Client TUI | `gpufabric` |
| `gpufabric host` | Launch standalone local Host monitoring TUI | `gpufabric host` |
| `gpufabric discover` | Scan local LAN subnet for worker nodes | `gpufabric discover` |
| `gpufabric ls` | List every GPU on workers or `$GPUFABRIC_WORKERS` | `gpufabric ls 192.168.1.50` |
| `gpufabric health` | Inspect worker health state, CUDA & driver versions | `gpufabric health 192.168.1.50` |
| `gpufabric gpu` | Query detailed GPU specs, VRAM size & compute capability | `gpufabric gpu 192.168.1.50 --device 0` |
| `gpufabric status` | Live snapshot of GPU utilization, VRAM usage & power draw | `gpufabric status 192.168.1.50` |
| `gpufabric compute` | Dispatch remote tensor operation with verification | `gpufabric compute 192.168.1.50 matmul --size 2000 --verify fast` |
| `gpufabric bench` | Run remote STREAM triad, matmul GFLOPS & Monte Carlo Pi | `gpufabric bench 192.168.1.50` |
| `gpufabric trust show` | Inspect Beta-reputation trust score for nodes/users | `gpufabric trust show node:worker-01` |
| `gpufabric trust events` | Audit chronological trust and fault events | `gpufabric trust events --limit 20` |
| `gpufabric alerts tail` | Stream security alerts (canaries, decoys, failed checks) | `gpufabric alerts tail` |
| `gpufabric-certs init` | Create mTLS Certificate Authority & worker certs | `gpufabric-certs init` |
| `gpufabric-certs add-client` | Issue client certificate for another machine | `gpufabric-certs add-client laptop` |
| `gpufabric manual` | Open user manual or specific chapter | `gpufabric manual setup` |
"""

MANUAL_MTLS = """
# 🔐 Mutual TLS Security & Certificates

GPU Fabric uses mutual TLS (mTLS) so that:
1. **The Client verifies the Worker**: Protects against MITM attacks and spoofed nodes.
2. **The Worker verifies the Client**: Restricts GPU compute access only to devices with certificates signed by your CA.

### Certificate Hierarchy
```
~/.config/gpufabric/tls/
├── ca.crt         # Certificate Authority (shared with all nodes)
├── ca.key         # Private CA signing key (keep secret on worker!)
├── server.crt     # Worker node certificate
├── server.key     # Worker private key
├── client.crt     # Local client certificate
├── client.key     # Local client private key
└── clients/       # Issued client bundles
    └── laptop/
        ├── ca.crt
        ├── client.crt
        └── client.key
```

### SAN (Subject Alternative Names)
When `gpufabric-certs init` runs, it inspects all local network interfaces and binds them into the certificate:
```bash
gpufabric-certs init --hosts 192.168.1.50,my-workstation.local
```
If you change your worker IP, re-run `gpufabric-certs init --force --hosts <new-ips>` to update the certificate SANs.
"""

MANUAL_TRUST = r"""
# 🛡️ Trust Layer Architecture

Non-dedicated lab PCs and workstations often feature consumer GPUs (e.g. RTX 3060/4060) without ECC memory,
and operate in zero-trust environments. The Trust Layer provides 3 critical protections:

### 1. Verifiable Results ($O(k n^2)$ Freivalds Check)
Recomputing a matrix multiplication on the client with CPU NumPy takes $O(n^3)$ time and defeats the purpose of GPU offloading.
GPU Fabric implements **Freivalds' algorithm**:
- For random Rademacher vectors $r \in \{-1, +1\}^n$:
  $$\text{Verify: } C \cdot r \stackrel{?}{=} A \cdot (B \cdot r)$$
- Verifies in $O(k n^2)$ without ever reconstructing $A \times B$.
- Bound on error detection probability: $\ge 1 - 2^{-k}$ ($99.9\%$ confidence in 10 rounds).
- Use `--verify fast` on `gpufabric compute`.

### 2. Beta-Reputation Model
- Tracks Bayesian belief $\operatorname{Beta}(\alpha, \beta)$ for every worker node and user.
- Honest computations add successful evidence ($\alpha$), while verification failures add penalty evidence ($\beta$).
- When trust falls below $0.70$, the node can automatically trigger a **Slurm drain** (`scontrol update node=... state=DRAIN`).

### 3. Deception & Honeypots
- **Decoy Workers**: Emulated worker nodes running on auxiliary ports that alert instantly on scan or contact.
- **Canary Tokens**: Unique validation jobs dispatched periodically to detect silent corruption or unauthorized access.
"""

MANUAL_TROUBLESHOOT = """
# 🩺 Troubleshooting Guide

### 1. Connection Refused (`[Errno 111]`)
- **Cause**: The worker process is not running or a firewall is blocking port 50051.
- **Fix**: Check `gpufabric-worker` is active on the worker machine.
  Allow port 50051 through firewall:
  ```bash
  # Linux (ufw)
  sudo ufw allow 50051/tcp
  ```

### 2. TLS SAN Mismatch / Unknown Authority
- **Cause**: Connecting via an IP address not listed in `server.crt` SANs, or client doesn't trust `ca.crt`.
- **Fix**: Re-initialize certificates on worker including the correct IP:
  ```bash
  gpufabric-certs init --force --hosts <worker-ip>,localhost
  ```
  Ensure client machine has `ca.crt` in `~/.config/gpufabric/tls/`.

### 3. cuBLAS Library Missing
- **Cause**: NVIDIA CUDA runtime libraries are missing from Python's path.
- **Fix**: Install required CUDA runtime packages via pip:
  ```bash
  pip install nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cublas-cu12
  ```

### 4. GPU Not Detected / NVML Error
- **Cause**: NVIDIA driver is not installed or permissions are restricted.
- **Fix**: Test with `nvidia-smi`. Ensure the user is in the `video` / `render` group on Linux.
"""

SECTIONS: Dict[str, str] = {
    "overview": MANUAL_OVERVIEW,
    "setup": MANUAL_SETUP,
    "worker": MANUAL_WORKER,
    "client": MANUAL_CLIENT,
    "commands": MANUAL_COMMANDS,
    "mtls": MANUAL_MTLS,
    "trust": MANUAL_TRUST,
    "troubleshoot": MANUAL_TROUBLESHOOT,
}

# -----------------------------------------------------------------------------
# Environment Diagnostic
# -----------------------------------------------------------------------------


def get_diagnostic_info() -> Dict[str, Any]:
    """Collect system, GPU, CUDA, and TLS status for first-time setup diagnostics."""
    try:
        tls_dir = default_tls_dir()
    except Exception:
        tls_dir = Path("~/.config/gpufabric/tls").expanduser()

    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "nvidia_available": False,
        "gpu_count": 0,
        "gpu_names": [],
        "driver_version": "N/A",
        "cupy_available": False,
        "cupy_version": "N/A",
        "tls_dir": str(tls_dir),
        "tls_initialized": False,
        "tls_files": [],
        "has_ca": False,
        "has_server": False,
        "has_client": False,
    }

    # NVML / NVIDIA Driver probe
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pynvml

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            info["nvidia_available"] = count > 0
            info["gpu_count"] = count
            names = []
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                names.append(name)
            info["gpu_names"] = names
            try:
                drv = pynvml.nvmlSystemGetDriverVersion()
                info["driver_version"] = (
                    drv.decode("utf-8", errors="replace") if isinstance(drv, bytes) else str(drv)
                )
            except Exception:
                pass
    except Exception:
        info["nvidia_available"] = False

    # CuPy probe
    try:
        import cupy

        info["cupy_available"] = True
        info["cupy_version"] = getattr(cupy, "__version__", "installed")
    except Exception:
        info["cupy_available"] = False

    # TLS directory probe
    try:
        tls_path = default_tls_dir()
        if tls_path.exists():
            files = sorted([f.name for f in tls_path.iterdir() if f.is_file()])
            info["tls_files"] = files
            info["has_ca"] = "ca.crt" in files
            info["has_server"] = "server.crt" in files and "server.key" in files
            info["has_client"] = "client.crt" in files and "client.key" in files
            info["tls_initialized"] = info["has_ca"] and (info["has_server"] or info["has_client"])
    except Exception:
        pass

    return info


def print_diagnostic_panel(diag: Dict[str, Any], console_out: Optional[Console] = None) -> None:
    """Print an environment diagnostics summary table."""
    c = console_out or console

    table = Table(title="🔍 Local Environment Diagnostics", expand=True)
    table.add_column("Component", style="cyan", width=26)
    table.add_column("Status / Details", style="bold")
    table.add_column("Role Readiness", style="dim", width=22)

    # Python
    py_ok = sys.version_info >= (3, 10)
    table.add_row(
        "Python Runtime",
        f"[green]v{diag['python']}[/green]"
        if py_ok
        else f"[red]v{diag['python']} (requires 3.10+)[/red]",
        "[green]✔ Ready[/green]" if py_ok else "[red]✘ Upgrade Required[/red]",
    )

    # Platform
    table.add_row("Operating System", diag["platform"], "[green]✔ Supported[/green]")

    # NVIDIA GPU
    if diag["nvidia_available"]:
        gpus_desc = ", ".join(diag["gpu_names"])
        table.add_row(
            f"NVIDIA GPU ({diag['gpu_count']} found)",
            f"[green]{gpus_desc}[/green] (Driver {diag['driver_version']})",
            "[green]✔ Worker Host Ready[/green]",
        )
    else:
        table.add_row(
            "NVIDIA GPU",
            "[yellow]No physical GPU detected or driver unavailable[/yellow]",
            "[dim]Client-Only Mode[/dim]",
        )

    # CuPy
    if diag["cupy_available"]:
        table.add_row(
            "CuPy Acceleration",
            f"[green]Installed (v{diag['cupy_version']})[/green]",
            "[green]✔ Kernel Compute Ready[/green]",
        )
    else:
        status_note = (
            "[yellow]Not installed (needed for Worker node)[/yellow]"
            if diag["nvidia_available"]
            else "[dim]Not required for Client[/dim]"
        )
        table.add_row("CuPy Acceleration", status_note, "[dim]Optional on Client[/dim]")

    # Mutual TLS
    if diag["tls_initialized"]:
        parts = []
        if diag["has_ca"]:
            parts.append("CA")
        if diag["has_server"]:
            parts.append("Server")
        if diag["has_client"]:
            parts.append("Client")
        certs_str = ", ".join(parts)
        table.add_row(
            "mTLS Certificates",
            f"[green]Initialized ({certs_str}) in {diag['tls_dir']}[/green]",
            "[green]✔ Secured[/green]",
        )
    else:
        table.add_row(
            "mTLS Certificates",
            f"[yellow]Not initialized in {diag['tls_dir']}[/yellow]",
            "[yellow]Setup Required[/yellow]",
        )

    c.print(table)


# -----------------------------------------------------------------------------
# Interactive Setup Assistant
# -----------------------------------------------------------------------------


def run_interactive_setup(console_out: Optional[Console] = None) -> None:
    """Run interactive step-by-step 1st time setup assistant with diagnostics."""
    c = console_out or console

    c.print(
        Panel(
            "[bold cyan]⚡ GPU Fabric Interactive Setup Assistant[/bold cyan]\n"
            "This wizard inspects your machine and guides you through setting up GPU Fabric.",
            border_style="cyan",
        )
    )

    diag = get_diagnostic_info()
    print_diagnostic_panel(diag, c)

    c.print("\n[bold]What role would you like to configure for this machine?[/bold]")
    c.print(
        "  [bold cyan]1[/bold cyan]) [bold]GPU Worker[/bold] (This machine has an NVIDIA GPU to share across LAN)"
    )
    c.print(
        "  [bold cyan]2[/bold cyan]) [bold]Client Machine[/bold] (Submit workloads and monitor remote GPUs)"
    )
    c.print(
        "  [bold cyan]3[/bold cyan]) [bold]Standalone / Local Test[/bold] (Run worker and client together on this machine)"
    )

    choice = Prompt.ask("Select an option", choices=["1", "2", "3"], default="3")

    if choice == "1":
        _setup_worker_role(diag, c)
    elif choice == "2":
        _setup_client_role(diag, c)
    else:
        _setup_standalone_role(diag, c)


def _setup_worker_role(diag: Dict[str, Any], c: Console) -> None:
    c.print("\n[bold green]=== Configuring as GPU Worker ===[/bold green]")

    # Check GPU
    if not diag["nvidia_available"]:
        c.print(
            "[bold red]Warning:[/bold red] No NVIDIA GPU or driver was detected. "
            "Ensure NVIDIA drivers and CUDA toolkit are installed before running compute workloads."
        )

    # Check CuPy
    if not diag["cupy_available"]:
        c.print(
            "[bold yellow]Action Required:[/bold yellow] CuPy is required for GPU kernel execution.\n"
            "Install CuPy matching your CUDA driver version:\n"
            "  [bold cyan]pip install cupy-cuda12x[/bold cyan]   # For CUDA 12.x\n"
            "  [bold cyan]pip install cupy-cuda13x[/bold cyan]   # For CUDA 13.x\n"
        )

    # Check TLS
    if not diag["has_server"]:
        c.print("[bold yellow]mTLS Certificates Needed:[/bold yellow]")
        if Confirm.ask(
            "Would you like to initialize certificates now with `gpufabric-certs init`?",
            default=True,
        ):
            try:
                from common.certs_cli import main as certs_main

                c.print("[cyan]Running certificate initialization...[/cyan]")
                ret = certs_main(["init"])
                if ret == 0 or ret is None:
                    c.print("[green]✔ Certificates successfully initialized![/green]")
            except Exception as e:
                c.print(f"[red]Failed to initialize certs automatically: {e}[/red]")
                c.print("Run manually: [bold cyan]gpufabric-certs init[/bold cyan]")
    else:
        c.print("[green]✔ mTLS server certificates are already configured.[/green]")

    c.print(
        Panel(
            "[bold]Next Steps to Start Your Worker:[/bold]\n\n"
            "1. [bold]Start the Worker Node:[/bold]\n"
            "   [cyan]gpufabric-worker[/cyan]\n\n"
            "2. [bold]Issue Certificates for Client Machines:[/bold]\n"
            "   [cyan]gpufabric-certs add-client <client-name>[/cyan]\n"
            "   Copy the generated files from [cyan]~/.config/gpufabric/tls/clients/<name>/[/cyan]\n"
            "   to the client's [cyan]~/.config/gpufabric/tls/[/cyan] directory.\n\n"
            "3. [bold]Monitor Local GPU Status:[/bold]\n"
            "   [cyan]gpufabric host[/cyan]",
            title="Worker Setup Summary",
            border_style="green",
        )
    )


def _setup_client_role(diag: Dict[str, Any], c: Console) -> None:
    c.print("\n[bold cyan]=== Configuring as GPU Fabric Client ===[/bold cyan]")

    if not diag["has_client"]:
        c.print(
            Panel(
                "[bold yellow]Step 1: Obtain mTLS Client Certificate Bundle[/bold yellow]\n\n"
                "To connect securely to a remote worker, you need a client certificate signed by the worker's CA.\n\n"
                "1. On the worker machine, run:\n"
                "   [bold cyan]gpufabric-certs add-client my-laptop[/bold cyan]\n\n"
                "2. Copy the 3 generated files ([cyan]ca.crt[/cyan], [cyan]client.crt[/cyan], [cyan]client.key[/cyan])\n"
                f"   into your local directory: [bold cyan]{diag['tls_dir']}[/bold cyan]\n\n"
                "*(Alternatively, if the worker was started with [dim]--insecure[/dim], use [dim]gpufabric --insecure[/dim])*",
                border_style="yellow",
            )
        )
    else:
        c.print("[green]✔ mTLS client certificates are found and ready.[/green]")

    c.print(
        Panel(
            "[bold]Step 2: Connect & Verify Workloads:[/bold]\n\n"
            "1. [bold]Discover Workers on your LAN:[/bold]\n"
            "   [cyan]gpufabric discover[/cyan]\n\n"
            "2. [bold]Inspect Remote GPU Specs:[/bold]\n"
            "   [cyan]gpufabric gpu <worker-ip>[/cyan]\n\n"
            "3. [bold]Run Verified Matrix Multiplication:[/bold]\n"
            "   [cyan]gpufabric compute <worker-ip> matmul --size 2000 --verify fast[/cyan]\n\n"
            "4. [bold]Launch the Interactive Dashboard:[/bold]\n"
            "   [cyan]gpufabric[/cyan]",
            title="Client Setup Summary",
            border_style="cyan",
        )
    )


def _setup_standalone_role(diag: Dict[str, Any], c: Console) -> None:
    c.print("\n[bold magenta]=== Configuring Standalone / Local Test Mode ===[/bold magenta]")

    if not diag["tls_initialized"]:
        if Confirm.ask("Initialize local mTLS certificates now?", default=True):
            try:
                from common.certs_cli import main as certs_main

                certs_main(["init"])
                c.print("[green]✔ Local certificates initialized![/green]")
            except Exception as e:
                c.print(f"[red]Error initializing certs: {e}[/red]")
                c.print("Run manually: [bold cyan]gpufabric-certs init[/bold cyan]")

    c.print(
        Panel(
            "[bold]Quickstart Instructions (Two Terminals):[/bold]\n\n"
            "[bold]Terminal 1 (Worker Node):[/bold]\n"
            "  [cyan]gpufabric-worker[/cyan]\n\n"
            "[bold]Terminal 2 (Client Commands):[/bold]\n"
            "  [cyan]gpufabric health 127.0.0.1[/cyan]\n"
            "  [cyan]gpufabric compute 127.0.0.1 matmul --size 1000 --verify fast[/cyan]\n"
            "  [cyan]gpufabric[/cyan]  (Launches interactive TUI dashboard)",
            title="Standalone Setup Summary",
            border_style="magenta",
        )
    )


# -----------------------------------------------------------------------------
# Main Manual Dispatcher
# -----------------------------------------------------------------------------


def show_manual(
    section: Optional[str] = None,
    interactive: bool = False,
    console_out: Optional[Console] = None,
) -> None:
    """Display user manual section or launch interactive setup assistant."""
    c = console_out or console

    if interactive or section == "interactive":
        run_interactive_setup(c)
        return

    if section == "all":
        for sec_name, content in SECTIONS.items():
            c.print(Markdown(content))
            c.print("\n" + "=" * 70 + "\n")
        return

    if section and section in SECTIONS:
        c.print(Markdown(SECTIONS[section]))
        return

    if section and section not in SECTIONS:
        c.print(f"[bold red]Unknown manual section:[/bold red] '{section}'")
        c.print(f"Available sections: {', '.join(sorted(SECTIONS.keys()))}, all, interactive")
        c.print("Try: [bold cyan]gpufabric manual setup[/bold cyan]")
        return

    # Default overview & interactive table of contents
    c.print(Markdown(MANUAL_OVERVIEW))

    t = Table(title="📖 Manual Chapters & Navigation", expand=True)
    t.add_column("Chapter", style="cyan", width=16)
    t.add_column("Command", style="bold green", width=30)
    t.add_column("Description")

    t.add_row(
        "First-Time Setup",
        "gpufabric manual setup",
        "Step-by-step walkthrough for worker nodes and clients",
    )
    t.add_row(
        "Interactive Setup",
        "gpufabric manual -i",
        "Automated hardware/cert diagnostics & setup wizard",
    )
    t.add_row(
        "Commands Reference",
        "gpufabric manual commands",
        "Complete reference for all CLI subcommands & flags",
    )
    t.add_row(
        "Worker Guide",
        "gpufabric manual worker",
        "Starting worker service, systemd daemons, and options",
    )
    t.add_row(
        "Client & SDK",
        "gpufabric manual client",
        "Python SDK quickstart and environment variables",
    )
    t.add_row(
        "mTLS Security",
        "gpufabric manual mtls",
        "Certificate hierarchy, SAN addresses & secure enrollment",
    )
    t.add_row(
        "Trust Layer",
        "gpufabric manual trust",
        "Freivalds verification, Beta reputation, decoys & canaries",
    )
    t.add_row(
        "Troubleshooting",
        "gpufabric manual troubleshoot",
        "Common errors, CUDA/cuBLAS fixes & network diagnostics",
    )

    c.print(t)
    c.print(
        "\n[dim]Tip: Run `gpufabric manual <chapter>` or `gpufabric manual -i` for guided setup.[/dim]\n"
    )
