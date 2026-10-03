"""Rich UI formatters for client CLI outputs."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from common.formatting import bytes_to_human
from common.gpufabric_pb2 import (
    ExecuteResponse,
    GPUInfoResponse,
    GPUStatusResponse,
    HealthResponse,
)

console = Console()


def print_health(target: str, info: HealthResponse):
    content = (
        f"[bold]Worker ID:[/bold] {info.worker_id}\n"
        f"[bold]Status:[/bold] [green]{info.status.upper()}[/green]\n"
        f"[bold]Version:[/bold] {info.version}\n"
        f"[bold]GPU Available:[/bold] {'[green]Yes[/green]' if info.gpu_available else '[red]No[/red]'}"
    )
    console.print(Panel(content, title=f"Worker @ {target}", expand=False))


def print_discovered_workers(workers, timeout: float = 3.0):
    """Format and print list of discovered LAN workers."""
    if not workers:
        console.print(
            f"[yellow]No GPU Fabric workers discovered on the local network (timeout: {timeout:.1f}s).[/yellow]\n"
            "[dim]Tip: Ensure worker machines are running on the same LAN subnet with discovery enabled, "
            "or specify an address directly (e.g. `gpufabric discover <ip>`).[/dim]"
        )
        return

    t = Table(title="GPU Fabric Workers")
    t.add_column("NAME", style="bold")
    t.add_column("ADDRESS", style="cyan")
    t.add_column("PORT", style="magenta")
    t.add_column("STATUS")

    for w in workers:
        status_str = (
            f"[green]{w.status}[/green]" if w.status == "ONLINE" else f"[red]{w.status}[/red]"
        )
        if w.status != "ONLINE" and w.error:
            status_str += f" [dim]({w.error})[/dim]"
        t.add_row(w.name, w.address, str(w.port), status_str)

    console.print(t)


def print_gpu_info(info: GPUInfoResponse):
    t = Table(title=f"GPU Information (Device {info.device_index})")
    t.add_column("Property", style="cyan")
    t.add_column("Value", style="bold green")
    t.add_row("GPU Model", info.name)
    t.add_row("Device Index", str(info.device_index))
    t.add_row("Total VRAM", f"{info.total_vram_human} ({info.total_vram_bytes:,} bytes)")
    t.add_row("Free VRAM", f"{info.free_vram_human} ({info.free_vram_bytes:,} bytes)")
    t.add_row("Used VRAM", f"{info.used_vram_human} ({info.used_vram_bytes:,} bytes)")
    t.add_row("Compute Capability", info.compute_capability or "N/A")
    t.add_row("Driver Version", info.driver_version or "N/A")
    console.print(t)


def print_status(status: GPUStatusResponse):
    t = Table(title=f"Live GPU Status (Device {status.device_index})")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", style="bold")
    t.add_row("GPU Core Utilization", f"{status.gpu_utilization_pct}%")
    t.add_row("Memory Utilization", f"{status.memory_utilization_pct}%")
    t.add_row("Free VRAM", status.free_vram_human)
    t.add_row("Used VRAM", status.used_vram_human)
    t.add_row("Temperature", f"{status.temperature_c} °C")
    t.add_row("Active Tasks", str(status.active_tasks))
    console.print(t)


def print_execution(resp: ExecuteResponse):
    t = Table(title="Execution Result")
    t.add_column("Property", style="cyan")
    t.add_column("Value", style="bold green")
    t.add_row("Task ID", resp.task_id)
    t.add_row("Workload", resp.workload_type)
    t.add_row("Status", resp.status)
    t.add_row("GPU Backend", resp.gpu_backend)
    t.add_row("Execution Time", f"{resp.execution_time_ms:.4f} ms")
    console.print(t)
    if resp.result:
        res = list(resp.result)
        if len(res) <= 10:
            console.print(f"  [bold green]Result Vector C:[/bold green] {res}")
        else:
            console.print(
                f"  [bold green]Result Vector C (first 5 & last 5):[/bold green] {res[:5]} ... {res[-5:]}"
            )


def print_compute(op: str, res, rel_error: float, verified: bool):
    t = Table(title="Compute Result")
    t.add_column("Property", style="cyan")
    t.add_column("Value", style="bold")
    t.add_row("Task ID", res.task_id)
    t.add_row("Operation", op)
    t.add_row("Device", str(res.device_index))
    t.add_row("Result shape", "x".join(map(str, res.result.shape)) or "scalar")
    t.add_row("GPU kernel time", f"{res.gpu_time_ms:.4f} ms")
    t.add_row("Worker total time", f"{res.total_time_ms:.4f} ms")
    t.add_row("Round trip time", f"{res.round_trip_ms:.4f} ms")
    t.add_row("Max relative error", f"{rel_error:.2e}")
    t.add_row(
        "Verified vs numpy",
        "[green]PASS[/green]" if verified else "[red]FAIL[/red]",
    )
    console.print(t)


def print_inventory(rows):
    """rows: list of (worker address, ListGPUsResponse or error string)."""
    t = Table(title="GPU Inventory")
    for col in ("Worker", "GPU", "Name", "VRAM free / total", "Util", "Temp", "CC", "Compute"):
        t.add_column(col)
    for target, result in rows:
        if isinstance(result, str):
            t.add_row(target, "-", f"[red]unreachable[/red] {result}", "", "", "", "", "")
            continue
        label = f"{target}\n[dim]{result.worker_id}[/dim]"
        compute = "[green]ready[/green]" if result.compute_ready else "[red]no CuPy[/red]"
        if not result.gpus:
            t.add_row(label, "-", "[yellow]no GPUs found[/yellow]", "", "", "", "", compute)
        for gpu in result.gpus:
            t.add_row(
                label,
                str(gpu.device_index),
                gpu.name,
                f"{bytes_to_human(gpu.free_vram_bytes)} / {bytes_to_human(gpu.total_vram_bytes)}",
                f"{gpu.gpu_utilization_pct}%",
                f"{gpu.temperature_c} °C",
                gpu.compute_capability or "N/A",
                compute,
            )
    console.print(t)


_STATUS_STYLE = {"PASS": "green", "FAIL": "red", "INFO": "cyan", "SKIP": "yellow"}


def print_bench_report(checks, sweep=None):
    t = Table(title="Benchmark Report")
    for col in ("Check", "Result", "Threshold", "Status"):
        t.add_column(col)
    for c in checks:
        style = _STATUS_STYLE.get(c.status, "white")
        t.add_row(c.name, c.value, c.threshold, f"[bold {style}]{c.status}[/bold {style}]")
    console.print(t)

    if sweep:
        st = Table(title="Triad Size Sweep (how much work fits on this GPU)")
        for col in ("Elements", "GPU memory used", "Bandwidth", "Note"):
            st.add_column(col)
        for n, bw, note in sweep:
            st.add_row(
                f"{n:,}",
                bytes_to_human(n * 12),
                f"{bw:.1f} GB/s" if bw is not None else "-",
                note,
            )
        console.print(st)
        largest = max((n for n, bw, _ in sweep if bw is not None), default=None)
        if largest:
            console.print(
                f"Largest triad that ran: [bold]{largest:,}[/bold] elements "
                f"({bytes_to_human(largest * 12)} of GPU memory)"
            )
        if sweep[-1][1] is not None:
            console.print("[dim]Stopped before the next size would exceed 90% of free VRAM.[/dim]")
