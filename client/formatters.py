"""Rich UI formatters for client CLI outputs."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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
