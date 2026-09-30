"""CLI interface for GPU Fabric client."""

import argparse
import random
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from client.client import (
    GPUFabricClient,
    GPUFabricConnectionError,
    GPUFabricError,
    GPUFabricWorkerError,
)
from common.protocol import DEFAULT_PORT

console = Console()


def handle_discover(client: GPUFabricClient, args):
    """Check connectivity and discover worker details."""
    with console.status("[bold green]Connecting to GPU Fabric worker..."):
        info = client.discover()

    panel_content = (
        f"[bold]Worker ID:[/bold] {info.worker_id}\n"
        f"[bold]Status:[/bold] [green]{info.status.upper()}[/green]\n"
        f"[bold]Version:[/bold] {info.version}\n"
        f"[bold]GPU Available:[/bold] {'[green]Yes[/green]' if info.gpu_available else '[red]No[/red]'}"
    )
    console.print(Panel(panel_content, title=f"Worker @ {client.base_url}", expand=False))


def handle_gpu(client: GPUFabricClient, args):
    """Retrieve and display GPU device specifications."""
    with console.status("[bold green]Querying GPU information..."):
        gpu_info = client.get_gpu_info(device_index=args.device)

    table = Table(title=f"GPU Information (Device {gpu_info.device_index})", show_header=True)
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="bold green")

    table.add_row("GPU Model", gpu_info.name)
    table.add_row("Device Index", str(gpu_info.device_index))
    table.add_row("Total VRAM", f"{gpu_info.total_vram_human} ({gpu_info.total_vram_bytes:,} bytes)")
    table.add_row("Free VRAM", f"{gpu_info.free_vram_human} ({gpu_info.free_vram_bytes:,} bytes)")
    table.add_row("Used VRAM", f"{gpu_info.used_vram_human} ({gpu_info.used_vram_bytes:,} bytes)")
    table.add_row("Compute Capability", gpu_info.compute_capability or "N/A")
    table.add_row("Driver Version", gpu_info.driver_version or "N/A")

    console.print(table)


def handle_status(client: GPUFabricClient, args):
    """Retrieve and display live GPU status and metrics."""
    with console.status("[bold green]Querying GPU status..."):
        status_info = client.get_status(device_index=args.device)

    table = Table(title=f"Live GPU Status (Device {status_info.device_index})", show_header=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="bold")

    gpu_util = f"{status_info.gpu_utilization_pct}%" if status_info.gpu_utilization_pct is not None else "N/A"
    mem_util = f"{status_info.memory_utilization_pct}%" if status_info.memory_utilization_pct is not None else "N/A"
    temp = f"{status_info.temperature_c} °C" if status_info.temperature_c is not None else "N/A"

    table.add_row("GPU Core Utilization", gpu_util)
    table.add_row("Memory Controller Utilization", mem_util)
    table.add_row("Free VRAM", status_info.free_vram_human)
    table.add_row("Used VRAM", status_info.used_vram_human)
    table.add_row("Temperature", temp)
    table.add_row("Active Tasks", str(status_info.active_tasks))

    console.print(table)


def handle_execute(client: GPUFabricClient, args):
    """Submit a vector addition workload to the worker's GPU."""
    # Determine input vectors
    if args.a and args.b:
        vector_a = [float(x) for x in args.a]
        vector_b = [float(x) for x in args.b]
    elif args.size:
        size = args.size
        console.print(f"[dim]Generating random vectors of size {size}...[/dim]")
        vector_a = [random.uniform(0.0, 10.0) for _ in range(size)]
        vector_b = [random.uniform(0.0, 10.0) for _ in range(size)]
    else:
        # Default simple test vectors
        vector_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        vector_b = [10.0, 20.0, 30.0, 40.0, 50.0]

    if len(vector_a) != len(vector_b):
        console.print(f"[bold red]Error:[/bold red] len(A) ({len(vector_a)}) != len(B) ({len(vector_b)})")
        sys.exit(1)

    console.print(f"[bold]Executing Workload:[/bold] C = A + B (vector size: {len(vector_a)})")
    if len(vector_a) <= 10:
        console.print(f"  [cyan]Vector A:[/cyan] {vector_a}")
        console.print(f"  [cyan]Vector B:[/cyan] {vector_b}")

    with console.status("[bold green]Submitting workload to worker GPU..."):
        resp = client.execute_vector_add(
            a=vector_a,
            b=vector_b,
            device_index=args.device,
        )

    table = Table(title="Execution Result", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="bold green")

    table.add_row("Task ID", resp.task_id)
    table.add_row("Workload", resp.workload_type)
    table.add_row("Status", resp.status)
    table.add_row("Device Index", str(resp.device_index))
    table.add_row("GPU Backend", resp.gpu_backend or "Unknown")
    table.add_row("Execution Time", f"{resp.execution_time_ms:.4f} ms")

    console.print(table)

    if resp.result:
        if len(resp.result) <= 10:
            console.print(f"  [bold green]Result Vector C:[/bold green] {resp.result}")
        else:
            console.print(
                f"  [bold green]Result Vector C (first 5 & last 5):[/bold green] "
                f"{resp.result[:5]} ... {resp.result[-5:]}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpufabric-client",
        description="GPU Fabric Client CLI - Control remote GPUs over LAN.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Target worker port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTP request timeout in seconds (default: 15.0)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # discover
    p_discover = subparsers.add_parser("discover", help="Discover/connect to a GPU Fabric worker")
    p_discover.add_argument("worker_ip", type=str, help="IP or hostname of the worker")

    # gpu
    p_gpu = subparsers.add_parser("gpu", help="Query worker GPU hardware specs")
    p_gpu.add_argument("worker_ip", type=str, help="IP or hostname of the worker")
    p_gpu.add_argument("--device", type=int, default=0, help="Device index (default: 0)")

    # status
    p_status = subparsers.add_parser("status", help="Query live worker GPU status and utilization")
    p_status.add_argument("worker_ip", type=str, help="IP or hostname of the worker")
    p_status.add_argument("--device", type=int, default=0, help="Device index (default: 0)")

    # execute
    p_exec = subparsers.add_parser("execute", help="Execute vector addition C = A + B on worker's GPU")
    p_exec.add_argument("worker_ip", type=str, help="IP or hostname of the worker")
    p_exec.add_argument("--device", type=int, default=0, help="Device index (default: 0)")
    p_exec.add_argument("--a", nargs="+", type=float, help="Values for Vector A (e.g. --a 1.0 2.0 3.0)")
    p_exec.add_argument("--b", nargs="+", type=float, help="Values for Vector B (e.g. --b 4.0 5.0 6.0)")
    p_exec.add_argument("--size", type=int, help="Generate random test vectors of this size")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    client = GPUFabricClient(
        host=args.worker_ip,
        port=args.port,
        timeout=args.timeout,
    )

    try:
        if args.command == "discover":
            handle_discover(client, args)
        elif args.command == "gpu":
            handle_gpu(client, args)
        elif args.command == "status":
            handle_status(client, args)
        elif args.command == "execute":
            handle_execute(client, args)
    except GPUFabricConnectionError as e:
        console.print(f"[bold red]Connection Error:[/bold red] {e}")
        sys.exit(1)
    except GPUFabricWorkerError as e:
        console.print(f"[bold red]Worker Error:[/bold red] {e}")
        sys.exit(1)
    except GPUFabricError as e:
        console.print(f"[bold red]GPU Fabric Error:[/bold red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Unexpected Error:[/bold red] {e}")
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
