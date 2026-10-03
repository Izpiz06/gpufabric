"""CLI sub-command executors."""

import os
import random
import sys
from typing import List, Optional, Tuple, Union

import numpy as np

from client import bench
from client.client import GPUFabricClient, GPUFabricError
from client.formatters import (
    console,
    print_bench_report,
    print_compute,
    print_discovered_workers,
    print_execution,
    print_gpu_info,
    print_health,
    print_inventory,
    print_status,
)
from client.verify import REFERENCE, relative_error, tolerance
from common.constants import DEFAULT_DISCOVERY_TIMEOUT
from common.gpufabric_pb2 import ListGPUsResponse

WORKERS_ENV = "GPUFABRIC_WORKERS"


def cmd_discover_explicit(client: GPUFabricClient, args):
    with console.status("[bold green]Connecting to worker..."):
        info = client.discover()
    print_health(client.target, info)


def cmd_discover_lan(args, client_kwargs=None):
    timeout = getattr(args, "timeout", None) or DEFAULT_DISCOVERY_TIMEOUT
    with console.status(
        f"[bold green]Searching for GPU Fabric workers on LAN (timeout {timeout:.1f}s)..."
    ):
        workers = GPUFabricClient.discover_lan(timeout=timeout, client_kwargs=client_kwargs)
    print_discovered_workers(workers, timeout=timeout)


def cmd_discover(client: Optional[GPUFabricClient], args, client_kwargs=None):
    if client is not None:
        cmd_discover_explicit(client, args)
    else:
        cmd_discover_lan(args, client_kwargs=client_kwargs)


def cmd_gpu(client: GPUFabricClient, args):
    with console.status("[bold green]Querying GPU hardware specs..."):
        info = client.get_gpu_info(device_index=args.device)
    print_gpu_info(info)


def cmd_status(client: GPUFabricClient, args):
    with console.status("[bold green]Querying live GPU telemetry..."):
        status = client.get_status(device_index=args.device)
    print_status(status)


def cmd_execute(client: GPUFabricClient, args):
    if args.a and args.b:
        va, vb = [float(x) for x in args.a], [float(x) for x in args.b]
    elif args.size:
        console.print(f"[dim]Generating random vectors of size {args.size}...[/dim]")
        va = [random.uniform(0.0, 10.0) for _ in range(args.size)]
        vb = [random.uniform(0.0, 10.0) for _ in range(args.size)]
    else:
        va, vb = [1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]

    if len(va) != len(vb):
        console.print(f"[bold red]Error:[/bold red] len(A) ({len(va)}) != len(B) ({len(vb)})")
        sys.exit(1)

    console.print(f"[bold]Executing Workload:[/bold] C = A + B (size: {len(va)})")
    if len(va) <= 10:
        console.print(f"  [cyan]Vector A:[/cyan] {va}")
        console.print(f"  [cyan]Vector B:[/cyan] {vb}")

    with console.status("[bold green]Submitting kernel to worker GPU..."):
        resp = client.execute_vector_add(va, vb, device_index=args.device)
    print_execution(resp)


def compute_input_shapes(op: str, size: int):
    """Operand shapes used by the `compute` command for a given size."""
    if op == "triad":
        return [(size,), (size,), ()]
    if op in ("vector_add", "vector_mul", "vector_dot"):
        return [(size,), (size,)]
    return [(size, size), (size, size)]


def cmd_compute(client: GPUFabricClient, args):
    rng = np.random.default_rng(args.seed)
    dtype = np.dtype(args.dtype)
    inputs = [rng.random(shape, dtype=dtype) for shape in compute_input_shapes(args.op, args.size)]
    shapes = " , ".join("x".join(map(str, x.shape)) for x in inputs)
    console.print(f"[bold]Compute:[/bold] {args.op} on {shapes} ({dtype.name})")

    with console.status("[bold green]Running on worker GPU..."):
        res = client.compute(args.op, *inputs, device_index=args.device)

    err = relative_error(res.result, REFERENCE[args.op](*inputs))
    ok = err <= tolerance(args.op, dtype)
    print_compute(args.op, res, err, ok)
    if not ok:
        sys.exit(1)


def resolve_workers(workers: List[str]) -> List[str]:
    """Workers from the command line, else from GPUFABRIC_WORKERS (comma-separated)."""
    if workers:
        return workers
    return [w.strip() for w in os.environ.get(WORKERS_ENV, "").split(",") if w.strip()]


def collect_inventory(
    workers: List[str], **client_kwargs
) -> List[Tuple[str, Union[ListGPUsResponse, str]]]:
    """Query ListGPUs on each worker. Unreachable workers get an error string.

    client_kwargs are passed to GPUFabricClient (port, timeout, tls_dir, ...).
    """
    rows = []
    for worker in workers:
        with GPUFabricClient(host=worker, **client_kwargs) as client:
            try:
                rows.append((client.target, client.list_gpus()))
            except GPUFabricError as e:
                rows.append((client.target, str(e)))
    return rows


def cmd_ls(args, client_kwargs):
    workers = resolve_workers(args.workers)
    if not workers:
        console.print(
            f"[bold red]Error:[/bold red] no workers given. Pass addresses or set {WORKERS_ENV}"
        )
        sys.exit(2)
    with console.status("[bold green]Querying workers..."):
        rows = collect_inventory(workers, **client_kwargs)
    print_inventory(rows)
    if all(isinstance(result, str) for _, result in rows):
        sys.exit(1)


def cmd_bench(client: GPUFabricClient, args):
    inventory = client.list_gpus()
    gpu = next((g for g in inventory.gpus if g.device_index == args.device), None)
    if gpu is None:
        console.print(f"[bold red]Error:[/bold red] worker has no GPU {args.device}")
        sys.exit(1)
    console.print(f"[bold]Benchmarking[/bold] {gpu.name} (GPU {args.device}) on {client.target}")

    checks = []
    with console.status("[bold green]Measuring end-to-end network throughput..."):
        median, _, verified = bench.measure_network(
            client, args.network_size, args.network_repeats, args.device
        )
    checks.append(bench.check_network(median, args.min_network_mb_s, verified))

    triad_size = args.triad_size or bench.auto_triad_size(gpu.free_vram_bytes)
    with console.status(f"[bold green]GPU triad on {triad_size:,} elements..."):
        r = client.run_benchmark("triad", triad_size, args.repeats, args.device)
    checks.append(
        bench.check_gpu_bandwidth(r.bandwidth_gb_s, r.peak_bandwidth_gb_s, args.min_gpu_bw_pct)
    )

    with console.status(f"[bold green]Matmul {args.matmul_size}x{args.matmul_size}..."):
        r = client.run_benchmark("matmul", args.matmul_size, args.repeats, args.device)
    checks.append(bench.matmul_info(r.gflops, args.matmul_size))

    with console.status(f"[bold green]Monte Carlo pi with {args.pi_samples:,} points..."):
        r = client.run_benchmark("monte_carlo_pi", args.pi_samples, args.repeats, args.device)
    checks.append(bench.check_pi(r.pi_estimate, args.pi_samples * r.repeats))

    sweep = None
    if args.sweep:
        with console.status("[bold green]Sweeping triad size up to the VRAM limit..."):
            sweep = bench.sweep_triad(client, gpu.free_vram_bytes, args.device)

    print_bench_report(checks, sweep)
    if any(c.status == "FAIL" for c in checks):
        sys.exit(1)
