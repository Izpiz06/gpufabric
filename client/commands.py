"""CLI sub-command executors."""

import random
import sys

import numpy as np

from client.client import GPUFabricClient
from client.formatters import (
    console,
    print_compute,
    print_execution,
    print_gpu_info,
    print_health,
    print_status,
)
from client.verify import REFERENCE, relative_error, tolerance


def cmd_discover(client: GPUFabricClient, args):
    with console.status("[bold green]Connecting to worker..."):
        info = client.discover()
    print_health(client.target, info)


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
