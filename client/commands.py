# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

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
from client.verify import verify_result
from common.constants import DEFAULT_DISCOVERY_TIMEOUT
from common.gpufabric_pb2 import ListGPUsResponse

WORKERS_ENV = "GPUFABRIC_WORKERS"


def cmd_health(client: GPUFabricClient, args):
    with console.status("[bold green]Checking worker health..."):
        info = client.health()
    print_health(client.target, info)


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

    # If mDNS discovered 0 workers, also probe any workers configured in GPUFABRIC_WORKERS
    if not workers and os.environ.get(WORKERS_ENV):
        env_workers = [w.strip() for w in os.environ[WORKERS_ENV].split(",") if w.strip()]
        if env_workers:
            from common.constants import DEFAULT_PORT
            from common.discovery import DiscoveredWorker

            console.print(
                f"[dim]mDNS returned 0 workers. Probing configured ${WORKERS_ENV} ({', '.join(env_workers)})...[/dim]"
            )
            for w in env_workers:
                target = w if ":" in w else f"{w}:{DEFAULT_PORT}"
                h, p = target.split(":", 1)
                kw = dict(client_kwargs or {})
                kw.pop("port", None)
                try:
                    with GPUFabricClient(host=h, port=int(p), **kw) as c:
                        health = c.health()
                        workers.append(
                            DiscoveredWorker(
                                name=health.hostname or h,
                                address=h,
                                port=int(p),
                                worker_id=health.worker_id or h,
                                version=health.version,
                                status="ONLINE" if health.status == "ok" else health.status.upper(),
                                gpu_available=health.gpu_available,
                            )
                        )
                except Exception as e:
                    workers.append(
                        DiscoveredWorker(
                            name=h,
                            address=h,
                            port=int(p),
                            worker_id=h,
                            status="UNREACHABLE",
                            error=str(e),
                        )
                    )

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

    verify_mode = getattr(args, "verify", None)
    if verify_mode is None:
        verify_mode = "fast" if (args.op == "matmul" and args.size >= 512) else "full"

    rounds = getattr(args, "rounds", 3)
    report = verify_result(args.op, inputs, res.result, mode=verify_mode, rounds=rounds)
    print_compute(args.op, res, report=report)
    if not report.passed:
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


def cmd_trust(args):
    from rich.table import Table

    from common.trust import get_default_trust_store

    store = get_default_trust_store()
    if args.trust_action == "show":
        if args.entity:
            item = store.get_entity(args.entity)
            if not item:
                console.print(f"[yellow]Entity '{args.entity}' not found in trust store.[/yellow]")
                return
            t = Table(title=f"Trust Reputation: {args.entity}")
            t.add_column("Property", style="cyan")
            t.add_column("Value", style="bold")
            t.add_row("Entity ID", item["entity_id"])
            t.add_row("Entity Type", item["entity_type"])
            t.add_row("Trust Score", f"{item['score']:.4f}")
            state_style = (
                "green"
                if item["state"] == "HEALTHY"
                else ("yellow" if item["state"] == "SUSPECT" else "red")
            )
            t.add_row("State", f"[{state_style}]{item['state']}[/{state_style}]")
            t.add_row("Alpha (Passes)", f"{item['alpha']:.2f}")
            t.add_row("Beta (Fails)", f"{item['beta']:.2f}")
            t.add_row("Total Events", str(item["total_events"]))
            t.add_row("Fail Count", str(item["fail_count"]))
            t.add_row("Consecutive Passes", str(item["consecutive_passes"]))
            t.add_row("Last Event", item["last_event_at"] or "N/A")
            console.print(t)
        else:
            items = store.list_entities()
            if not items:
                console.print("[dim]No entities recorded in trust store yet.[/dim]")
                return
            t = Table(title="Trust Store Entities")
            t.add_column("Entity ID", style="cyan")
            t.add_column("Type")
            t.add_column("Score", style="bold")
            t.add_column("State")
            t.add_column("Alpha/Beta")
            t.add_column("Events")
            t.add_column("Last Event")
            for it in items:
                state_style = (
                    "green"
                    if it["state"] == "HEALTHY"
                    else ("yellow" if it["state"] == "SUSPECT" else "red")
                )
                t.add_row(
                    it["entity_id"],
                    it["entity_type"],
                    f"{it['score']:.4f}",
                    f"[{state_style}]{it['state']}[/{state_style}]",
                    f"{it['alpha']:.1f} / {it['beta']:.1f}",
                    str(it["total_events"]),
                    it["last_event_at"] or "N/A",
                )
            console.print(t)

    elif args.trust_action == "events":
        events = store.get_events(args.entity, limit=args.limit)
        if not events:
            console.print("[dim]No trust events recorded yet.[/dim]")
            return
        t = Table(title=f"Trust Score History ({len(events)} events)")
        t.add_column("Timestamp", style="dim")
        t.add_column("Entity", style="cyan")
        t.add_column("Event")
        t.add_column("Weight")
        t.add_column("Score Transition", style="bold")
        t.add_column("State")
        t.add_column("Detail")
        for ev in events:
            t.add_row(
                ev["timestamp"][:19].replace("T", " "),
                ev["entity"],
                ev["event"],
                f"{ev['weight']:.1f}",
                f"{ev['old_score']:.3f} -> {ev['new_score']:.3f}",
                f"{ev['old_state']} -> {ev['new_state']}",
                ev["detail"] or "",
            )
        console.print(t)

    elif args.trust_action == "reset":
        ok = store.reset_entity(args.entity)
        if ok:
            console.print(f"[green]Successfully reset trust score for '{args.entity}'.[/green]")
        else:
            console.print(f"[yellow]Entity '{args.entity}' was not found in trust store.[/yellow]")


def cmd_alerts(args):
    from rich.table import Table

    from common.alerts import get_default_alert_sink

    sink = get_default_alert_sink()
    if args.alerts_action == "tail":
        alerts = sink.read_alerts(limit=args.limit, kind=args.kind)
        if not alerts:
            console.print("[dim]No security alerts logged.[/dim]")
            return
        t = Table(title=f"Security Alerts Tail ({len(alerts)} items)")
        t.add_column("Timestamp", style="dim")
        t.add_column("Kind", style="bold red")
        t.add_column("Subject", style="cyan")
        t.add_column("Peer")
        t.add_column("Method")
        t.add_column("Cert CN")
        t.add_column("Detail")
        for a in alerts:
            t.add_row(
                a.timestamp[:19].replace("T", " "),
                a.kind,
                a.subject,
                a.peer,
                a.method,
                a.cert_cn,
                a.detail,
            )
        console.print(t)


def cmd_manual(args):
    from client.manual import show_manual

    section = getattr(args, "section", None)
    if getattr(args, "setup", False):
        section = "setup"
    interactive = getattr(args, "interactive", False)
    show_manual(section=section, interactive=interactive)
