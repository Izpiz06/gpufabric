"""CLI entry point and parser for GPU Fabric Client."""

import argparse
import sys
from pathlib import Path

from client import bench
from client.client import OPERATIONS, GPUFabricClient, GPUFabricError
from client.commands import (
    WORKERS_ENV,
    cmd_bench,
    cmd_compute,
    cmd_discover,
    cmd_enroll,
    cmd_execute,
    cmd_gpu,
    cmd_health,
    cmd_ls,
    cmd_status,
)
from client.formatters import console
from common.constants import (
    DEFAULT_ENROLL_PORT,
    DEFAULT_MAX_MESSAGE_MB,
    DEFAULT_PORT,
    MAX_MESSAGE_MB_LIMIT,
)
from common.tls import default_tls_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gpufabric-client", description="GPU Fabric gRPC Client CLI.")
    p.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"Worker port (default: {DEFAULT_PORT})"
    )
    p.add_argument("--timeout", type=float, default=15.0, help="Timeout in seconds (default: 15.0)")
    p.add_argument(
        "--max-message-mb",
        type=int,
        default=DEFAULT_MAX_MESSAGE_MB,
        help=f"Max gRPC message size in MiB (default: {DEFAULT_MAX_MESSAGE_MB})",
    )

    p.add_argument(
        "--tls-dir",
        type=Path,
        default=None,
        help=f"Directory with ca.crt, client.crt, client.key (default: {default_tls_dir()})",
    )
    p.add_argument(
        "--insecure",
        action="store_true",
        help="Connect without TLS (only for workers started with --insecure)",
    )

    sub = p.add_subparsers(dest="command", help="Command")

    p_ls = sub.add_parser("ls", help="List every GPU on one or more workers")
    p_ls.add_argument(
        "workers",
        nargs="*",
        help=f"Worker addresses (host or host:port). Defaults to ${WORKERS_ENV} (comma-separated)",
    )

    p_disc = sub.add_parser(
        "discover", help="Discover workers on local LAN or query specific worker"
    )
    p_disc.add_argument(
        "worker_ip",
        type=str,
        nargs="?",
        default=None,
        help="Optional IP or hostname of worker (omit to scan LAN automatically)",
    )

    p_health = sub.add_parser("health", help="Query worker health and monitoring status")
    p_health.add_argument("worker_ip", type=str, help="IP or hostname of worker")

    p_gpu = sub.add_parser("gpu", help="Query GPU info")
    p_gpu.add_argument("worker_ip", type=str, help="IP or hostname of worker")
    p_gpu.add_argument("--device", type=int, default=0, help="Device index (default: 0)")

    p_stat = sub.add_parser("status", help="Query live GPU status")
    p_stat.add_argument("worker_ip", type=str, help="IP or hostname of worker")
    p_stat.add_argument("--device", type=int, default=0, help="Device index (default: 0)")

    p_exec = sub.add_parser("execute", help="Execute vector add on remote GPU")
    p_exec.add_argument("worker_ip", type=str, help="IP or hostname of worker")
    p_exec.add_argument("--device", type=int, default=0, help="Device index (default: 0)")
    p_exec.add_argument("--a", nargs="+", type=float, help="Vector A elements")
    p_exec.add_argument("--b", nargs="+", type=float, help="Vector B elements")
    p_exec.add_argument("--size", type=int, help="Random vector size")

    p_comp = sub.add_parser(
        "compute", help="Run a tensor operation on the remote GPU and verify it against numpy"
    )
    p_comp.add_argument("worker_ip", type=str, help="IP or hostname of worker")
    p_comp.add_argument("op", choices=sorted(OPERATIONS), help="Operation to run")
    p_comp.add_argument(
        "--size", type=int, default=1000, help="Vector length or square matrix size (default: 1000)"
    )
    p_comp.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p_comp.add_argument("--device", type=int, default=0, help="Device index (default: 0)")
    p_comp.add_argument("--seed", type=int, default=0, help="Random seed for the inputs")

    p_bench = sub.add_parser(
        "bench",
        help="Benchmark a worker: network throughput, GPU bandwidth, matmul, Monte Carlo pi",
    )
    p_bench.add_argument("worker_ip", type=str, help="IP or hostname of worker")
    p_bench.add_argument("--device", type=int, default=0, help="Device index (default: 0)")
    p_bench.add_argument("--repeats", type=int, default=5, help="Timed runs per benchmark")
    p_bench.add_argument(
        "--min-network-mb-s",
        type=float,
        default=bench.DEFAULT_MIN_NETWORK_MB_S,
        help=f"Minimum end-to-end throughput in MB/s (default: {bench.DEFAULT_MIN_NETWORK_MB_S})",
    )
    p_bench.add_argument(
        "--min-gpu-bw-pct",
        type=float,
        default=bench.DEFAULT_MIN_GPU_BW_PCT,
        help="Minimum triad bandwidth as %% of theoretical peak "
        f"(default: {bench.DEFAULT_MIN_GPU_BW_PCT:.0f})",
    )
    p_bench.add_argument(
        "--network-size",
        type=int,
        default=1_000_000,
        help="Elements per vector for the network triad (default: 1000000, ~12 MB per run)",
    )
    p_bench.add_argument("--network-repeats", type=int, default=3)
    p_bench.add_argument(
        "--triad-size", type=int, default=None, help="GPU triad elements (default: auto)"
    )
    p_bench.add_argument("--matmul-size", type=int, default=1000)
    p_bench.add_argument("--pi-samples", type=int, default=100_000_000)
    p_bench.add_argument(
        "--sweep",
        action="store_true",
        help="Also double the triad size until GPU memory runs out",
    )

    p_enroll = sub.add_parser(
        "enroll",
        help="Enroll this client with a worker node using a short-lived authorization token",
    )
    p_enroll.add_argument("worker_ip", type=str, help="IP or hostname of worker")
    p_enroll.add_argument(
        "--token", type=str, required=True, help="Authorization token generated on worker"
    )
    p_enroll.add_argument(
        "--name", type=str, default=None, help="Client name (default: local hostname)"
    )
    p_enroll.add_argument(
        "--enroll-port",
        type=int,
        default=DEFAULT_ENROLL_PORT,
        help=f"Enrollment port on worker (default: {DEFAULT_ENROLL_PORT})",
    )
    p_enroll.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing client certificates in tls directory",
    )

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        from client.tui.app import GPUFabricApp

        app = GPUFabricApp(
            port=args.port,
            timeout=args.timeout,
            max_message_mb=args.max_message_mb,
            tls_dir=args.tls_dir,
            insecure=args.insecure,
        )
        app.run()
        return
    if not 1 <= args.max_message_mb <= MAX_MESSAGE_MB_LIMIT:
        parser.error(f"--max-message-mb must be between 1 and {MAX_MESSAGE_MB_LIMIT}")

    client_kwargs = {
        "port": args.port,
        "timeout": args.timeout,
        "max_message_mb": args.max_message_mb,
        "tls_dir": args.tls_dir,
        "insecure": args.insecure,
    }
    try:
        if args.command == "enroll":
            cmd_enroll(args, client_kwargs=client_kwargs)
            return
        if args.command == "ls":
            cmd_ls(args, client_kwargs)
            return
        if args.command == "discover" and not args.worker_ip:
            cmd_discover(None, args, client_kwargs=client_kwargs)
            return
        client = GPUFabricClient(host=args.worker_ip, **client_kwargs)
    except GPUFabricError as e:
        console.print(f"[bold red]GPU Fabric Error:[/bold red] {e}")
        sys.exit(2)

    with client:
        try:
            if args.command == "discover":
                cmd_discover(client, args)
            elif args.command == "health":
                cmd_health(client, args)
            elif args.command == "gpu":
                cmd_gpu(client, args)
            elif args.command == "status":
                cmd_status(client, args)
            elif args.command == "execute":
                cmd_execute(client, args)
            elif args.command == "compute":
                cmd_compute(client, args)
            elif args.command == "bench":
                cmd_bench(client, args)
        except GPUFabricError as e:
            console.print(f"[bold red]GPU Fabric Error:[/bold red] {e}")
            sys.exit(1)
        except Exception as e:
            console.print(f"[bold red]Unexpected Error:[/bold red] {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
