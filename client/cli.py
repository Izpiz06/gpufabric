"""CLI entry point and parser for GPU Fabric Client."""

import argparse
import sys

from client.client import GPUFabricClient, GPUFabricError
from client.commands import cmd_discover, cmd_execute, cmd_gpu, cmd_status
from client.formatters import console
from common.constants import DEFAULT_MAX_MESSAGE_MB, DEFAULT_PORT, MAX_MESSAGE_MB_LIMIT


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

    sub = p.add_subparsers(dest="command", help="Command")

    p_disc = sub.add_parser("discover", help="Discover worker")
    p_disc.add_argument("worker_ip", type=str, help="IP or hostname of worker")

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

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)
    if not 1 <= args.max_message_mb <= MAX_MESSAGE_MB_LIMIT:
        parser.error(f"--max-message-mb must be between 1 and {MAX_MESSAGE_MB_LIMIT}")

    with GPUFabricClient(
        host=args.worker_ip,
        port=args.port,
        timeout=args.timeout,
        max_message_mb=args.max_message_mb,
    ) as client:
        try:
            if args.command == "discover":
                cmd_discover(client, args)
            elif args.command == "gpu":
                cmd_gpu(client, args)
            elif args.command == "status":
                cmd_status(client, args)
            elif args.command == "execute":
                cmd_execute(client, args)
        except GPUFabricError as e:
            console.print(f"[bold red]GPU Fabric Error:[/bold red] {e}")
            sys.exit(1)
        except Exception as e:
            console.print(f"[bold red]Unexpected Error:[/bold red] {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
