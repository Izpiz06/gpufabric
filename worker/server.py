"""CLI entry point for running the GPU Fabric worker server."""

import argparse
import logging

import uvicorn

from common.protocol import DEFAULT_PORT
from worker.app import create_app


def parse_args():
    parser = argparse.ArgumentParser(
        prog="gpufabric-worker",
        description="Run a GPU Fabric worker node to expose local GPUs over LAN.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host IP to bind the worker server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to listen on (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Custom unique identifier for this worker (defaults to hostname)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)",
    )
    return parser.parse_args()


def run_worker():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("gpufabric.worker")
    logger.info(f"Starting GPU Fabric worker on {args.host}:{args.port}")

    app = create_app(worker_id=args.worker_id)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    run_worker()
