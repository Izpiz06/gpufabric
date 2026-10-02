"""gRPC worker server initialization and lifecycle runner."""

import argparse
import logging
from concurrent import futures
from typing import Optional

import grpc

from common.constants import DEFAULT_MAX_MESSAGE_MB, DEFAULT_PORT, MAX_MESSAGE_MB_LIMIT
from common.gpufabric_pb2_grpc import add_GPUFabricServiceServicer_to_server
from common.grpc_options import message_size_options
from worker.executor import GPUExecutor
from worker.gpu import GPUManager
from worker.service import GPUFabricServicer
from worker.state import WorkerState

logger = logging.getLogger("gpufabric.worker")


def create_grpc_server(
    worker_id: Optional[str] = None,
    max_workers: int = 10,
    max_message_mb: int = DEFAULT_MAX_MESSAGE_MB,
) -> grpc.Server:
    """Instantiate and configure the gRPC server."""
    state = WorkerState(worker_id=worker_id)
    gpu_mgr = GPUManager()
    executor = GPUExecutor()

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=message_size_options(max_message_mb),
    )
    servicer = GPUFabricServicer(state, gpu_mgr, executor)
    add_GPUFabricServiceServicer_to_server(servicer, server)
    return server


def parse_args():
    parser = argparse.ArgumentParser(
        prog="gpufabric-worker",
        description="Run a GPU Fabric gRPC worker node to expose GPUs over LAN.",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host IP to bind (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"Port to bind (default: {DEFAULT_PORT})"
    )
    parser.add_argument("--worker-id", type=str, default=None, help="Custom worker identifier")
    parser.add_argument(
        "--log-level", type=str, default="info", choices=["debug", "info", "warning", "error"]
    )
    parser.add_argument(
        "--max-message-mb",
        type=int,
        default=DEFAULT_MAX_MESSAGE_MB,
        help=f"Max gRPC message size in MiB (default: {DEFAULT_MAX_MESSAGE_MB})",
    )
    args = parser.parse_args()
    if not 1 <= args.max_message_mb <= MAX_MESSAGE_MB_LIMIT:
        parser.error(f"--max-message-mb must be between 1 and {MAX_MESSAGE_MB_LIMIT}")
    return args


def run_worker():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    bind_address = f"{args.host}:{args.port}"
    server = create_grpc_server(
        worker_id=args.worker_id,
        max_message_mb=args.max_message_mb,
    )
    server.add_insecure_port(bind_address)
    logger.info(f"GPU Fabric gRPC worker running on {bind_address}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    run_worker()
