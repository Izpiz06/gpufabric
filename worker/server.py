"""gRPC worker server initialization and lifecycle runner."""

import argparse
import logging
from concurrent import futures
from typing import Optional

import grpc

from common.constants import DEFAULT_PORT
from common.gpufabric_pb2_grpc import add_GPUFabricServiceServicer_to_server
from worker.executor import GPUExecutor
from worker.gpu import GPUManager
from worker.service import GPUFabricServicer
from worker.state import WorkerState

logger = logging.getLogger("gpufabric.worker")


def create_grpc_server(worker_id: Optional[str] = None, max_workers: int = 10) -> grpc.Server:
    """Instantiate and configure the gRPC server."""
    state = WorkerState(worker_id=worker_id)
    gpu_mgr = GPUManager()
    executor = GPUExecutor()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
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
    return parser.parse_args()


def run_worker():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    bind_address = f"{args.host}:{args.port}"
    server = create_grpc_server(worker_id=args.worker_id)
    server.add_insecure_port(bind_address)
    logger.info(f"GPU Fabric gRPC worker running on {bind_address}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    run_worker()
