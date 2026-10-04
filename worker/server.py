# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""gRPC worker server initialization and lifecycle runner."""

import argparse
import logging
import sys
from concurrent import futures
from pathlib import Path
from typing import Optional

import grpc

from common.constants import DEFAULT_MAX_MESSAGE_MB, DEFAULT_PORT, MAX_MESSAGE_MB_LIMIT
from common.gpufabric_pb2_grpc import add_GPUFabricServiceServicer_to_server
from common.grpc_options import message_size_options
from common.tls import TLSConfigError, default_tls_dir, server_credentials
from worker.executor import GPUExecutor
from worker.gpu import GPUManager
from worker.service import GPUFabricServicer
from worker.state import WorkerState

logger = logging.getLogger("gpufabric.worker")


def create_grpc_server(
    worker_id: Optional[str] = None,
    max_workers: int = 10,
    max_message_mb: int = DEFAULT_MAX_MESSAGE_MB,
    warmup: bool = True,
) -> grpc.Server:
    """Instantiate and configure the gRPC server."""
    state = WorkerState(worker_id=worker_id)
    gpu_mgr = GPUManager()
    executor = GPUExecutor()
    if warmup:
        executor.warmup()

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=message_size_options(max_message_mb),
    )
    servicer = GPUFabricServicer(state, gpu_mgr, executor)
    add_GPUFabricServiceServicer_to_server(servicer, server)
    return server


def load_credentials(tls_dir: Optional[Path], insecure: bool) -> Optional[grpc.ServerCredentials]:
    """mTLS credentials from tls_dir (default dir if None), or None when insecure.

    Raises TLSConfigError if certificates are missing, so a worker never
    falls back to plaintext without being told to.
    """
    if insecure:
        return None
    return server_credentials(tls_dir or default_tls_dir())


def bind(server: grpc.Server, address: str, credentials) -> int:
    """Bind with mTLS if credentials are given, else without TLS. Returns the port."""
    if credentials is None:
        return server.add_insecure_port(address)
    return server.add_secure_port(address, credentials)


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
    parser.add_argument(
        "--no-warmup", action="store_true", help="Skip the GPU warm-up kernel at startup"
    )
    parser.add_argument(
        "--tls-dir",
        type=Path,
        default=None,
        help=f"Directory with ca.crt, server.crt, server.key (default: {default_tls_dir()})",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS: traffic is unencrypted and any client can connect",
    )
    parser.add_argument(
        "--no-discovery",
        action="store_true",
        help="Disable automatic LAN service advertisement (mDNS/DNS-SD)",
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
    try:
        credentials = load_credentials(args.tls_dir, args.insecure)
    except TLSConfigError as e:
        logger.error(f"{e} Or pass --insecure to run without TLS.")
        sys.exit(2)

    server = create_grpc_server(
        worker_id=args.worker_id,
        max_message_mb=args.max_message_mb,
        warmup=not args.no_warmup,
    )
    bound_port = bind(server, bind_address, credentials)
    if credentials is None:
        logger.warning(
            "Running WITHOUT TLS: traffic is unencrypted and any client on the network "
            "can use this GPU."
        )
        logger.info(f"GPU Fabric gRPC worker running on {bind_address}")
    else:
        logger.info(f"GPU Fabric gRPC worker running on {bind_address} with mutual TLS")

    server.start()

    advertiser = None
    if not args.no_discovery:
        from common.discovery import WorkerAdvertiser

        effective_id = args.worker_id or WorkerState(None).worker_id
        advertiser = WorkerAdvertiser(
            worker_id=effective_id,
            port=bound_port,
            host=args.host,
        )
        advertiser.start()

    try:
        server.wait_for_termination()
    finally:
        if advertiser is not None:
            advertiser.stop()


if __name__ == "__main__":
    run_worker()
