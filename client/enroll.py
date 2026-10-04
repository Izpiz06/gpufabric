"""Client enrollment workflow for pairing with a GPU Fabric worker without SSH/SCP."""

import logging
import socket
from pathlib import Path
from typing import Any, Optional, Tuple

import grpc
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from client.client import GPUFabricClient, GPUFabricError
from common.constants import DEFAULT_ENROLL_PORT, DEFAULT_PORT
from common.gpufabric_pb2 import EnrollRequest
from common.gpufabric_pb2_grpc import EnrollmentServiceStub
from common.tls import default_tls_dir, generate_client_csr, save_client_bundle

logger = logging.getLogger("gpufabric.client.enroll")


class EnrollmentError(GPUFabricError):
    """Raised when client enrollment fails."""


def enroll_client(
    worker_ip: str,
    token: str,
    client_name: Optional[str] = None,
    enroll_port: int = DEFAULT_ENROLL_PORT,
    port: int = DEFAULT_PORT,
    tls_dir: Optional[Path] = None,
    force: bool = False,
    verify_mtls: bool = True,
    timeout: float = 15.0,
) -> Tuple[Path, Optional[Any]]:
    """Enroll this client with a worker node using a short-lived authorization token.

    Workflow:
    1. Generates a local EC private key (never leaves this machine).
    2. Generates a Certificate Signing Request (CSR).
    3. Submits the CSR and token to the worker's enrollment service.
    4. Worker validates token, signs the CSR with its CA, and returns client.crt and ca.crt.
    5. Saves client.key (0o600), client.crt (0o644), and ca.crt (0o644) into tls_dir.
    6. Optionally verifies mTLS connection against the worker's main port.

    Returns:
        (tls_dir, gpus_info_or_none)
    """
    dest_dir = Path(tls_dir or default_tls_dir())
    token = token.strip()
    if not token:
        raise EnrollmentError("An enrollment token is required.")

    if not force:
        has_key = (dest_dir / "client.key").is_file()
        has_cert = (dest_dir / "client.crt").is_file()
        if has_key and has_cert:
            raise EnrollmentError(
                f"Client credentials already exist in {dest_dir}. Use --force to overwrite."
            )

    effective_name = (client_name or "").strip()
    if not effective_name:
        try:
            effective_name = socket.gethostname() or "client"
        except Exception:
            effective_name = "client"

    # Step 1: Generate client private key locally (never shared with worker)
    key = ec.generate_private_key(ec.SECP256R1())
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    # Step 2: Create Certificate Signing Request
    csr_pem = generate_client_csr(effective_name, key)

    # Step 3: Call Enrollment RPC on the worker
    channel = grpc.insecure_channel(f"{worker_ip}:{enroll_port}")
    try:
        stub = EnrollmentServiceStub(channel)
        req = EnrollRequest(token=token, client_name=effective_name, csr_pem=csr_pem)
        resp = stub.Enroll(req, timeout=timeout)
    except grpc.RpcError as e:
        raise EnrollmentError(
            f"Failed to connect to enrollment service at {worker_ip}:{enroll_port}: "
            f"{e.details() or e}"
        ) from e
    finally:
        channel.close()

    # Step 4: Validate worker response
    if not resp.success:
        raise EnrollmentError(f"Enrollment rejected by worker: {resp.message}")
    if not resp.client_cert_pem or not resp.ca_cert_pem:
        raise EnrollmentError("Worker response did not contain the required certificates.")

    # Step 5: Save key and certificates with strict permissions
    save_client_bundle(
        tls_dir=dest_dir,
        client_key_pem=key_pem,
        client_cert_pem=resp.client_cert_pem,
        ca_cert_pem=resp.ca_cert_pem,
    )

    gpu_info = None
    # Step 6: Verify mTLS connection against the main worker port
    if verify_mtls:
        try:
            client = GPUFabricClient(
                host=worker_ip,
                port=port,
                tls_dir=dest_dir,
                timeout=timeout,
                insecure=False,
            )
            with client:
                gpu_info = client.get_gpus()
        except Exception as e:
            logger.warning("mTLS verification warning: %s", e)

    return dest_dir, gpu_info
