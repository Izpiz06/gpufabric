"""gRPC Enrollment Service for issuing client certificates via authorization tokens."""

import logging
from concurrent import futures
from pathlib import Path
from typing import Optional

import grpc

from common.constants import DEFAULT_ENROLL_PORT
from common.gpufabric_pb2 import EnrollRequest, EnrollResponse
from common.gpufabric_pb2_grpc import (
    EnrollmentServiceServicer,
    add_EnrollmentServiceServicer_to_server,
)
from common.tls import sign_client_csr
from common.tokens import TokenStore

logger = logging.getLogger("gpufabric.worker.enrollment")


class EnrollmentServicer(EnrollmentServiceServicer):
    """Handles client enrollment by validating tokens and signing client CSRs."""

    def __init__(self, tls_dir: Path, token_store: Optional[TokenStore] = None):
        self.tls_dir = Path(tls_dir)
        self.token_store = token_store or TokenStore(self.tls_dir / "enrollment_tokens.json")

    def Enroll(self, request: EnrollRequest, context) -> EnrollResponse:
        token = request.token.strip()
        client_name = request.client_name.strip()
        csr_pem = request.csr_pem

        if not token:
            return EnrollResponse(success=False, message="Missing enrollment token.")
        if not csr_pem:
            return EnrollResponse(
                success=False, message="Missing Certificate Signing Request (CSR)."
            )

        if not self.token_store.validate_and_consume(token, client_name=client_name):
            return EnrollResponse(
                success=False,
                message="Invalid, expired, or already used enrollment token.",
            )

        try:
            client_cert_pem, ca_cert_pem = sign_client_csr(
                ca_dir=self.tls_dir,
                csr_pem=csr_pem,
                name=client_name or None,
            )
        except Exception as e:
            logger.error("Failed to sign client CSR: %s", e)
            return EnrollResponse(
                success=False,
                message=f"Failed to sign client certificate: {e}",
            )

        logger.info("Successfully enrolled client '%s'", client_name or "client")
        return EnrollResponse(
            success=True,
            message="Enrollment successful.",
            client_cert_pem=client_cert_pem,
            ca_cert_pem=ca_cert_pem,
        )


def create_enrollment_server(
    tls_dir: Path,
    host: str = "0.0.0.0",
    port: int = DEFAULT_ENROLL_PORT,
    token_store: Optional[TokenStore] = None,
    max_workers: int = 4,
) -> grpc.Server:
    """Create and bind a standalone gRPC server for client enrollment."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    servicer = EnrollmentServicer(tls_dir, token_store=token_store)
    add_EnrollmentServiceServicer_to_server(servicer, server)
    bound_port = server.add_insecure_port(f"{host}:{port}")
    server.bound_port = bound_port
    return server
