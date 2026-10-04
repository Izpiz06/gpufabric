# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""GPU Fabric common protobuf and helper exports."""

from common.constants import (
    API_VERSION,
    DEFAULT_DISCOVERY_TIMEOUT,
    DEFAULT_ENROLL_PORT,
    DEFAULT_PORT,
    DEFAULT_TOKEN_TTL_SECONDS,
    DISCOVERY_SERVICE_TYPE,
)
from common.discovery import (
    DiscoveredCandidate,
    DiscoveredWorker,
    WorkerAdvertiser,
    browse_lan_candidates,
    get_local_ip_addresses,
)
from common.formatting import bytes_to_human
from common.gpufabric_pb2 import (
    EnrollRequest,
    EnrollResponse,
    ExecuteRequest,
    ExecuteResponse,
    GPUInfoRequest,
    GPUInfoResponse,
    GPUStatusRequest,
    GPUStatusResponse,
    HealthRequest,
    HealthResponse,
    HealthState,
    WorkloadType,
)
from common.gpufabric_pb2_grpc import (
    EnrollmentServiceServicer,
    EnrollmentServiceStub,
    GPUFabricServiceServicer,
    GPUFabricServiceStub,
    add_EnrollmentServiceServicer_to_server,
    add_GPUFabricServiceServicer_to_server,
)
from common.tokens import EnrollmentToken, TokenStore

__all__ = [
    "DEFAULT_PORT",
    "DEFAULT_ENROLL_PORT",
    "DEFAULT_TOKEN_TTL_SECONDS",
    "API_VERSION",
    "DISCOVERY_SERVICE_TYPE",
    "DEFAULT_DISCOVERY_TIMEOUT",
    "bytes_to_human",
    "DiscoveredCandidate",
    "DiscoveredWorker",
    "WorkerAdvertiser",
    "browse_lan_candidates",
    "get_local_ip_addresses",
    "HealthRequest",
    "HealthResponse",
    "HealthState",
    "GPUInfoRequest",
    "GPUInfoResponse",
    "GPUStatusRequest",
    "GPUStatusResponse",
    "ExecuteRequest",
    "ExecuteResponse",
    "WorkloadType",
    "GPUFabricServiceStub",
    "GPUFabricServiceServicer",
    "add_GPUFabricServiceServicer_to_server",
    "EnrollRequest",
    "EnrollResponse",
    "EnrollmentServiceStub",
    "EnrollmentServiceServicer",
    "add_EnrollmentServiceServicer_to_server",
    "EnrollmentToken",
    "TokenStore",
]
