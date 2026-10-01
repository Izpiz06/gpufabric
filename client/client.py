"""gRPC client SDK for GPU Fabric."""

from typing import List

import grpc

from common.constants import DEFAULT_PORT
from common.gpufabric_pb2 import (
    ExecuteRequest,
    ExecuteResponse,
    GPUInfoRequest,
    GPUInfoResponse,
    GPUStatusRequest,
    GPUStatusResponse,
    HealthRequest,
    HealthResponse,
    WorkloadType,
)
from common.gpufabric_pb2_grpc import GPUFabricServiceStub


class GPUFabricError(Exception):
    """Base exception for GPU Fabric client errors."""

    pass


class GPUFabricClient:
    """Client for querying GPU workers and executing workloads via gRPC."""

    def __init__(self, host: str = "localhost", port: int = DEFAULT_PORT, timeout: float = 15.0):
        # Format address
        if ":" in host:
            self.target = host
        else:
            self.target = f"{host}:{port}"

        self.timeout = timeout
        self._channel = grpc.insecure_channel(self.target)
        self._stub = GPUFabricServiceStub(self._channel)

    def close(self):
        self._channel.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def health(self) -> HealthResponse:
        try:
            return self._stub.GetHealth(HealthRequest(), timeout=self.timeout)
        except grpc.RpcError as e:
            raise GPUFabricError(f"Health check failed on {self.target}: {e.details() or e.code()}")

    def discover(self) -> HealthResponse:
        return self.health()

    def get_gpu_info(self, device_index: int = 0) -> GPUInfoResponse:
        try:
            req = GPUInfoRequest(device_index=device_index)
            return self._stub.GetGPUInfo(req, timeout=self.timeout)
        except grpc.RpcError as e:
            raise GPUFabricError(f"Failed to fetch GPU info: {e.details() or e.code()}")

    def get_status(self, device_index: int = 0) -> GPUStatusResponse:
        try:
            req = GPUStatusRequest(device_index=device_index)
            return self._stub.GetGPUStatus(req, timeout=self.timeout)
        except grpc.RpcError as e:
            raise GPUFabricError(f"Failed to fetch GPU status: {e.details() or e.code()}")

    def execute_vector_add(
        self, a: List[float], b: List[float], device_index: int = 0
    ) -> ExecuteResponse:
        try:
            req = ExecuteRequest(
                workload_type=WorkloadType.VECTOR_ADD,
                a=a,
                b=b,
                device_index=device_index,
            )
            return self._stub.Execute(req, timeout=self.timeout)
        except grpc.RpcError as e:
            raise GPUFabricError(f"Execution failed on GPU: {e.details() or e.code()}")
