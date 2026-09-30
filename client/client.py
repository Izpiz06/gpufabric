"""GPU Fabric Client library for communicating with remote GPU workers."""

from typing import List, Optional
import httpx

from common.models import (
    ExecuteRequest,
    ExecuteResponse,
    GPUInfoResponse,
    GPUStatusResponse,
    HealthResponse,
    WorkloadType,
)
from common.protocol import (
    DEFAULT_PORT,
    EXECUTE_ENDPOINT,
    GPU_INFO_ENDPOINT,
    HEALTH_ENDPOINT,
    STATUS_ENDPOINT,
)


class GPUFabricError(Exception):
    """Base exception for GPU Fabric client errors."""
    pass


class GPUFabricConnectionError(GPUFabricError):
    """Raised when unable to connect to the GPU worker."""
    pass


class GPUFabricWorkerError(GPUFabricError):
    """Raised when worker returns an error status or bad request."""
    pass


class GPUFabricClient:
    """Client for querying GPU workers and executing workloads remotely."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = DEFAULT_PORT,
        base_url: Optional[str] = None,
        timeout: float = 15.0,
    ):
        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            # Handle if host was passed with http:// or port included
            host_clean = host.strip()
            if host_clean.startswith("http://") or host_clean.startswith("https://"):
                self.base_url = host_clean.rstrip("/")
            else:
                if ":" in host_clean:
                    self.base_url = f"http://{host_clean}"
                else:
                    self.base_url = f"http://{host_clean}:{port}"

        self.timeout = timeout
        self._http_client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def close(self):
        """Close underlying HTTP client."""
        self._http_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def discover(self) -> HealthResponse:
        """Alias for health check to verify connectivity to worker."""
        return self.health()

    def health(self) -> HealthResponse:
        """Query worker health and basic information."""
        try:
            resp = self._http_client.get(HEALTH_ENDPOINT)
        except httpx.RequestError as e:
            raise GPUFabricConnectionError(f"Failed to connect to worker at {self.base_url}: {e}")

        if resp.status_code != 200:
            raise GPUFabricWorkerError(
                f"Worker returned status {resp.status_code}: {resp.text}"
            )
        return HealthResponse.model_validate(resp.json())

    def get_gpu_info(self, device_index: int = 0) -> GPUInfoResponse:
        """Retrieve static GPU hardware specs and current memory state."""
        try:
            resp = self._http_client.get(
                GPU_INFO_ENDPOINT, params={"device_index": device_index}
            )
        except httpx.RequestError as e:
            raise GPUFabricConnectionError(f"Failed to connect to worker at {self.base_url}: {e}")

        if resp.status_code != 200:
            raise GPUFabricWorkerError(
                f"Worker returned status {resp.status_code}: {resp.json().get('detail', resp.text)}"
            )
        return GPUInfoResponse.model_validate(resp.json())

    def get_status(self, device_index: int = 0) -> GPUStatusResponse:
        """Retrieve real-time GPU load, memory usage, and temperature."""
        try:
            resp = self._http_client.get(
                STATUS_ENDPOINT, params={"device_index": device_index}
            )
        except httpx.RequestError as e:
            raise GPUFabricConnectionError(f"Failed to connect to worker at {self.base_url}: {e}")

        if resp.status_code != 200:
            raise GPUFabricWorkerError(
                f"Worker returned status {resp.status_code}: {resp.json().get('detail', resp.text)}"
            )
        return GPUStatusResponse.model_validate(resp.json())

    def execute_vector_add(
        self,
        a: List[float],
        b: List[float],
        device_index: int = 0,
    ) -> ExecuteResponse:
        """
        Submit a vector addition workload (C = A + B) to execute on the worker's GPU.
        """
        payload = ExecuteRequest(
            workload_type=WorkloadType.VECTOR_ADD,
            a=a,
            b=b,
            device_index=device_index,
        )
        try:
            resp = self._http_client.post(
                EXECUTE_ENDPOINT,
                json=payload.model_dump(),
            )
        except httpx.RequestError as e:
            raise GPUFabricConnectionError(f"Failed to connect to worker at {self.base_url}: {e}")

        if resp.status_code != 200:
            detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise GPUFabricWorkerError(f"Worker execution failed (HTTP {resp.status_code}): {detail}")

        return ExecuteResponse.model_validate(resp.json())
