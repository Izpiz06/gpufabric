# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Decoy honeypot worker node implementation for adversary deception.

Simulates authentic worker telemetry and discovery without requiring NVIDIA GPU hardware
or CuPy dependencies. Every inbound RPC triggers a structured security alert.
"""

import logging
import random
import time
from typing import Any, Dict, Optional

import grpc

from common.alerts import AlertSink, emit_alert, extract_cert_cn, get_default_alert_sink
from common.constants import API_VERSION
from common.formatting import bytes_to_human
from common.gpufabric_pb2 import (
    BenchmarkResponse,
    ComputeResponse,
    ExecuteResponse,
    GPUDevice,
    GPUInfoResponse,
    GPUStatusResponse,
    HealthResponse,
    HealthState,
    ListGPUsResponse,
)
from common.gpufabric_pb2_grpc import (
    GPUFabricServiceServicer,
    add_GPUFabricServiceServicer_to_server,
)
from common.grpc_options import message_size_options

logger = logging.getLogger("gpufabric.worker.decoy")

DECOY_PROFILES: Dict[str, Dict[str, Any]] = {
    "RTX 4060 8GB": {
        "name": "NVIDIA GeForce RTX 4060",
        "total_vram": 8_589_934_592,  # 8 GiB
        "compute_capability": "8.9",
        "driver_version": "550.54.14",
        "cuda_version": "12.4",
        "base_temp": 42.0,
        "base_util": 8.0,
        "power_limit_w": 115,
        "power_usage_w": 28,
    },
    "RTX 3060 12GB": {
        "name": "NVIDIA GeForce RTX 3060",
        "total_vram": 12_884_901_888,
        "compute_capability": "8.6",
        "driver_version": "550.54.14",
        "cuda_version": "12.4",
        "base_temp": 46.0,
        "base_util": 10.0,
        "power_limit_w": 170,
        "power_usage_w": 35,
    },
}


class DecoyServicer(GPUFabricServiceServicer):
    """Deception honeypot servicer that presents realistic GPU hardware profiles.

    Never executes user tensor computations or returns results. Computations return
    UNAVAILABLE ('GPU busy') after a short delay and log full contact telemetry.
    """

    def __init__(
        self,
        worker_id: Optional[str] = None,
        profile_name: str = "RTX 4060 8GB",
        alert_sink: Optional[AlertSink] = None,
    ):
        self.worker_id = worker_id or f"decoy-worker-{profile_name.lower().replace(' ', '-')}"
        self.profile_name = profile_name if profile_name in DECOY_PROFILES else "RTX 4060 8GB"
        self.profile = DECOY_PROFILES[self.profile_name]
        self.alert_sink = alert_sink or get_default_alert_sink()
        self._start_time = time.time()

    def _record_contact(self, method: str, context: Any, detail: str = "") -> None:
        peer = context.peer() if context else ""
        cert_cn = extract_cert_cn(context)
        subject_id = f"user:{cert_cn}" if cert_cn else f"node:{self.worker_id}"

        alert_detail = f"Adversary contact on decoy node ({self.profile_name}) via RPC {method}."
        if detail:
            alert_detail = f"{alert_detail} {detail}"

        alert = emit_alert(
            kind="decoy_contact",
            subject=subject_id,
            peer=peer,
            method=method,
            cert_cn=cert_cn,
            detail=alert_detail,
        )
        if self.alert_sink is not None and self.alert_sink != get_default_alert_sink():
            self.alert_sink.emit(alert)

        logger.warning(
            f"DECOY CONTACT: method={method} peer={peer} cert_cn={cert_cn!r} "
            f"decoy_id={self.worker_id}"
        )

    def _get_jittered_telemetry(self) -> Dict[str, Any]:
        """Generate realistic slight jitter around base profile telemetry."""
        jitter_temp = round(self.profile["base_temp"] + random.uniform(-1.5, 2.0), 1)
        jitter_util = max(0, min(100, int(self.profile["base_util"] + random.uniform(-3, 5))))
        jitter_used = int(self.profile["total_vram"] * (0.15 + random.uniform(-0.02, 0.03)))
        free_vram = self.profile["total_vram"] - jitter_used
        jitter_pwr = max(
            10,
            min(
                self.profile["power_limit_w"],
                int(self.profile["power_usage_w"] + random.uniform(-4, 6)),
            ),
        )

        return {
            "temp": int(jitter_temp),
            "gpu_util": jitter_util,
            "mem_util": max(0, min(100, jitter_util - 2)),
            "used_vram": jitter_used,
            "free_vram": free_vram,
            "power_usage_w": jitter_pwr,
        }

    def GetHealth(self, request, context) -> HealthResponse:
        self._record_contact("GetHealth", context, detail="Health query received")
        return HealthResponse(
            status="ok",
            worker_id=self.worker_id,
            version=API_VERSION,
            gpu_available=True,
            health_state=HealthState.HEALTHY,
            hostname=f"host-{self.worker_id}",
            api_version=API_VERSION,
            cuda_version=self.profile["cuda_version"],
            driver_version=self.profile["driver_version"],
            last_heartbeat=time.time(),
            health_details="All GPU systems operational.",
            active_tasks=0,
            gpu_count=1,
        )

    def GetGPUInfo(self, request, context) -> GPUInfoResponse:
        self._record_contact(
            "GetGPUInfo", context, detail=f"Device {request.device_index} info query"
        )
        if request.device_index != 0:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Invalid device index {request.device_index}",
            )

        telem = self._get_jittered_telemetry()
        total = self.profile["total_vram"]
        return GPUInfoResponse(
            device_index=0,
            name=self.profile["name"],
            total_vram_bytes=total,
            free_vram_bytes=telem["free_vram"],
            used_vram_bytes=telem["used_vram"],
            total_vram_human=bytes_to_human(total),
            free_vram_human=bytes_to_human(telem["free_vram"]),
            used_vram_human=bytes_to_human(telem["used_vram"]),
            compute_capability=self.profile["compute_capability"],
            driver_version=self.profile["driver_version"],
        )

    def GetGPUStatus(self, request, context) -> GPUStatusResponse:
        self._record_contact(
            "GetGPUStatus", context, detail=f"Device {request.device_index} telemetry query"
        )
        if request.device_index != 0:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Invalid device index {request.device_index}",
            )

        telem = self._get_jittered_telemetry()
        total = self.profile["total_vram"]
        return GPUStatusResponse(
            device_index=0,
            gpu_utilization_pct=telem["gpu_util"],
            memory_utilization_pct=telem["mem_util"],
            total_vram_bytes=total,
            free_vram_bytes=telem["free_vram"],
            used_vram_bytes=telem["used_vram"],
            free_vram_human=bytes_to_human(telem["free_vram"]),
            used_vram_human=bytes_to_human(telem["used_vram"]),
            temperature_c=telem["temp"],
            active_tasks=0,
            power_usage_w=telem["power_usage_w"],
            power_limit_w=self.profile["power_limit_w"],
            available=True,
            current_workload="IDLE",
        )

    def ListGPUs(self, request, context) -> ListGPUsResponse:
        self._record_contact("ListGPUs", context, detail="Inventory listing query")
        telem = self._get_jittered_telemetry()
        total = self.profile["total_vram"]
        gpu = GPUDevice(
            device_index=0,
            name=self.profile["name"],
            total_vram_bytes=total,
            free_vram_bytes=telem["free_vram"],
            compute_capability=self.profile["compute_capability"],
            gpu_utilization_pct=telem["gpu_util"],
            temperature_c=telem["temp"],
        )
        return ListGPUsResponse(
            worker_id=self.worker_id,
            version=API_VERSION,
            driver_version=self.profile["driver_version"],
            compute_ready=True,
            gpus=[gpu],
        )

    def Compute(self, request, context) -> ComputeResponse:
        self._record_contact("Compute", context, detail=f"Target op: {request.op} on decoy")
        time.sleep(0.040)  # plausible simulated scheduling attempt
        context.abort(
            grpc.StatusCode.UNAVAILABLE,
            "GPU device is currently busy with high-priority workload.",
        )

    def Execute(self, request, context) -> ExecuteResponse:
        self._record_contact("Execute", context, detail="Execute vector_add on decoy")
        time.sleep(0.035)
        context.abort(
            grpc.StatusCode.UNAVAILABLE,
            "GPU device is currently busy with high-priority workload.",
        )

    def RunBenchmark(self, request, context) -> BenchmarkResponse:
        self._record_contact("RunBenchmark", context, detail="Benchmark request on decoy")
        time.sleep(0.050)
        context.abort(
            grpc.StatusCode.UNAVAILABLE,
            "GPU device is currently busy with high-priority workload.",
        )


def create_decoy_server(
    worker_id: Optional[str] = None,
    profile_name: str = "RTX 4060 8GB",
    max_workers: int = 10,
    max_message_mb: int = 256,
    alert_sink: Optional[AlertSink] = None,
) -> grpc.Server:
    """Instantiate a gRPC server configured with DecoyServicer without GPU requirement."""
    from concurrent import futures

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=message_size_options(max_message_mb),
    )
    servicer = DecoyServicer(worker_id=worker_id, profile_name=profile_name, alert_sink=alert_sink)
    add_GPUFabricServiceServicer_to_server(servicer, server)
    return server
