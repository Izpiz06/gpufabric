# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Known-answer probe client for auditing worker calculation integrity."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from client.client import GPUFabricClient
from client.verify import relative_error, tolerance
from common.alerts import AlertSink, emit_alert, get_default_alert_sink

logger = logging.getLogger("gpufabric.client.probe")


@dataclass
class ProbeReport:
    """Outcome of a known-answer worker audit probe."""

    passed: bool
    worker_target: str
    rel_error: float
    time_ms: float
    detail: str


def probe_worker(
    client: GPUFabricClient,
    device_index: int = 0,
    size: int = 16,
    seed: Optional[int] = None,
    alert_sink: Optional[AlertSink] = None,
) -> ProbeReport:
    """Submit a lightweight known-answer matmul probe to test worker execution integrity.

    The correct answer is precomputed in advance on the client CPU. If the remote worker
    corrupts, drops, or alters the calculation, a 'probe_fail' alert is emitted immediately.
    """
    rng = np.random.default_rng(seed if seed is not None else 1337)
    a = rng.uniform(-3.0, 3.0, size=(size, size)).astype(np.float32)
    b = rng.uniform(-3.0, 3.0, size=(size, size)).astype(np.float32)
    expected = np.matmul(a, b)

    try:
        res = client.compute("matmul", a, b, device_index=device_index)
        err = relative_error(res.result, expected)
        tol = tolerance("matmul", np.float32)
        passed = bool(err <= tol)
        time_ms = res.gpu_time_ms
        detail = (
            f"Known-answer probe passed (err={err:.2e}, tol={tol:.2e})"
            if passed
            else f"Known-answer probe failed (err={err:.2e} > tol={tol:.2e})"
        )
    except Exception as e:
        passed = False
        err = 1.0
        time_ms = 0.0
        detail = f"Probe execution failed with error: {e}"

    if not passed:
        sink = alert_sink or get_default_alert_sink()
        alert = emit_alert(
            kind="probe_fail",
            subject=f"node:{client.target}",
            peer=client.target,
            method="probe_worker",
            cert_cn="",
            detail=detail,
        )
        if alert_sink is not None and alert_sink != get_default_alert_sink():
            sink.emit(alert)

    return ProbeReport(
        passed=passed,
        worker_target=client.target,
        rel_error=err,
        time_ms=time_ms,
        detail=detail,
    )
