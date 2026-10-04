# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Tests for common utilities and protobuf models."""

import pytest

from common.constants import DEFAULT_PORT, MAX_MESSAGE_MB_LIMIT
from common.formatting import bytes_to_human
from common.gpufabric_pb2 import (
    ExecuteRequest,
    ExecuteResponse,
    GPUInfoResponse,
    HealthResponse,
    WorkloadType,
)
from common.grpc_options import message_size_options


def test_bytes_to_human():
    assert bytes_to_human(0) == "0.00 B"
    assert bytes_to_human(1024) == "1.00 KiB"
    assert bytes_to_human(1024 * 1024) == "1.00 MiB"
    assert bytes_to_human(8 * 1024 * 1024 * 1024) == "8.00 GiB"
    assert bytes_to_human(-10) == "0.00 B"
    assert DEFAULT_PORT == 50051


def test_protobuf_messages():
    health = HealthResponse(status="ok", worker_id="test-node", version="0.1.0", gpu_available=True)
    assert health.worker_id == "test-node"
    assert health.gpu_available is True

    info = GPUInfoResponse(
        device_index=0,
        name="NVIDIA RTX 4090",
        total_vram_bytes=24000000000,
        compute_capability="8.9",
    )
    assert info.name == "NVIDIA RTX 4090"

    req = ExecuteRequest(
        workload_type=WorkloadType.VECTOR_ADD,
        a=[1.0, 2.0],
        b=[3.0, 4.0],
        device_index=0,
    )
    assert list(req.a) == [1.0, 2.0]

    resp = ExecuteResponse(
        task_id="t-1",
        workload_type="vector_add",
        status="success",
        result=[4.0, 6.0],
        execution_time_ms=0.12,
    )
    assert resp.status == "success"
    assert list(resp.result) == [4.0, 6.0]


def test_message_size_options():
    opts = dict(message_size_options(256))
    assert opts["grpc.max_send_message_length"] == 256 * 1024 * 1024
    assert opts["grpc.max_receive_message_length"] == 256 * 1024 * 1024


@pytest.mark.parametrize("bad", [0, -1, MAX_MESSAGE_MB_LIMIT + 1])
def test_message_size_options_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        message_size_options(bad)


def test_copyright_headers_compliance():
    from scripts.check_headers import main as check_headers_main

    assert check_headers_main() == 0
