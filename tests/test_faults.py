# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for worker-side fault injector."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from worker.executor import GPUExecutor
from worker.faults import FaultInjector
from worker.server import parse_args


def test_fault_injector_defaults_and_validation():
    fi = FaultInjector()
    assert fi.is_active is False
    assert fi.mode == "none"

    arr = np.ones((10, 10), dtype=np.float32)
    assert np.array_equal(fi.inject(arr), arr)

    with pytest.raises(ValueError, match="Unknown fault mode"):
        FaultInjector(mode="invalid_mode")


def test_fault_injector_random_elements():
    arr = np.ones(1000, dtype=np.float32)
    fi1 = FaultInjector(mode="random_elements", rate=0.01, magnitude=5.0, seed=42)
    fi2 = FaultInjector(mode="random_elements", rate=0.01, magnitude=5.0, seed=42)

    res1 = fi1.inject(arr)
    res2 = fi2.inject(arr)

    # Determinism
    assert np.array_equal(res1, res2)

    # Corrupted count matches rate
    diff_mask = res1 != arr
    assert np.sum(diff_mask) == 10
    assert np.allclose(res1[diff_mask], 6.0)


def test_fault_injector_bitflip():
    for dtype in (np.float32, np.float64):
        arr = np.full(500, 3.14159, dtype=dtype)
        fi = FaultInjector(mode="bitflip", rate=0.02, seed=123)
        res = fi.inject(arr)
        diff_count = np.sum(res != arr)
        assert diff_count == 10


def test_fault_injector_zeros_and_scale():
    arr = np.full((10, 10), 4.0, dtype=np.float32)

    fi_zeros = FaultInjector(mode="zeros")
    assert np.all(fi_zeros.inject(arr) == 0.0)

    fi_scale = FaultInjector(mode="scale", magnitude=0.25)
    assert np.allclose(fi_scale.inject(arr), 5.0)


def test_fault_injector_stale():
    fi_stale = FaultInjector(mode="stale")
    arr1 = np.ones(5, dtype=np.float32)
    arr2 = np.full(5, 7.0, dtype=np.float32)

    # First call has no previous result; returns zeros
    res1 = fi_stale.inject(arr1)
    assert np.all(res1 == 0.0)

    # Second call returns previous result (arr1)
    res2 = fi_stale.inject(arr2)
    assert np.array_equal(res2, arr1)


def test_gpu_executor_hooks_fault_injector():
    mock_injector = MagicMock()
    mock_injector.inject.side_effect = lambda x: x + 999.0

    executor = GPUExecutor(fault_injector=mock_injector)
    executor.device_count = 1

    mock_cp = MagicMock()
    mock_cp.cuda.Device.return_value.__enter__.return_value = None
    mock_cp.asarray.side_effect = lambda x: x
    mock_cp.add.return_value = np.array([3.0, 7.0], dtype=np.float32)
    mock_cp.asnumpy.side_effect = lambda x: x

    with patch.dict("sys.modules", {"cupy": mock_cp}):
        with patch.object(executor, "is_gpu_ready", return_value=True):
            a = np.array([1.0, 2.0], dtype=np.float32)
            b = np.array([2.0, 5.0], dtype=np.float32)
            res, _, _ = executor.compute("vector_add", [a, b])
            assert mock_injector.inject.called
            assert np.allclose(res, [1002.0, 1006.0])


def test_server_cli_fault_args():
    with patch(
        "sys.argv",
        [
            "gpufabric-worker",
            "--inject-fault",
            "bitflip",
            "--fault-rate",
            "0.05",
            "--fault-magnitude",
            "2.5",
            "--fault-seed",
            "99",
        ],
    ):
        args = parse_args()
        assert args.inject_fault == "bitflip"
        assert args.fault_rate == 0.05
        assert args.fault_magnitude == 2.5
        assert args.fault_seed == 99
