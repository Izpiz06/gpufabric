"""Tests for GPUExecutor."""

from unittest.mock import MagicMock, patch

import pytest

from worker.executor import GPUExecutionError, GPUExecutor


def test_executor_validation():
    executor = GPUExecutor()
    with pytest.raises(ValueError, match="Vector size mismatch"):
        executor.execute_vector_add([1.0, 2.0], [1.0])


def test_executor_no_gpu():
    executor = GPUExecutor()
    with patch.object(executor, "is_gpu_ready", return_value=False):
        with pytest.raises(GPUExecutionError):
            executor.execute_vector_add([1.0], [2.0])


def test_executor_cupy_mock():
    executor = GPUExecutor()
    executor.backend = "cupy"

    mock_cp = MagicMock()
    mock_cp.cuda.Device.return_value.__enter__.return_value = None
    mock_cp.asarray.side_effect = lambda x, dtype: MagicMock(
        __add__=lambda self, other: MagicMock(tolist=lambda: [3.0, 7.0])
    )

    with patch.dict("sys.modules", {"cupy": mock_cp}):
        with patch.object(executor, "is_gpu_ready", return_value=True):
            res, elapsed, backend = executor.execute_vector_add([1.0, 2.0], [2.0, 5.0])
            assert res == [3.0, 7.0]
            assert backend == "cupy"
            assert elapsed >= 0.0
