"""Tests for GPUExecutor."""

from unittest.mock import MagicMock, patch
import pytest
from worker.executor import GPUExecutor, GPUExecutionError


def test_executor_dimension_mismatch():
    executor = GPUExecutor()
    with pytest.raises(ValueError, match="Vector dimensions mismatch"):
        executor.execute_vector_add([1.0, 2.0], [1.0])


def test_executor_no_gpu_available():
    executor = GPUExecutor()
    executor.backend = "none"
    with patch.object(executor, "is_gpu_ready", return_value=False):
        with pytest.raises(GPUExecutionError, match="No CUDA-capable GPU"):
            executor.execute_vector_add([1.0, 2.0], [3.0, 4.0])


def test_executor_cupy_mock():
    executor = GPUExecutor()
    executor.backend = "cupy"

    mock_cp = MagicMock()
    mock_cp.cuda.Device.return_value.__enter__.return_value = None
    mock_cp.asarray.side_effect = lambda x, dtype: MagicMock(
        __add__=lambda self, other: MagicMock(tolist=lambda: [5.0, 7.0])
    )

    with patch.dict("sys.modules", {"cupy": mock_cp}):
        with patch.object(executor, "is_gpu_ready", return_value=True):
            result, exec_time, backend = executor.execute_vector_add([1.0, 2.0], [4.0, 5.0])
            assert result == [5.0, 7.0]
            assert backend == "cupy"
            assert exec_time >= 0.0
