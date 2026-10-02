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


def test_warmup_skips_without_gpu():
    executor = GPUExecutor()
    with patch.object(executor, "is_gpu_ready", return_value=False):
        with patch.object(executor, "execute_vector_add") as run:
            executor.warmup()
            run.assert_not_called()


def test_warmup_runs_each_device_and_survives_failures():
    executor = GPUExecutor()
    executor.device_count = 2
    with patch.object(executor, "is_gpu_ready", return_value=True):
        with patch.object(
            executor, "execute_vector_add", side_effect=[RuntimeError("boom"), ([2.0], 0.1, "cupy")]
        ) as run:
            executor.warmup()
            assert [c.args[2] for c in run.call_args_list] == [0, 1]
