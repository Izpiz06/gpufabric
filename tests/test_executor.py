"""Tests for GPUExecutor."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from worker.executor import GPUExecutionError, GPUExecutor, validate_inputs


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


def _f32(*shape):
    return np.zeros(shape, dtype=np.float32)


@pytest.mark.parametrize(
    "op, inputs, message",
    [
        ("transpose", [_f32(2), _f32(2)], "Unsupported operation"),
        ("vector_add", [_f32(2)], "takes 2 inputs"),
        ("vector_add", [_f32(2), np.zeros(2, dtype=np.float64)], "share a dtype"),
        ("vector_mul", [_f32(2, 2), _f32(2, 2)], "1-D vectors"),
        ("vector_dot", [_f32(2), _f32(3)], "size mismatch"),
        ("matrix_add", [_f32(2), _f32(2)], "2-D matrices"),
        ("matrix_add", [_f32(2, 3), _f32(3, 2)], "shape mismatch"),
        ("matmul", [_f32(2, 3), _f32(2, 3)], "inner dimensions"),
    ],
)
def test_validate_inputs_rejects(op, inputs, message):
    with pytest.raises(ValueError, match=message):
        validate_inputs(op, inputs)


@pytest.mark.parametrize(
    "op, inputs",
    [
        ("vector_add", [_f32(5), _f32(5)]),
        ("vector_dot", [_f32(5), _f32(5)]),
        ("matrix_add", [_f32(2, 3), _f32(2, 3)]),
        ("matmul", [_f32(2, 3), _f32(3, 4)]),
    ],
)
def test_validate_inputs_accepts(op, inputs):
    validate_inputs(op, inputs)


def test_compute_rejects_out_of_range_device():
    executor = GPUExecutor()
    executor.device_count = 1
    with pytest.raises(ValueError, match="Invalid device_index 3"):
        executor.compute("vector_add", [_f32(2), _f32(2)], device_index=3)


def test_compute_without_gpu():
    executor = GPUExecutor()
    with patch.object(executor, "is_gpu_ready", return_value=False):
        with pytest.raises(GPUExecutionError):
            executor.compute("vector_add", [_f32(2), _f32(2)])
