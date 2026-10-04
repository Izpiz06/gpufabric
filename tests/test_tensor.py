# Copyright (c) 2026 Shreyas Mene
# SPDX-License-Identifier: Apache-2.0

"""Tests for numpy <-> Tensor conversion."""

import numpy as np
import pytest

from common.gpufabric_pb2 import DType, Tensor
from common.tensor import TensorError, from_tensor, to_tensor


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_roundtrip(dtype):
    arr = np.arange(12, dtype=dtype).reshape(3, 4)
    t = to_tensor(arr)
    assert list(t.shape) == [3, 4]
    out = from_tensor(t)
    assert out.dtype == arr.dtype
    np.testing.assert_array_equal(out, arr)


def test_scalar_roundtrip():
    t = to_tensor(np.float32(2.5))
    assert list(t.shape) == []
    assert from_tensor(t) == np.float32(2.5)


def test_non_contiguous_and_big_endian_inputs():
    arr = np.arange(16, dtype=np.float32).reshape(4, 4).T
    np.testing.assert_array_equal(from_tensor(to_tensor(arr)), arr)
    be = np.arange(4, dtype=">f8")
    out = from_tensor(to_tensor(be))
    assert out.dtype == np.dtype("<f8")
    np.testing.assert_array_equal(out, be)


def test_unsupported_dtype():
    with pytest.raises(TensorError, match="Unsupported dtype"):
        to_tensor(np.arange(3, dtype=np.int32))


def test_missing_dtype():
    with pytest.raises(TensorError, match="dtype"):
        from_tensor(Tensor(shape=[1], data=b"\x00" * 4))


def test_data_size_mismatch():
    t = Tensor(dtype=DType.FLOAT32, shape=[3], data=b"\x00" * 8)
    with pytest.raises(TensorError, match="expected 12"):
        from_tensor(t)


def test_negative_dimension():
    t = Tensor(dtype=DType.FLOAT32, shape=[-1], data=b"")
    with pytest.raises(TensorError, match="negative"):
        from_tensor(t)
