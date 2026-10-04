# Copyright (c) 2026 Shreyas Mene
# SPDX-License-Identifier: Apache-2.0

"""Conversion between numpy arrays and protobuf Tensor messages."""

import math

import numpy as np

from common.gpufabric_pb2 import DType, Tensor

# Wire format is little-endian regardless of the host's byte order.
_DTYPE_TO_NUMPY = {
    DType.FLOAT32: np.dtype("<f4"),
    DType.FLOAT64: np.dtype("<f8"),
}
_NUMPY_TO_DTYPE = {np.dtype(np.float32): DType.FLOAT32, np.dtype(np.float64): DType.FLOAT64}


class TensorError(ValueError):
    """Raised when a Tensor message or array cannot be converted."""


def to_tensor(array) -> Tensor:
    """Encode a numpy array (or array-like) as a Tensor message."""
    arr = np.asarray(array)
    dtype = _NUMPY_TO_DTYPE.get(np.dtype(arr.dtype).newbyteorder("="))
    if dtype is None:
        raise TensorError(f"Unsupported dtype {arr.dtype}; use float32 or float64")
    # np.ascontiguousarray would turn a 0-d scalar into shape (1,); asarray keeps it.
    arr = np.asarray(arr, dtype=_DTYPE_TO_NUMPY[dtype], order="C")
    return Tensor(dtype=dtype, shape=list(arr.shape), data=arr.tobytes())


def from_tensor(tensor: Tensor) -> np.ndarray:
    """Decode a Tensor message into a read-only numpy array."""
    np_dtype = _DTYPE_TO_NUMPY.get(tensor.dtype)
    if np_dtype is None:
        raise TensorError(f"Unsupported or missing tensor dtype: {tensor.dtype}")
    shape = tuple(tensor.shape)
    if any(dim < 0 for dim in shape):
        raise TensorError(f"Tensor shape has a negative dimension: {list(shape)}")
    expected = math.prod(shape) * np_dtype.itemsize
    if len(tensor.data) != expected:
        raise TensorError(
            f"Tensor data is {len(tensor.data)} bytes, expected {expected} "
            f"for shape {list(shape)} and {np_dtype.name}"
        )
    return np.frombuffer(tensor.data, dtype=np_dtype).reshape(shape)
