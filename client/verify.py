# Copyright (c) 2026 Shreyas Mene
# SPDX-License-Identifier: Apache-2.0

"""Reference results and error checks for verifying remote GPU output."""

import numpy as np

# Plain numpy versions of each Compute operation, run on the client's CPU.
REFERENCE = {
    "vector_add": np.add,
    "vector_mul": np.multiply,
    "vector_dot": np.dot,
    "matrix_add": np.add,
    "matmul": np.matmul,
    "triad": lambda b, c, s: b + s * c,
}

# Element-wise ops are exact in IEEE arithmetic. Reductions (dot, matmul) sum
# in a different order on the GPU, so they need a looser bound.
_TOLERANCE = {
    np.dtype(np.float32): {"elementwise": 1e-6, "reduction": 1e-4},
    np.dtype(np.float64): {"elementwise": 1e-12, "reduction": 1e-10},
}


def relative_error(result: np.ndarray, expected: np.ndarray) -> float:
    """Max absolute difference, scaled by the largest expected magnitude."""
    expected = np.asarray(expected, dtype=np.float64)
    scale = max(float(np.max(np.abs(expected), initial=0.0)), 1e-30)
    diff = np.abs(np.asarray(result, dtype=np.float64) - expected)
    return float(np.max(diff, initial=0.0)) / scale


def tolerance(op: str, dtype) -> float:
    kind = "reduction" if op in ("vector_dot", "matmul") else "elementwise"
    return _TOLERANCE[np.dtype(dtype)][kind]
