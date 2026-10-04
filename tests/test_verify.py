# Copyright (c) 2026 Shreyas Mene
# SPDX-License-Identifier: Apache-2.0

"""Tests for client-side result verification helpers."""

import numpy as np
import pytest

from client.commands import compute_input_shapes
from client.verify import relative_error, tolerance


def test_relative_error_is_scaled():
    expected = np.array([100.0, 200.0])
    assert relative_error(expected, expected) == 0.0
    assert relative_error(np.array([100.0, 202.0]), expected) == pytest.approx(0.01)


def test_relative_error_scalar_and_zero():
    assert relative_error(np.float32(3.0), np.float64(3.0)) == 0.0
    assert relative_error(np.zeros(3), np.zeros(3)) == 0.0


def test_tolerance_reductions_are_looser():
    assert tolerance("matmul", np.float32) > tolerance("vector_add", np.float32)
    assert tolerance("vector_dot", np.float64) < tolerance("vector_dot", np.float32)


def test_compute_input_shapes():
    assert compute_input_shapes("vector_dot", 7) == [(7,), (7,)]
    assert compute_input_shapes("matmul", 4) == [(4, 4), (4, 4)]
