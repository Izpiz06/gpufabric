# Copyright (c) 2026 Shreyas Mene
# SPDX-License-Identifier: Apache-2.0

"""Tests for client-side result verification helpers."""

import numpy as np
import pytest

from client.commands import compute_input_shapes
from client.verify import (
    REFERENCE,
    freivalds_check,
    relative_error,
    tolerance,
    verify_result,
)


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


def test_freivalds_accepts_honest():
    """Verify Freivalds algorithm accepts honest float32 and float64 results across sizes."""
    rng = np.random.default_rng(12345)
    for dtype in (np.float32, np.float64):
        for n in (32, 64, 128, 256):
            a = rng.uniform(-5.0, 5.0, size=(n, n)).astype(dtype)
            b = rng.uniform(-5.0, 5.0, size=(n, n)).astype(dtype)
            c = a @ b
            res = freivalds_check(a, b, c, rounds=3, rng=rng)
            assert res.passed is True
            assert res.rounds == 3
            assert res.max_residual <= res.threshold


def test_freivalds_rejects_corrupted():
    """Verify Freivalds rejects various corruption patterns."""
    rng = np.random.default_rng(54321)
    n = 64
    a = rng.uniform(-2.0, 2.0, size=(n, n)).astype(np.float32)
    b = rng.uniform(-2.0, 2.0, size=(n, n)).astype(np.float32)
    c_honest = a @ b

    # 1. Single corrupted element (large magnitude)
    c_elem = c_honest.copy()
    c_elem[0, 0] += 100.0
    assert freivalds_check(a, b, c_elem, rounds=3, rng=rng).passed is False

    # 2. Whole row corrupted
    c_row = c_honest.copy()
    c_row[1, :] += 50.0
    assert freivalds_check(a, b, c_row, rounds=3, rng=rng).passed is False

    # 3. Zeros (lazy node)
    c_zeros = np.zeros_like(c_honest)
    assert freivalds_check(a, b, c_zeros, rounds=3, rng=rng).passed is False

    # 4. Scaled output
    c_scaled = c_honest * 1.2
    assert freivalds_check(a, b, c_scaled, rounds=3, rng=rng).passed is False


def test_freivalds_statistical_bound():
    """Statistical verification of Freivalds miss probability under worst-case error.

    The theoretical worst-case error matrix D = C - AB for Freivalds has a single row
    with two equal non-zero entries (e.g. D[0, 0] = 1, D[0, 1] = 1).
    Since r_0, r_1 in {-1, +1} independently:
      (r_0, r_1) in {(-1, -1), (+1, +1)} -> |r_0 + r_1| = 2 (detected)
      (r_0, r_1) in {(-1, +1), (+1, -1)} -> |r_0 + r_1| = 0 (missed)
    Thus Pr[D @ r = 0] is exactly 0.5 per round, yielding a worst-case miss rate of (1/2)^k.

    Over N = 300 trials, the empirical miss rate with k rounds follows Binomial(N, p=2^-k).
    With N=300 and alpha=0.01, the 3-sigma normal approximation slack is 3 * sqrt(p*(1-p)/N).
    For k=1: p=0.5, slack = 3*sqrt(0.25/300) ~= 0.087 (upper bound <= 0.587).
    For k=2: p=0.25, slack = 3*sqrt(0.1875/300) ~= 0.075 (upper bound <= 0.325).
    For k=3: p=0.125, slack = 3*sqrt(0.109/300) ~= 0.057 (upper bound <= 0.182).
    """
    n = 16
    trials = 300

    for k in (1, 2, 3):
        theoretical_bound = 0.5**k
        # 3.5-sigma slack to guard against stochastic test flakiness
        slack = 3.5 * np.sqrt(theoretical_bound * (1.0 - theoretical_bound) / trials)
        max_allowed_miss_rate = theoretical_bound + slack

        misses = 0
        for seed in range(trials):
            rng = np.random.default_rng(seed + k * 10000)
            a = np.zeros((n, n), dtype=np.float64)
            b = np.zeros((n, n), dtype=np.float64)
            # D = C - A @ B has non-zeros at (0, 0) and (0, 1) with large magnitude
            c = np.zeros((n, n), dtype=np.float64)
            c[0, 0] = 1000.0
            c[0, 1] = 1000.0

            res = freivalds_check(a, b, c, rounds=k, rng=rng)
            if res.passed:
                misses += 1

        empirical_miss_rate = misses / trials
        assert empirical_miss_rate <= max_allowed_miss_rate, (
            f"k={k}: empirical miss rate {empirical_miss_rate:.4f} exceeded "
            f"theoretical bound {theoretical_bound:.4f} + slack {slack:.4f}"
        )


def test_sampled_and_fast_verification():
    """Test verify_result with fast sampled check on elementwise and triad ops."""
    rng = np.random.default_rng(999)
    size = 2000

    # vector_add honest
    a = rng.random(size, dtype=np.float32)
    b = rng.random(size, dtype=np.float32)
    c_honest = a + b

    rep = verify_result("vector_add", [a, b], c_honest, mode="fast", sample=256, rng=rng)
    assert rep.passed is True
    assert "sampled" in rep.method
    assert rep.time_ms >= 0.0

    # vector_add corrupted in 10 elements
    c_corrupted = c_honest.copy()
    c_corrupted[:10] += 5.0

    # 100% sampling must detect 100% of corruptions
    rep_100 = verify_result("vector_add", [a, b], c_corrupted, mode="fast", sample=size, rng=rng)
    assert rep_100.passed is False

    # triad fast verification
    b_vec = rng.random(size, dtype=np.float32)
    c_vec = rng.random(size, dtype=np.float32)
    s_scalar = np.float32(2.5)
    triad_honest = REFERENCE["triad"](b_vec, c_vec, s_scalar)

    rep_triad = verify_result(
        "triad", [b_vec, c_vec, s_scalar], triad_honest, mode="fast", sample=128, rng=rng
    )
    assert rep_triad.passed is True

    # mode="none"
    rep_none = verify_result("triad", [b_vec, c_vec, s_scalar], triad_honest, mode="none")
    assert rep_none.passed is True
    assert rep_none.method == "none"
