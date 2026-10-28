# Copyright (c) 2026 Shreyas Mene
# SPDX-License-Identifier: Apache-2.0

"""Reference results and error checks for verifying remote GPU output."""

import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

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


@dataclass
class FreivaldsResult:
    """Outcome of Freivalds' probabilistic matrix multiplication verification."""

    passed: bool
    rounds: int
    max_residual: float
    threshold: float


@dataclass
class VerificationReport:
    """Structured report returned by verify_result."""

    method: str
    passed: bool
    time_ms: float
    detail: str


def freivalds_check(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    rounds: int = 3,
    rng: Optional[np.random.Generator] = None,
    threshold_factor: float = 10.0,
) -> FreivaldsResult:
    """Verify C = A @ B via Freivalds' probabilistic algorithm in O(k * n^2).

    For each round k, draws a Rademacher random vector r in {-1, +1}^p.
    Computes in float64 on the CPU:
        lhs = C @ r
        rhs = A @ (B @ r)
    without ever forming the full A @ B product.

    Theoretical Error Bound:
    Higham (2002) Theorem 3.5 gives ||fl(A B) - A B||_inf <= n * eps * ||A||_inf * ||B||_inf.
    When projected onto a random Rademacher vector r in {-1, 1}^p, the variance across the
    p columns yields an additional sqrt(p) concentration factor under the Central Limit
    Theorem. The base physical scale is:
        scale = eps * n * ||A||_max * ||B||_max * sqrt(p)
    where eps = finfo(a.dtype).eps and n = A.shape[1].
    threshold_factor defaults to 10.0, providing a ~10x margin above honest floating-point
    rounding and accumulation differences while reliably flagging bit flips and corruption.
    """
    if a.ndim != 2 or b.ndim != 2 or c.ndim != 2:
        raise ValueError(
            f"Freivalds check expects 2D matrices, got {a.shape}, {b.shape}, {c.shape}"
        )
    if a.shape[0] != c.shape[0] or a.shape[1] != b.shape[0] or b.shape[1] != c.shape[1]:
        raise ValueError(f"Matrix dimension mismatch: {a.shape} @ {b.shape} != {c.shape}")

    if rounds < 1:
        raise ValueError("rounds must be >= 1")

    if rng is None:
        rng = np.random.default_rng()

    m, n = a.shape
    p = b.shape[1]

    # Convert operands to float64 for high-precision verification on CPU
    a_64 = np.asarray(a, dtype=np.float64)
    b_64 = np.asarray(b, dtype=np.float64)
    c_64 = np.asarray(c, dtype=np.float64)

    eps = float(np.finfo(a.dtype).eps)
    norm_a = max(float(np.max(np.abs(a_64), initial=0.0)), 1e-12)
    norm_b = max(float(np.max(np.abs(b_64), initial=0.0)), 1e-12)
    scale = eps * float(n) * norm_a * norm_b * max(float(np.sqrt(p)), 1.0)
    threshold = float(threshold_factor * scale)

    max_residual = 0.0
    passed = True

    for _ in range(rounds):
        r = rng.choice([-1.0, 1.0], size=p)
        lhs = c_64 @ r
        rhs = a_64 @ (b_64 @ r)
        res = float(np.max(np.abs(lhs - rhs), initial=0.0))
        if res > max_residual:
            max_residual = res
        if res > threshold:
            passed = False
            break

    return FreivaldsResult(
        passed=passed,
        rounds=rounds,
        max_residual=max_residual,
        threshold=threshold,
    )


def verify_result(
    op: str,
    inputs: Sequence[Any],
    result: Any,
    mode: str = "fast",
    rounds: int = 3,
    sample: int = 1024,
    rng: Optional[np.random.Generator] = None,
    threshold_factor: float = 10.0,
) -> VerificationReport:
    """Verify GPU compute output against CPU references under selectable modes.

    Modes:
      - 'full': Standard NumPy reference calculation on CPU with relative_error().
      - 'fast':
          * matmul: Freivalds' algorithm in O(k * n^2).
          * vector_dot: Full CPU recomputation (already O(n)).
          * vector_add, vector_mul, matrix_add, triad: Randomly sample `sample`
            elements and compare against CPU reference.
      - 'none': Skip verification entirely.
    """
    if mode not in ("full", "fast", "none"):
        raise ValueError(f"Unknown verification mode '{mode}', expected 'full', 'fast', or 'none'")

    t0 = time.perf_counter()

    if mode == "none":
        time_ms = (time.perf_counter() - t0) * 1000.0
        return VerificationReport(
            method="none",
            passed=True,
            time_ms=time_ms,
            detail="Verification skipped (mode=none)",
        )

    res_arr = np.asarray(result)

    if mode == "full" or (mode == "fast" and op == "vector_dot"):
        expected = REFERENCE[op](*inputs)
        err = relative_error(res_arr, expected)
        tol = tolerance(op, res_arr.dtype)
        passed = bool(err <= tol)
        time_ms = (time.perf_counter() - t0) * 1000.0
        method_name = "full" if mode == "full" else "full (O(n) vector_dot)"
        return VerificationReport(
            method=method_name,
            passed=passed,
            time_ms=time_ms,
            detail=f"Relative error: {err:.2e} (tolerance: {tol:.2e})",
        )

    if mode == "fast" and op == "matmul":
        a, b = inputs[0], inputs[1]
        f_res = freivalds_check(
            a,
            b,
            res_arr,
            rounds=rounds,
            rng=rng,
            threshold_factor=threshold_factor,
        )
        time_ms = (time.perf_counter() - t0) * 1000.0
        return VerificationReport(
            method=f"freivalds (rounds={rounds})",
            passed=f_res.passed,
            time_ms=time_ms,
            detail=(
                f"Freivalds max residual: {f_res.max_residual:.2e} "
                f"(threshold: {f_res.threshold:.2e}, factor: {threshold_factor})"
            ),
        )

    # Fast sampled verification for element-wise and triad operations
    if rng is None:
        rng = np.random.default_rng()

    total_elements = res_arr.size
    sample_count = min(sample, total_elements)

    if sample_count == total_elements:
        indices = np.arange(total_elements)
    else:
        indices = rng.choice(total_elements, size=sample_count, replace=False)

    sampled_inputs = []
    for inp in inputs:
        inp_arr = np.asarray(inp)
        if inp_arr.ndim == 0:
            sampled_inputs.append(inp_arr)
        else:
            sampled_inputs.append(inp_arr.ravel()[indices])

    expected_sample = REFERENCE[op](*sampled_inputs)
    result_sample = res_arr.ravel()[indices]

    err = relative_error(result_sample, expected_sample)
    tol = tolerance(op, res_arr.dtype)
    passed = bool(err <= tol)
    time_ms = (time.perf_counter() - t0) * 1000.0

    return VerificationReport(
        method=f"sampled ({sample_count}/{total_elements} elements)",
        passed=passed,
        time_ms=time_ms,
        detail=f"Sampled relative error: {err:.2e} (tolerance: {tol:.2e})",
    )
