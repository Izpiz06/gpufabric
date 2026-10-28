# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Worker-side fault injection engine for research and detection evaluation."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("gpufabric.worker.faults")

VALID_FAULT_MODES = ("none", "random_elements", "bitflip", "zeros", "stale", "scale")


class FaultInjector:
    """Injects intentional computational faults into GPU computation outputs.

    Research tool only: used to evaluate verification detection sensitivity
    (e.g. Freivalds, sampled checks) under controlled synthetic corruptions.
    """

    def __init__(
        self,
        mode: str = "none",
        rate: float = 0.001,
        magnitude: float = 1.0,
        seed: Optional[int] = None,
    ):
        if mode not in VALID_FAULT_MODES:
            raise ValueError(f"Unknown fault mode '{mode}'; valid modes: {VALID_FAULT_MODES}")
        self.mode = mode
        self.rate = float(rate)
        self.magnitude = float(magnitude)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._previous_result: Optional[np.ndarray] = None

    @property
    def is_active(self) -> bool:
        """True if fault injection is actively corrupting results."""
        return self.mode != "none"

    def inject(self, result: np.ndarray) -> np.ndarray:
        """Apply configured fault mode to computation result array."""
        if not self.is_active:
            return result

        if self.mode == "zeros":
            return np.zeros_like(result)

        if self.mode == "scale":
            return result * (1.0 + self.magnitude)

        if self.mode == "stale":
            if (
                self._previous_result is not None
                and self._previous_result.shape == result.shape
                and self._previous_result.dtype == result.dtype
            ):
                corrupted = self._previous_result.copy()
            else:
                corrupted = np.zeros_like(result)
            self._previous_result = result.copy()
            return corrupted

        res = result.copy()
        flat = res.ravel()
        total = flat.size
        k = max(1, int(round(total * self.rate))) if (self.rate > 0 and total > 0) else 0

        if k == 0:
            return res

        indices = self.rng.choice(total, size=k, replace=False)

        if self.mode == "random_elements":
            zeros_mask = flat[indices] == 0
            flat[indices[zeros_mask]] += self.magnitude
            flat[indices[~zeros_mask]] *= 1.0 + self.magnitude
            return res

        if self.mode == "bitflip":
            if res.dtype == np.float32:
                view = flat.view(np.uint32)
                for idx in indices:
                    bit = int(self.rng.integers(0, 32))
                    view[idx] ^= 1 << bit
            elif res.dtype == np.float64:
                view = flat.view(np.uint64)
                for idx in indices:
                    bit = int(self.rng.integers(0, 64))
                    view[idx] ^= 1 << bit
            else:
                flat[indices] += 1
            return res

        return res
