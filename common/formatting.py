# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Data formatting utilities."""


def bytes_to_human(n_bytes: int) -> str:
    """Convert raw bytes count to human-readable string representation."""
    if n_bytes <= 0:
        return "0.00 B"
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    val = float(n_bytes)
    idx = 0
    while val >= 1024.0 and idx < len(units) - 1:
        val /= 1024.0
        idx += 1
    return f"{val:.2f} {units[idx]}"
