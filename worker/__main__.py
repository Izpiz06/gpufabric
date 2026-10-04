# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Entry point for python -m worker."""

from worker.server import run_worker

if __name__ == "__main__":
    run_worker()
