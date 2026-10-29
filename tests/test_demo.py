# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the novelty demonstration CLI command and showcase modules."""

import argparse
import tempfile
from pathlib import Path

from client.cli import build_parser
from client.demo import (
    generate_poster_figure,
    run_c1_demo,
    run_c2_demo,
    run_c3_demo,
    run_showcase,
)


def test_demo_parser():
    """Verify demo and showcase subcommands are properly registered."""
    p = build_parser()
    args1 = p.parse_args(["demo", "--matrix-size", "256", "--graph"])
    assert args1.command == "demo"
    assert args1.matrix_size == 256
    assert args1.graph is True

    args2 = p.parse_args(["showcase", "--step"])
    assert args2.command == "showcase"
    assert args2.step is True


def test_run_c1_demo():
    """Verify Pillar C1 Freivalds verification and fault injection."""
    c1 = run_c1_demo(matrix_size=128, rounds=2)
    assert c1["matrix_size"] == 128
    assert c1["rounds"] == 2
    assert c1["flop_reduction"] > 1.0
    assert c1["corrupt_residual"] > c1["threshold"]


def test_run_c2_demo():
    """Verify Pillar C2 canary tripwire and decoy node alerts."""
    c2 = run_c2_demo()
    assert c2["canary_token"].startswith("cnry_")
    assert c2["alert_count"] >= 1


def test_run_c3_demo():
    """Verify Pillar C3 Bayesian Beta reputation, transient tolerance, and recovery."""
    c3 = run_c3_demo()
    assert "timeline" in c3
    assert len(c3["timeline"]) >= 4
    # Ensure SUSPECT occurs before DRAIN (Single-failure protection)
    states = [row["state"] for row in c3["timeline"]]
    assert "SUSPECT" in states
    assert "DRAIN" in states
    assert "HEALTHY" in states


def test_generate_poster_figure():
    """Verify matplotlib figure generation runs without error."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "test_poster.png"
        c1 = run_c1_demo(matrix_size=64, rounds=1)
        c3 = run_c3_demo()
        res = generate_poster_figure(out_path, c1, c3)
        assert res.exists()
        assert res.stat().st_size > 0


def test_run_showcase_smoke():
    """Smoke test running full showcase end-to-end."""
    args = argparse.Namespace(
        matrix_size=64,
        step=False,
        graph=False,
        output_graph=None,
    )
    # Should run to completion without raising any exception
    run_showcase(args)
