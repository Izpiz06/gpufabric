# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GPU Fabric Terminal User Interface (TUI)."""

import asyncio
from unittest.mock import patch

from textual.widgets import Input, Label, Select, Static

from client.tui.app import GPUFabricApp
from client.tui.screens.benchmark import BenchmarkModal
from client.tui.screens.details import DetailsModal
from client.tui.widgets.gpu_card import GPUCard
from client.tui.widgets.header import GPUFabricHeader
from client.tui.widgets.worker_table import WorkerTable
from common.discovery import DiscoveredWorker
from common.gpufabric_pb2 import (
    BenchmarkResponse,
    GPUDevice,
    GPUStatusResponse,
    ListGPUsResponse,
)


def _make_mock_gpu_response(worker_id="test-worker", gpus_count=1):
    gpus = []
    for i in range(gpus_count):
        gpus.append(
            GPUDevice(
                device_index=i,
                name=f"NVIDIA RTX 4090 (Mock {i})",
                total_vram_bytes=24_000_000_000,
                free_vram_bytes=16_000_000_000,
                compute_capability="8.9",
                gpu_utilization_pct=45,
                temperature_c=55,
            )
        )
    return ListGPUsResponse(
        worker_id=worker_id,
        version="0.1.0",
        driver_version="550.54.14",
        compute_ready=True,
        gpus=gpus,
    )


def test_tui_startup_empty():
    """Test TUI starts up cleanly in empty state without crashing."""

    async def _test():
        app = GPUFabricApp(auto_discover_on_mount=False)
        async with app.run_test():
            header = app.query_one(GPUFabricHeader)
            assert header.total_workers == 0
            assert header.online_workers == 0
            assert header.total_gpus == 0

            table = app.query_one(WorkerTable)
            assert len(table.worker_entries) == 0

            gpu_card = app.query_one(GPUCard)
            assert gpu_card.current_worker_target is None

    asyncio.run(_test())


def test_tui_seed_workers_and_metrics():
    """Test TUI with pre-seeded worker addresses."""

    async def _test():
        app = GPUFabricApp(
            seed_workers=["192.168.1.50:50051", "192.168.1.51:50051"],
            auto_discover_on_mount=False,
        )
        async with app.run_test():
            header = app.query_one(GPUFabricHeader)
            assert header.total_workers == 2

            table = app.query_one(WorkerTable)
            assert len(table.worker_entries) == 2
            assert "192.168.1.50:50051" in table.worker_entries
            assert "192.168.1.51:50051" in table.worker_entries

    asyncio.run(_test())


def test_tui_handle_refresh_results_and_selection():
    """Test updating workers with list_gpus response and verifying GPU card rendering."""

    async def _test():
        app = GPUFabricApp(
            seed_workers=["192.168.1.50:50051"],
            auto_discover_on_mount=False,
        )
        async with app.run_test():
            resp = _make_mock_gpu_response("worker-alpha", gpus_count=2)
            app._handle_refresh_results({"192.168.1.50:50051": (resp, None)})

            header = app.query_one(GPUFabricHeader)
            assert header.total_workers == 1
            assert header.online_workers == 1
            assert header.total_gpus == 2

            table = app.query_one(WorkerTable)
            entry = table.worker_entries["192.168.1.50:50051"]
            assert entry.status == "ONLINE"
            assert entry.gpu_count == 2
            assert entry.compute_ready is True

            gpu_card = app.query_one(GPUCard)
            assert gpu_card.current_worker_target == "192.168.1.50:50051"
            assert len(gpu_card.gpus) == 2
            assert str(app.query_one("#driver-text", Label).render()) == "550.54.14"

    asyncio.run(_test())


def test_tui_unreachable_worker_handling():
    """Test graceful handling when a worker fails to connect."""

    async def _test():
        app = GPUFabricApp(
            seed_workers=["192.168.1.99:50051"],
            auto_discover_on_mount=False,
        )
        async with app.run_test():
            app._handle_refresh_results(
                {"192.168.1.99:50051": (None, "Connection refused: [Errno 111]")}
            )

            table = app.query_one(WorkerTable)
            entry = table.worker_entries["192.168.1.99:50051"]
            assert entry.status == "UNREACHABLE"
            assert "Connection refused" in entry.error

            header = app.query_one(GPUFabricHeader)
            assert header.total_workers == 1
            assert header.online_workers == 0

            gpu_card = app.query_one(GPUCard)
            empty_msg = gpu_card.query_one("#empty-state-msg", Static)
            assert "Connection refused" in str(empty_msg.render())

    asyncio.run(_test())


def test_tui_discovery_handling():
    """Test handling of discovered LAN candidates."""

    async def _test():
        app = GPUFabricApp(auto_discover_on_mount=False)
        async with app.run_test():
            discovered = [
                DiscoveredWorker(
                    name="lan-worker",
                    address="192.168.1.60",
                    port=50051,
                    worker_id="id-60",
                    version="0.1.0",
                    status="ONLINE",
                    gpu_available=True,
                )
            ]

            with patch.object(app, "_refresh_workers_task"):
                app._handle_discovered_workers(discovered)

            assert "192.168.1.60:50051" in app.known_workers
            entry = app.known_workers["192.168.1.60:50051"]
            assert entry.name == "lan-worker"
            assert entry.status == "ONLINE"

    asyncio.run(_test())


def test_tui_telemetry_update():
    """Test updating live telemetry on GPUCard."""

    async def _test():
        app = GPUFabricApp(
            seed_workers=["192.168.1.50:50051"],
            auto_discover_on_mount=False,
        )
        async with app.run_test():
            resp = _make_mock_gpu_response("worker-alpha", gpus_count=1)
            app._handle_refresh_results({"192.168.1.50:50051": (resp, None)})

            # Feed live status
            status = GPUStatusResponse(
                device_index=0,
                gpu_utilization_pct=88,
                memory_utilization_pct=60,
                total_vram_bytes=24_000_000_000,
                free_vram_bytes=6_000_000_000,
                used_vram_bytes=18_000_000_000,
                temperature_c=72,
                active_tasks=3,
            )
            app._update_gpu_card_status("192.168.1.50:50051", status)

            assert str(app.query_one("#util-text", Label).render()) == "88%"
            assert str(app.query_one("#mem-util-text", Label).render()) == "60%"
            assert str(app.query_one("#tasks-text", Label).render()) == "3"

    asyncio.run(_test())


def test_details_modal_render():
    """Test DetailsModal rendering with worker specs and GPU list."""

    async def _test():
        resp = _make_mock_gpu_response("worker-spec-test", gpus_count=2)
        modal = DetailsModal(target="192.168.1.50:50051", gpus_response=resp)

        app = GPUFabricApp(auto_discover_on_mount=False)
        async with app.run_test() as pilot:
            app.push_screen(modal)
            await pilot.pause()

            content = modal.query_one("#details-content", Static)
            assert content is not None

    asyncio.run(_test())


def test_benchmark_modal_interaction():
    """Test BenchmarkModal options and execution flow."""

    async def _test():
        modal = BenchmarkModal(
            online_workers=["192.168.1.50:50051"],
            default_target="192.168.1.50:50051",
        )

        app = GPUFabricApp(auto_discover_on_mount=False)
        async with app.run_test() as pilot:
            app.push_screen(modal)
            await pilot.pause()

            # Test benchmark type change updates size input
            type_select = modal.query_one("#bench-type-select", Select)
            type_select.value = "matmul"
            await pilot.pause()

            size_input = modal.query_one("#bench-size-input", Input)
            assert size_input.value == BenchmarkModal.DEFAULT_SIZES["matmul"]

            # Test successful benchmark callback
            resp = BenchmarkResponse(
                benchmark=1,
                size=1000,
                device_index=0,
                repeats=5,
                best_ms=1.25,
                mean_ms=1.30,
                bandwidth_gb_s=850.5,
                peak_bandwidth_gb_s=1008.0,
            )
            modal._handle_benchmark_success(
                "192.168.1.50:50051", "triad", resp, size=1000, repeats=5, device_idx=0
            )

            res_text = modal.query_one("#bench-results-text", Static)
            assert res_text is not None

    asyncio.run(_test())


def test_tui_power_and_workload_telemetry():
    """Test updating power draw and active workload on GPUCard."""

    async def _test():
        app = GPUFabricApp(
            seed_workers=["192.168.1.50:50051"],
            auto_discover_on_mount=False,
        )
        async with app.run_test():
            resp = _make_mock_gpu_response("worker-alpha", gpus_count=1)
            app._handle_refresh_results({"192.168.1.50:50051": (resp, None)})

            status = GPUStatusResponse(
                device_index=0,
                gpu_utilization_pct=50,
                memory_utilization_pct=40,
                total_vram_bytes=24_000_000_000,
                free_vram_bytes=12_000_000_000,
                used_vram_bytes=12_000_000_000,
                temperature_c=65,
                active_tasks=1,
                power_usage_w=120,
                power_limit_w=300,
                available=True,
                current_workload="Compute:matmul",
            )
            app._update_gpu_card_status("192.168.1.50:50051", status)

            assert "120 W / 300 W" in str(app.query_one("#power-text", Label).render())
            assert "Compute:matmul" in str(app.query_one("#workload-text", Label).render())
            assert "Ready" in str(app.query_one("#avail-text", Label).render())

    asyncio.run(_test())


def test_tui_degraded_worker_badge():
    """Test worker table displaying degraded status badge."""

    async def _test():
        app = GPUFabricApp(
            seed_workers=["192.168.1.50:50051"],
            auto_discover_on_mount=False,
        )
        async with app.run_test():
            resp = _make_mock_gpu_response("worker-alpha", gpus_count=1)
            resp.health_state = 2  # DEGRADED
            app._handle_refresh_results({"192.168.1.50:50051": (resp, None)})

            table = app.query_one(WorkerTable)
            entry = table.worker_entries["192.168.1.50:50051"]
            assert entry.health_state == 2

    asyncio.run(_test())
