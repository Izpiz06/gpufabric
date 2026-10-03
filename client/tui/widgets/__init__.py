"""TUI widgets package."""

from client.tui.widgets.gpu_card import GPUCard
from client.tui.widgets.header import GPUFabricHeader
from client.tui.widgets.worker_table import WorkerEntry, WorkerTable

__all__ = ["GPUFabricHeader", "WorkerTable", "WorkerEntry", "GPUCard"]
