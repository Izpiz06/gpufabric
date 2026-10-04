# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Worker table widget for GPU Fabric TUI."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rich.text import Text
from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import DataTable, Static


@dataclass
class WorkerEntry:
    target: str  # host:port
    name: str
    worker_id: str
    status: str  # "ONLINE", "UNREACHABLE", "CONNECTING"
    gpu_count: int
    compute_ready: bool
    version: str = ""
    health_state: str = ""
    cuda_version: str = ""
    hostname: str = ""
    error: str = ""
    raw_response: Optional[Any] = None


class WorkerTable(Widget):
    """Table showing discovered/configured GPU Fabric workers."""

    DEFAULT_CSS = """
    WorkerTable {
        height: 1fr;
        border: round $primary;
        padding: 0 1;
        background: $surface;
    }
    .table-title {
        text-style: bold;
        color: $primary-lighten-2;
    }
    .table-subtitle {
        color: $text-muted;
        margin-bottom: 1;
    }
    DataTable {
        height: 1fr;
    }
    """

    class WorkerSelected(Message):
        """Emitted when a worker is selected/highlighted in the table."""

        def __init__(self, target: str, entry: Optional[WorkerEntry] = None):
            super().__init__()
            self.target = target
            self.entry = entry

    class WorkerActivated(Message):
        """Emitted when a worker is double-clicked or Enter is pressed on it."""

        def __init__(self, target: str, entry: Optional[WorkerEntry] = None):
            super().__init__()
            self.target = target
            self.entry = entry

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.worker_entries: Dict[str, WorkerEntry] = {}
        self._selected_target: Optional[str] = None

    def compose(self) -> ComposeResult:
        yield Static("NODES & WORKERS", classes="table-title")
        yield Static(
            "Select node to inspect telemetry • Enter for full details", classes="table-subtitle"
        )
        table = DataTable(cursor_type="row", id="workers-datatable")
        table.zebra_stripes = True
        yield table

    def on_mount(self) -> None:
        table = self.query_one("#workers-datatable", DataTable)
        table.add_column("Status", key="status", width=12)
        table.add_column("Worker", key="worker", width=20)
        table.add_column("Address", key="address", width=22)
        table.add_column("GPUs", key="gpus", width=8)
        table.add_column("Compute Ready", key="compute", width=15)

    def update_workers(self, worker_entries: List[WorkerEntry]) -> None:
        """Update the table rows with new worker entries."""
        table = self.query_one("#workers-datatable", DataTable)
        self.worker_entries = {w.target: w for w in worker_entries}

        prev_target = self._selected_target
        table.clear()

        for w in worker_entries:
            if w.status == "ONLINE":
                if w.health_state in ("DEGRADED", 2, "2"):
                    status_text = Text("▲ DEGRADED", style="bold yellow")
                elif w.health_state in ("UNAVAILABLE", 3, "3"):
                    status_text = Text("✖ UNAVAIL", style="bold red")
                elif w.health_state in ("HEALTHY", 1, "1"):
                    status_text = Text("● HEALTHY", style="bold green")
                else:
                    status_text = Text("● ONLINE", style="bold green")
            elif w.status == "CONNECTING":
                status_text = Text("◌ SYNCING", style="bold cyan")
            else:
                status_text = Text("✖ OFFLINE", style="bold red")

            gpus_text = f"{w.gpu_count} GPU" if w.gpu_count == 1 else f"{w.gpu_count} GPUs"
            if w.status != "ONLINE":
                gpus_text = "-"

            compute_text = (
                Text("✓ Ready", style="green")
                if w.compute_ready
                else Text("✗ No CuPy", style="red" if w.status == "ONLINE" else "dim")
            )

            worker_label = w.name or w.worker_id or w.target
            table.add_row(
                status_text,
                worker_label,
                w.target,
                gpus_text,
                compute_text,
                key=w.target,
            )

        if worker_entries:
            # Restore previous selection or select first
            target_to_select = (
                prev_target
                if (prev_target and prev_target in self.worker_entries)
                else worker_entries[0].target
            )
            try:
                row_idx = list(self.worker_entries.keys()).index(target_to_select)
                table.move_cursor(row=row_idx)
                self._selected_target = target_to_select
                self.post_message(
                    self.WorkerSelected(target_to_select, self.worker_entries.get(target_to_select))
                )
            except (ValueError, IndexError):
                pass
        else:
            self._selected_target = None
            self.post_message(self.WorkerSelected("", None))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key and event.row_key.value in self.worker_entries:
            target = event.row_key.value
            self._selected_target = target
            self.post_message(self.WorkerSelected(target, self.worker_entries.get(target)))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key and event.row_key.value in self.worker_entries:
            target = event.row_key.value
            self._selected_target = target
            self.post_message(self.WorkerActivated(target, self.worker_entries.get(target)))
