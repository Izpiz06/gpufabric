"""Details modal screen for GPU Fabric TUI."""

from typing import Any, Optional

from rich.table import Table
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from common.formatting import bytes_to_human


class DetailsModal(ModalScreen):
    """Modal screen showing comprehensive worker and GPU details."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("enter", "dismiss", "Close", show=False),
    ]

    DEFAULT_CSS = """
    DetailsModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }
    .modal-dialog {
        width: 85%;
        height: 85%;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    .modal-title {
        text-style: bold;
        color: $accent;
        padding-bottom: 1;
        border-bottom: solid $primary;
    }
    .details-scroll {
        height: 1fr;
        margin: 1 0;
    }
    .modal-footer {
        height: 3;
        align: right middle;
    }
    """

    def __init__(
        self,
        target: str,
        gpus_response: Optional[Any] = None,
        health_info: Optional[Any] = None,
        error: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target = target
        self.gpus_response = gpus_response
        self.health_info = health_info
        self.error = error

    def compose(self) -> ComposeResult:
        with Container(classes="modal-dialog"):
            yield Static(f"📋 Worker Details: {self.target}", classes="modal-title")
            with VerticalScroll(classes="details-scroll"):
                yield Static(id="details-content")
            with Horizontal(classes="modal-footer"):
                yield Button("Close (Esc)", variant="primary", id="close-btn")

    def on_mount(self) -> None:
        content_static = self.query_one("#details-content", Static)

        if self.error:
            content_static.update(
                f"[bold red]Worker Unreachable[/bold red]\n\n"
                f"Address: {self.target}\n"
                f"Error: {self.error}\n\n"
                f"Make sure the worker is running and reachable over the network."
            )
            return

        worker_id = getattr(self.gpus_response, "worker_id", "N/A") or getattr(
            self.health_info, "worker_id", "N/A"
        )
        version = getattr(self.gpus_response, "version", "N/A") or getattr(
            self.health_info, "version", "N/A"
        )
        driver = getattr(self.gpus_response, "driver_version", "N/A") or "N/A"
        compute_ready = getattr(self.gpus_response, "compute_ready", False)

        # Worker Table
        wt = Table(title="Worker Information", show_header=True, expand=True)
        wt.add_column("Property", style="cyan", width=24)
        wt.add_column("Value", style="bold")
        wt.add_row("Address", self.target)
        wt.add_row("Worker ID", worker_id)
        wt.add_row("API Version", version)
        wt.add_row("NVIDIA Driver Version", driver)
        wt.add_row(
            "CuPy Compute Engine",
            "[bold green]Ready (Available)[/bold green]"
            if compute_ready
            else "[bold yellow]Not available (NVML only)[/bold yellow]",
        )

        gpus = list(getattr(self.gpus_response, "gpus", []))
        # GPU Table
        gt = Table(title=f"Detected GPUs ({len(gpus)})", show_header=True, expand=True)
        gt.add_column("Device", style="cyan", width=8)
        gt.add_column("Name", style="bold", width=24)
        gt.add_column("VRAM Total", width=14)
        gt.add_column("VRAM Free", width=14)
        gt.add_column("Compute Cap", width=12)
        gt.add_column("Util %", width=8)
        gt.add_column("Temp", width=8)

        for g in gpus:
            gt.add_row(
                f"GPU {g.device_index}",
                g.name,
                bytes_to_human(g.total_vram_bytes),
                bytes_to_human(g.free_vram_bytes),
                g.compute_capability or "N/A",
                f"{g.gpu_utilization_pct}%",
                f"{g.temperature_c} °C",
            )

        if not gpus:
            gt.add_row("-", "No GPUs detected on this worker", "-", "-", "-", "-", "-")

        from rich.console import Group
        from rich.text import Text

        content_static.update(Group(wt, Text("\n"), gt))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-btn":
            self.dismiss()
