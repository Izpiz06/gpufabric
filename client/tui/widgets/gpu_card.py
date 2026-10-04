"""GPU card and live telemetry widget for GPU Fabric TUI."""

from typing import Any, Dict, List, Optional

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Label, ProgressBar, Select, Static

from common.formatting import bytes_to_human


class GPUCard(Widget):
    """Widget displaying live telemetry and hardware specifications for a selected GPU."""

    DEFAULT_CSS = """
    GPUCard {
        height: 1fr;
        border: round $accent;
        padding: 0 1;
        background: $surface;
    }
    .card-title {
        text-style: bold;
        color: $accent-lighten-2;
    }
    .card-subtitle {
        color: $text-muted;
        margin-bottom: 1;
    }
    .gpu-selector-row {
        height: 3;
        margin-bottom: 1;
        align: left middle;
    }
    .metric-row {
        height: 2;
        margin: 0;
        align: left middle;
    }
    .metric-name {
        width: 18;
        color: $text-muted;
    }
    .metric-value {
        width: 26;
        text-style: bold;
        color: $text;
    }
    .metric-bar {
        width: 1fr;
    }
    .info-grid {
        height: auto;
        margin-top: 1;
        border-top: solid $panel;
        padding-top: 1;
    }
    .empty-state {
        height: 100%;
        align: center middle;
        text-align: center;
        color: $text-muted;
        padding: 2;
    }
    """

    class DeviceChanged(Message):
        """Emitted when user selects a different GPU device index."""

        def __init__(self, device_index: int):
            super().__init__()
            self.device_index = device_index

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_worker_target: Optional[str] = None
        self.gpus: List[Any] = []
        self.selected_device_index: int = 0
        self.status_data: Optional[Dict[str, Any]] = None

    def compose(self) -> ComposeResult:
        yield Static("GPU TELEMETRY & HARDWARE", classes="card-title")
        yield Static(
            "Live device load, memory, thermals, and workload status", classes="card-subtitle"
        )
        with Container(id="gpu-card-content"):
            with Horizontal(classes="gpu-selector-row"):
                yield Label("GPU Device: ", classes="metric-name")
                yield Select(
                    [("Default (0)", 0)],
                    id="gpu-device-select",
                    prompt="Select Device",
                    allow_blank=True,
                )

            with Vertical(id="telemetry-gauges"):
                # VRAM
                with Horizontal(classes="metric-row"):
                    yield Label("VRAM:", classes="metric-name")
                    yield Label("0.0 / 0.0 GB (0%)", id="vram-text", classes="metric-value")
                    yield ProgressBar(
                        total=100,
                        show_percentage=False,
                        show_eta=False,
                        id="vram-bar",
                        classes="metric-bar",
                    )

                # Core Utilization
                with Horizontal(classes="metric-row"):
                    yield Label("GPU Core:", classes="metric-name")
                    yield Label("0%", id="util-text", classes="metric-value")
                    yield ProgressBar(
                        total=100,
                        show_percentage=False,
                        show_eta=False,
                        id="util-bar",
                        classes="metric-bar",
                    )

                # Memory Utilization
                with Horizontal(classes="metric-row"):
                    yield Label("Memory Bus:", classes="metric-name")
                    yield Label("0%", id="mem-util-text", classes="metric-value")
                    yield ProgressBar(
                        total=100,
                        show_percentage=False,
                        show_eta=False,
                        id="mem-util-bar",
                        classes="metric-bar",
                    )

                # Power Draw
                with Horizontal(classes="metric-row"):
                    yield Label("Power Draw:", classes="metric-name")
                    yield Label("0 W / 0 W (0%)", id="power-text", classes="metric-value")
                    yield ProgressBar(
                        total=100,
                        show_percentage=False,
                        show_eta=False,
                        id="power-bar",
                        classes="metric-bar",
                    )

                # Specs Grid
                with Vertical(classes="info-grid"):
                    with Horizontal(classes="metric-row"):
                        yield Label("Temperature:", classes="metric-name")
                        yield Label("-", id="temp-text", classes="metric-value")
                        yield Label("Active Tasks:", classes="metric-name")
                        yield Label("0", id="tasks-text", classes="metric-value")

                    with Horizontal(classes="metric-row"):
                        yield Label("Workload:", classes="metric-name")
                        yield Label("IDLE", id="workload-text", classes="metric-value")
                        yield Label("Availability:", classes="metric-name")
                        yield Label("Available", id="avail-text", classes="metric-value")

                    with Horizontal(classes="metric-row"):
                        yield Label("Compute Cap:", classes="metric-name")
                        yield Label("-", id="cc-text", classes="metric-value")
                        yield Label("Driver Version:", classes="metric-name")
                        yield Label("-", id="driver-text", classes="metric-value")

                    with Horizontal(classes="metric-row"):
                        yield Label("Peak Bandwidth:", classes="metric-name")
                        yield Label("-", id="bw-text", classes="metric-value")
                        yield Label("Worker:", classes="metric-name")
                        yield Label("-", id="worker-text", classes="metric-value")

            yield Static(
                "No GPU selected or worker offline", id="empty-state-msg", classes="empty-state"
            )

    def on_mount(self) -> None:
        self._show_empty("No worker selected. Discover workers with [D] or refresh with [R].")

    def _show_empty(self, message: str) -> None:
        try:
            self.query_one("#telemetry-gauges").display = False
            self.query_one(".gpu-selector-row").display = False
            empty_msg = self.query_one("#empty-state-msg", Static)
            empty_msg.update(message)
            empty_msg.display = True
        except Exception:
            pass

    def _show_content(self) -> None:
        try:
            self.query_one("#empty-state-msg").display = False
            self.query_one(".gpu-selector-row").display = True
            self.query_one("#telemetry-gauges").display = True
        except Exception:
            pass

    def on_select_changed(self, event: Select.Changed) -> None:
        if (
            event.select.id == "gpu-device-select"
            and event.value is not None
            and event.value != Select.BLANK
        ):
            try:
                dev_idx = int(event.value)
                if dev_idx != self.selected_device_index:
                    self.selected_device_index = dev_idx
                    self.post_message(self.DeviceChanged(dev_idx))
            except (ValueError, TypeError):
                pass

    def set_worker(self, target: str, gpus_response: Optional[Any], error: str = "") -> None:
        """Update GPUCard with inventory response from worker."""
        self.current_worker_target = target

        if error or not gpus_response:
            err_msg = error or "Worker is offline or unreachable"
            self._show_empty(f"[red]● Worker {target}[/red]\n\n{err_msg}")
            return

        self.gpus = list(getattr(gpus_response, "gpus", []))
        if not self.gpus:
            self._show_empty(
                f"[yellow]Worker {target}[/yellow]\n\nNo NVIDIA GPUs detected on this worker."
            )
            return

        self._show_content()

        # Update Select options
        select_options = [(f"GPU {g.device_index}: {g.name}", g.device_index) for g in self.gpus]
        select = self.query_one("#gpu-device-select", Select)
        select.set_options(select_options)

        # Retain selected device or default to first
        valid_indices = [g.device_index for g in self.gpus]
        if self.selected_device_index not in valid_indices:
            self.selected_device_index = valid_indices[0]

        select.value = self.selected_device_index

        # Update static specs from gpus_response
        driver = getattr(gpus_response, "driver_version", "N/A") or "N/A"
        self.query_one("#driver-text", Label).update(driver)
        self.query_one("#worker-text", Label).update(target)

        self.update_telemetry()

    def update_telemetry(self, status: Optional[Any] = None) -> None:
        """Update live telemetry metrics for current device."""
        if not self.gpus:
            return

        gpu_spec = next(
            (g for g in self.gpus if g.device_index == self.selected_device_index), self.gpus[0]
        )

        # Extract values
        total_vram = getattr(status, "total_vram_bytes", 0) or getattr(
            gpu_spec, "total_vram_bytes", 0
        )
        free_vram = getattr(status, "free_vram_bytes", 0) or getattr(gpu_spec, "free_vram_bytes", 0)
        used_vram = getattr(status, "used_vram_bytes", 0) or (
            total_vram - free_vram if total_vram >= free_vram else 0
        )

        gpu_util = getattr(status, "gpu_utilization_pct", None)
        if gpu_util is None:
            gpu_util = getattr(gpu_spec, "gpu_utilization_pct", 0)

        mem_util = getattr(status, "memory_utilization_pct", 0)
        temp_c = getattr(status, "temperature_c", None)
        if temp_c is None:
            temp_c = getattr(gpu_spec, "temperature_c", 0)

        active_tasks = getattr(status, "active_tasks", 0)
        compute_cap = getattr(gpu_spec, "compute_capability", "N/A") or "N/A"
        power_w = getattr(status, "power_usage_w", 0) or getattr(gpu_spec, "power_usage_w", 0)
        power_limit_w = getattr(status, "power_limit_w", 0) or getattr(gpu_spec, "power_limit_w", 0)
        workload = (
            getattr(status, "current_workload", "")
            or getattr(gpu_spec, "current_workload", "")
            or "IDLE"
        )
        available = getattr(status, "available", True)

        # Update UI elements
        vram_pct = int((used_vram / total_vram * 100)) if total_vram > 0 else 0
        vram_text = f"{bytes_to_human(used_vram)} / {bytes_to_human(total_vram)} ({vram_pct}%)"
        self.query_one("#vram-text", Label).update(vram_text)
        self.query_one("#vram-bar", ProgressBar).progress = vram_pct

        self.query_one("#util-text", Label).update(f"{gpu_util}%")
        self.query_one("#util-bar", ProgressBar).progress = int(gpu_util)

        self.query_one("#mem-util-text", Label).update(f"{mem_util}%")
        self.query_one("#mem-util-bar", ProgressBar).progress = int(mem_util)

        # Power telemetry
        if power_limit_w > 0:
            power_pct = int(min(100, (power_w / power_limit_w) * 100))
            power_text = f"{power_w} W / {power_limit_w} W ({power_pct}%)"
        elif power_w > 0:
            power_pct = 0
            power_text = f"{power_w} W"
        else:
            power_pct = 0
            power_text = "N/A"
        try:
            self.query_one("#power-text", Label).update(power_text)
            self.query_one("#power-bar", ProgressBar).progress = power_pct
        except Exception:
            pass

        # Temperature color
        if temp_c < 60:
            temp_text = Text(f"{temp_c} °C (Cool)", style="bold green")
        elif temp_c < 80:
            temp_text = Text(f"{temp_c} °C (Warm)", style="bold yellow")
        else:
            temp_text = Text(f"{temp_c} °C (Hot)", style="bold red")
        self.query_one("#temp-text", Label).update(temp_text)

        self.query_one("#tasks-text", Label).update(str(active_tasks))
        self.query_one("#cc-text", Label).update(compute_cap)

        # Workload & availability
        try:
            if workload == "IDLE":
                wl_text = Text("IDLE", style="dim")
            else:
                wl_text = Text(f"⚡ {workload}", style="bold cyan")
            self.query_one("#workload-text", Label).update(wl_text)

            av_text = (
                Text("✓ Ready", style="bold green")
                if available
                else Text("✗ Busy", style="bold red")
            )
            self.query_one("#avail-text", Label).update(av_text)
        except Exception:
            pass
