# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Local system and GPU hardware monitoring TUI for GPU Fabric host."""

import ipaddress
import platform
import socket
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Footer, Label, ProgressBar, Select, Static

from common.formatting import bytes_to_human
from common.tls import default_tls_dir, detect_host_addresses
from worker.diagnostics import CudaDiagnostics, probe_cuda_environment
from worker.executor import GPUExecutor
from worker.gpu import GPUManager
from worker.state import WorkerState


def get_local_host_info() -> Dict[str, Any]:
    """Gather general host identity, platform, network, and TLS details without network connections."""
    hostname = socket.gethostname()
    try:
        raw_addrs = detect_host_addresses(include_localhost=False, include_hostname=False)
        ips: List[str] = []
        for addr in raw_addrs:
            try:
                ipaddress.ip_address(addr)
                if addr not in ips:
                    ips.append(addr)
            except ValueError:
                pass
    except Exception:
        ips = []

    try:
        tls_dir = default_tls_dir()
        if tls_dir.exists():
            certs = sorted([f.name for f in tls_dir.iterdir() if f.suffix in (".crt", ".key")])
            tls_desc = ", ".join(certs) if certs else "Directory exists (no certs)"
        else:
            tls_desc = "Not initialized"
    except Exception:
        tls_dir = Path("~/.config/gpufabric/tls").expanduser()
        tls_desc = "Unknown"

    return {
        "hostname": hostname,
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "ips": ", ".join(ips) if ips else "None detected",
        "tls_dir": str(tls_dir),
        "tls_desc": tls_desc,
    }


class HostHeader(Widget):
    """Header banner with host identity, status badge, and GPU counts."""

    DEFAULT_CSS = """
    HostHeader {
        dock: top;
        height: 3;
        background: $surface;
        color: $text;
        border-bottom: solid $primary;
        padding: 0 1;
    }
    .header-title {
        text-style: bold;
        color: $accent-lighten-2;
        width: 32;
        padding: 0 1;
        align: left middle;
    }
    .header-badges {
        align: right middle;
        height: 100%;
        width: 1fr;
    }
    .badge {
        margin: 0 1;
        padding: 0 1;
        background: $panel;
        color: $text;
        border: solid $panel-lighten-1;
    }
    """

    host_name: reactive[str] = reactive("")
    host_status: reactive[str] = reactive("INITIALIZING")
    gpu_count: reactive[int] = reactive(0)

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Static("◆ GPU FABRIC HOST MONITOR", classes="header-title")
            with Horizontal(classes="header-badges"):
                yield Label(id="header-host-badge", classes="badge")
                yield Label(id="header-gpus-badge", classes="badge")
                yield Label(id="header-status-badge", classes="badge")

    def watch_host_name(self, val: str) -> None:
        self._update_badges()

    def watch_host_status(self, val: str) -> None:
        self._update_badges()

    def watch_gpu_count(self, val: int) -> None:
        self._update_badges()

    def _update_badges(self) -> None:
        try:
            h_lbl = self.query_one("#header-host-badge", Label)
            g_lbl = self.query_one("#header-gpus-badge", Label)
            s_lbl = self.query_one("#header-status-badge", Label)

            h_lbl.update(
                Text.assemble(("Host: ", "dim"), (self.host_name or "localhost", "bold cyan"))
            )
            g_lbl.update(Text.assemble(("GPUs: ", "dim"), (str(self.gpu_count), "bold magenta")))

            if self.host_status == "HEALTHY":
                status_text = Text.assemble(("Status: ", "dim"), ("HEALTHY", "bold green"))
            elif self.host_status == "DEGRADED":
                status_text = Text.assemble(("Status: ", "dim"), ("DEGRADED", "bold yellow"))
            else:
                status_text = Text.assemble(
                    ("Status: ", "dim"), (self.host_status or "UNKNOWN", "bold red")
                )
            s_lbl.update(status_text)
        except Exception:
            pass


class GPUFabricHostApp(App):
    """Interactive standalone TUI for local GPU Fabric host monitoring without network connections."""

    TITLE = "GPU Fabric Host Monitor"
    SUB_TITLE = "Local Hardware & Runtime Environment"

    BINDINGS = [
        Binding("r", "refresh", "Refresh", show=True),
        Binding("d", "cycle_device", "Next Device", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    CSS = """
    Screen {
        background: $background;
        color: $text;
    }
    .main-container {
        height: 1fr;
        padding: 0 1;
    }
    #host-info-panel {
        width: 48%;
        margin-right: 1;
        border: round $primary;
        background: $surface;
        padding: 0 1;
    }
    #gpu-telemetry-panel {
        width: 52%;
        border: round $accent;
        background: $surface;
        padding: 0 1;
    }
    .panel-title {
        text-style: bold;
        color: $accent-lighten-2;
    }
    .panel-subtitle {
        color: $text-muted;
        margin-bottom: 1;
    }
    .section-header {
        text-style: bold;
        color: $primary-lighten-2;
        margin-top: 1;
        margin-bottom: 0;
        border-bottom: solid $panel;
        padding-bottom: 0;
    }
    .info-row {
        height: 1;
        margin: 0;
        align: left middle;
    }
    .info-key {
        width: 18;
        color: $text-muted;
    }
    .info-val {
        width: 1fr;
        text-style: bold;
        color: $text;
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
    .alerts-container {
        margin-top: 1;
        padding: 1;
        background: $panel;
        border: solid $panel-lighten-1;
        height: auto;
    }
    .empty-gpu-state {
        height: 100%;
        align: center middle;
        text-align: center;
        color: $text-muted;
        padding: 2;
    }
    Footer {
        dock: bottom;
        background: $surface;
    }
    """

    def __init__(
        self,
        worker_id: Optional[str] = None,
        refresh_interval: float = 2.0,
        gpu_manager: Optional[GPUManager] = None,
        executor: Optional[GPUExecutor] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.worker_state = WorkerState(worker_id=worker_id)
        self.refresh_interval = refresh_interval
        self.gpu_mgr = gpu_manager or GPUManager()
        self.executor = executor or GPUExecutor()
        self.selected_device_index: int = 0
        self.host_info: Dict[str, Any] = get_local_host_info()
        self.diagnostics: CudaDiagnostics = probe_cuda_environment(self.gpu_mgr, self.executor)
        self._timer = None

    def compose(self) -> ComposeResult:
        yield HostHeader(id="host-header")
        with Horizontal(classes="main-container"):
            # Left Panel: Host & Runtime Environment
            with VerticalScroll(id="host-info-panel"):
                yield Static("HOST & RUNTIME ENVIRONMENT", classes="panel-title")
                yield Static(
                    "Local system identity, network, and CUDA diagnostics", classes="panel-subtitle"
                )

                yield Static("SYSTEM DETAILS", classes="section-header")
                with Horizontal(classes="info-row"):
                    yield Label("Hostname:", classes="info-key")
                    yield Label(self.host_info["hostname"], id="lbl-hostname", classes="info-val")
                with Horizontal(classes="info-row"):
                    yield Label("Worker ID:", classes="info-key")
                    yield Label(self.worker_state.worker_id, id="lbl-worker-id", classes="info-val")
                with Horizontal(classes="info-row"):
                    yield Label("Platform:", classes="info-key")
                    yield Label(self.host_info["platform"], id="lbl-platform", classes="info-val")
                with Horizontal(classes="info-row"):
                    yield Label("Python:", classes="info-key")
                    yield Label(self.host_info["python"], id="lbl-python", classes="info-val")
                with Horizontal(classes="info-row"):
                    yield Label("Network IPs:", classes="info-key")
                    yield Label(self.host_info["ips"], id="lbl-ips", classes="info-val")
                with Horizontal(classes="info-row"):
                    yield Label("TLS Config:", classes="info-key")
                    yield Label(self.host_info["tls_desc"], id="lbl-tls", classes="info-val")

                yield Static("CUDA & RUNTIME DIAGNOSTICS", classes="section-header")
                with Horizontal(classes="info-row"):
                    yield Label("Host Status:", classes="info-key")
                    yield Label("-", id="lbl-status", classes="info-val")
                with Horizontal(classes="info-row"):
                    yield Label("NVIDIA Driver:", classes="info-key")
                    yield Label("-", id="lbl-driver", classes="info-val")
                with Horizontal(classes="info-row"):
                    yield Label("CUDA Driver:", classes="info-key")
                    yield Label("-", id="lbl-cuda-driver", classes="info-val")
                with Horizontal(classes="info-row"):
                    yield Label("CUDA Runtime:", classes="info-key")
                    yield Label("-", id="lbl-cudart", classes="info-val")
                with Horizontal(classes="info-row"):
                    yield Label("CuPy Backend:", classes="info-key")
                    yield Label("-", id="lbl-cupy", classes="info-val")
                with Horizontal(classes="info-row"):
                    yield Label("cuBLAS Library:", classes="info-key")
                    yield Label("-", id="lbl-cublas", classes="info-val")

                yield Static("DIAGNOSTICS & ALERTS", classes="section-header")
                with Vertical(classes="alerts-container"):
                    yield Static("-", id="alerts-text")

            # Right Panel: Local GPU Telemetry & Hardware
            with VerticalScroll(id="gpu-telemetry-panel"):
                yield Static("LOCAL GPU TELEMETRY", classes="panel-title")
                yield Static(
                    "Live device load, memory, thermals, and power draw", classes="panel-subtitle"
                )

                with Horizontal(classes="gpu-selector-row"):
                    yield Label("Device:", classes="metric-name")
                    yield Select(
                        [("GPU 0", 0)],
                        id="host-device-select",
                        prompt="Select Device",
                        allow_blank=False,
                    )

                with Vertical(id="telemetry-content"):
                    with Horizontal(classes="metric-row"):
                        yield Label("VRAM:", classes="metric-name")
                        yield Label(
                            "0.0 / 0.0 GB (0%)", id="host-vram-text", classes="metric-value"
                        )
                        yield ProgressBar(
                            total=100,
                            show_percentage=False,
                            show_eta=False,
                            id="host-vram-bar",
                            classes="metric-bar",
                        )

                    with Horizontal(classes="metric-row"):
                        yield Label("GPU Core:", classes="metric-name")
                        yield Label("0%", id="host-util-text", classes="metric-value")
                        yield ProgressBar(
                            total=100,
                            show_percentage=False,
                            show_eta=False,
                            id="host-util-bar",
                            classes="metric-bar",
                        )

                    with Horizontal(classes="metric-row"):
                        yield Label("Memory Bus:", classes="metric-name")
                        yield Label("0%", id="host-mem-util-text", classes="metric-value")
                        yield ProgressBar(
                            total=100,
                            show_percentage=False,
                            show_eta=False,
                            id="host-mem-util-bar",
                            classes="metric-bar",
                        )

                    with Horizontal(classes="metric-row"):
                        yield Label("Power Draw:", classes="metric-name")
                        yield Label("0 W", id="host-power-text", classes="metric-value")
                        yield ProgressBar(
                            total=100,
                            show_percentage=False,
                            show_eta=False,
                            id="host-power-bar",
                            classes="metric-bar",
                        )

                    with Horizontal(classes="metric-row"):
                        yield Label("Temperature:", classes="metric-name")
                        yield Label("-", id="host-temp-text", classes="metric-value")

                    with Horizontal(classes="metric-row"):
                        yield Label("Device Name:", classes="metric-name")
                        yield Label("-", id="host-gpu-name", classes="metric-value")

                    with Horizontal(classes="metric-row"):
                        yield Label("Compute Cap:", classes="metric-name")
                        yield Label("-", id="host-cc-text", classes="metric-value")

                    with Horizontal(classes="metric-row"):
                        yield Label("Peak Bandwidth:", classes="metric-name")
                        yield Label("-", id="host-bw-text", classes="metric-value")

                    with Horizontal(classes="metric-row"):
                        yield Label("Availability:", classes="metric-name")
                        yield Label("✓ Available", id="host-avail-text", classes="metric-value")

                yield Static(
                    "No NVIDIA GPU detected on this host.\n\nNVML initialization failed or no CUDA-compatible hardware is installed.",
                    id="no-gpu-msg",
                    classes="empty-gpu-state",
                )

        yield Footer()

    def on_mount(self) -> None:
        self.update_all()
        if self.refresh_interval > 0:
            self._timer = self.set_interval(self.refresh_interval, self.update_telemetry_only)

    def action_refresh(self) -> None:
        """Manually trigger a full diagnostics and telemetry update."""
        self.update_all()
        self.notify("Host diagnostics and telemetry refreshed", timeout=1.5)

    def action_cycle_device(self) -> None:
        """Cycle to the next GPU device index if multiple GPUs exist."""
        count = self.gpu_mgr.device_count() if self.gpu_mgr.is_available() else 0
        if count > 1:
            self.selected_device_index = (self.selected_device_index + 1) % count
            try:
                select = self.query_one("#host-device-select", Select)
                select.value = self.selected_device_index
            except Exception:
                pass
            self.update_telemetry_only()
            self.notify(f"Switched to GPU {self.selected_device_index}", timeout=1.0)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "host-device-select" and event.value is not None:
            try:
                dev_idx = int(event.value)
                if dev_idx != self.selected_device_index:
                    self.selected_device_index = dev_idx
                    self.update_telemetry_only()
            except (ValueError, TypeError):
                pass

    def update_all(self) -> None:
        """Refresh system diagnostics, device options, and telemetry."""
        # 1. Probe diagnostics
        try:
            self.diagnostics = probe_cuda_environment(self.gpu_mgr, self.executor)
        except Exception:
            pass

        # 2. Update Header
        header = self.query_one("#host-header", HostHeader)
        header.host_name = self.host_info["hostname"]
        header.host_status = self.diagnostics.status
        count = self.gpu_mgr.device_count() if self.gpu_mgr.is_available() else 0
        header.gpu_count = count

        # 3. Update Left Panel Labels
        status_lbl = self.query_one("#lbl-status", Label)
        if self.diagnostics.status == "HEALTHY":
            status_lbl.update(Text("HEALTHY", style="bold green"))
        elif self.diagnostics.status == "DEGRADED":
            status_lbl.update(Text("DEGRADED", style="bold yellow"))
        else:
            status_lbl.update(Text("UNAVAILABLE", style="bold red"))

        self.query_one("#lbl-driver", Label).update(self.diagnostics.driver_version or "N/A")
        self.query_one("#lbl-cuda-driver", Label).update(
            self.diagnostics.cuda_driver_version or "N/A"
        )

        if self.diagnostics.cudart_available:
            self.query_one("#lbl-cudart", Label).update(
                Text(f"AVAILABLE ({self.diagnostics.cudart_version})", style="bold green")
            )
        else:
            self.query_one("#lbl-cudart", Label).update(
                Text("MISSING", style="bold red" if count > 0 else "dim")
            )

        if self.diagnostics.cupy_available:
            self.query_one("#lbl-cupy", Label).update(Text("AVAILABLE", style="bold green"))
        else:
            self.query_one("#lbl-cupy", Label).update(
                Text("MISSING", style="bold red" if count > 0 else "dim")
            )

        if self.diagnostics.cublas_available:
            self.query_one("#lbl-cublas", Label).update(Text("AVAILABLE", style="bold green"))
        else:
            missing_tag = (
                f" ({', '.join(self.diagnostics.missing_libraries)})"
                if self.diagnostics.missing_libraries
                else ""
            )
            self.query_one("#lbl-cublas", Label).update(
                Text(f"MISSING{missing_tag}", style="bold red" if count > 0 else "dim")
            )

        # Alerts and Hints
        alert_lines = []
        if self.diagnostics.warnings:
            for w in self.diagnostics.warnings:
                alert_lines.append(f"[yellow]⚠ {w}[/yellow]")
        if self.diagnostics.remediation_hints:
            for h in self.diagnostics.remediation_hints:
                alert_lines.append(f"[cyan]💡 {h}[/cyan]")
        if not alert_lines:
            alert_lines.append(
                "[green]✓ All hardware, runtime, and compute dependencies operational.[/green]"
            )
        self.query_one("#alerts-text", Static).update("\n".join(alert_lines))

        # 4. Device Selector Setup
        select = self.query_one("#host-device-select", Select)
        if count > 0:
            options = []
            for i in range(count):
                try:
                    info = self.gpu_mgr.get_info(i)
                    name = info.get("name", f"GPU {i}")
                except Exception:
                    name = f"GPU {i}"
                options.append((f"GPU {i}: {name}", i))
            select.set_options(options)
            if self.selected_device_index >= count:
                self.selected_device_index = 0
            select.value = self.selected_device_index
            self.query_one(".gpu-selector-row").display = count > 1
            self.query_one("#telemetry-content").display = True
            self.query_one("#no-gpu-msg").display = False
        else:
            self.query_one(".gpu-selector-row").display = False
            self.query_one("#telemetry-content").display = False
            self.query_one("#no-gpu-msg").display = True

        self.update_telemetry_only()

    def update_telemetry_only(self) -> None:
        """Quick refresh of hardware gauges for the selected device."""
        count = self.gpu_mgr.device_count() if self.gpu_mgr.is_available() else 0
        if count == 0:
            return

        dev_idx = min(self.selected_device_index, count - 1)
        try:
            info = self.gpu_mgr.get_info(dev_idx)
            status = self.gpu_mgr.get_status(dev_idx)
            bw = self.gpu_mgr.peak_memory_bandwidth(dev_idx)
        except Exception:
            return

        name = info.get("name", f"GPU {dev_idx}")
        cc = info.get("compute_capability", "N/A")
        total_vram = status.get("total_vram", 0)
        free_vram = status.get("free_vram", 0)
        used_vram = status.get("used_vram", total_vram - free_vram)
        gpu_util = status.get("gpu_util", 0)
        mem_util = status.get("mem_util", 0)
        temp_c = status.get("temp", 0)
        power_w = status.get("power_usage_w", 0)
        power_limit_w = status.get("power_limit_w", 0)

        # Update static labels
        self.query_one("#host-gpu-name", Label).update(name)
        self.query_one("#host-cc-text", Label).update(str(cc))
        self.query_one("#host-bw-text", Label).update(f"{bw:.2f} GB/s" if bw > 0 else "N/A")

        # VRAM gauge
        vram_pct = int((used_vram / total_vram * 100)) if total_vram > 0 else 0
        vram_text = f"{bytes_to_human(used_vram)} / {bytes_to_human(total_vram)} ({vram_pct}%)"
        self.query_one("#host-vram-text", Label).update(vram_text)
        self.query_one("#host-vram-bar", ProgressBar).progress = vram_pct

        # GPU Core Load
        self.query_one("#host-util-text", Label).update(f"{gpu_util}%")
        self.query_one("#host-util-bar", ProgressBar).progress = int(gpu_util)

        # Memory Bus Load
        self.query_one("#host-mem-util-text", Label).update(f"{mem_util}%")
        self.query_one("#host-mem-util-bar", ProgressBar).progress = int(mem_util)

        # Power Draw
        if power_limit_w > 0:
            power_pct = int(min(100, (power_w / power_limit_w) * 100))
            power_text = f"{power_w} W / {power_limit_w} W ({power_pct}%)"
        elif power_w > 0:
            power_pct = 0
            power_text = f"{power_w} W"
        else:
            power_pct = 0
            power_text = "N/A"
        self.query_one("#host-power-text", Label).update(power_text)
        self.query_one("#host-power-bar", ProgressBar).progress = power_pct

        # Temperature
        if temp_c < 60:
            t_text = Text(f"{temp_c} °C (Cool)", style="bold green")
        elif temp_c < 80:
            t_text = Text(f"{temp_c} °C (Warm)", style="bold yellow")
        else:
            t_text = Text(f"{temp_c} °C (Hot)", style="bold red")
        self.query_one("#host-temp-text", Label).update(t_text)


def main():
    """CLI launcher for standalone host TUI."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="gpufabric-host",
        description="Local system and GPU hardware monitoring TUI for GPU Fabric host.",
    )
    parser.add_argument("--worker-id", type=str, default=None, help="Custom worker identifier")
    parser.add_argument(
        "--refresh",
        type=float,
        default=2.0,
        help="Telemetry refresh interval in seconds (default: 2.0)",
    )
    args = parser.parse_args()

    app = GPUFabricHostApp(worker_id=args.worker_id, refresh_interval=args.refresh)
    app.run()


if __name__ == "__main__":
    main()
