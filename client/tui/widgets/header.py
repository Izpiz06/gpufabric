"""Header and summary metrics widget for GPU Fabric TUI."""

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, Static


class GPUFabricHeader(Widget):
    """Header banner with system metrics summary."""

    DEFAULT_CSS = """
    GPUFabricHeader {
        dock: top;
        height: 3;
        background: $surface;
        color: $text;
        border-bottom: heavy $accent;
        padding: 0 1;
    }
    .header-title {
        text-style: bold;
        color: $accent;
        width: 22;
        padding: 0 1;
    }
    .metrics-bar {
        align: right middle;
        height: 100%;
        width: 1fr;
    }
    .metric-badge {
        margin: 0 1;
        padding: 0 1;
        background: $panel;
        color: $text;
    }
    .status-badge {
        margin-left: 1;
        padding: 0 1;
        color: $text-muted;
    }
    """

    total_workers: reactive[int] = reactive(0)
    total_gpus: reactive[int] = reactive(0)
    online_workers: reactive[int] = reactive(0)
    status_message: reactive[str] = reactive("Ready")

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Static("⚡ GPU FABRIC", classes="header-title")
            with Horizontal(classes="metrics-bar"):
                yield Label(id="workers-metric", classes="metric-badge")
                yield Label(id="gpus-metric", classes="metric-badge")
                yield Label(id="online-metric", classes="metric-badge")
                yield Label(id="status-metric", classes="status-badge")

    def watch_total_workers(self, val: int) -> None:
        self._update_metrics()

    def watch_total_gpus(self, val: int) -> None:
        self._update_metrics()

    def watch_online_workers(self, val: int) -> None:
        self._update_metrics()

    def watch_status_message(self, val: str) -> None:
        try:
            self.query_one("#status-metric", Label).update(f"Status: {val}")
        except Exception:
            pass

    def _update_metrics(self) -> None:
        try:
            w_lbl = self.query_one("#workers-metric", Label)
            g_lbl = self.query_one("#gpus-metric", Label)
            o_lbl = self.query_one("#online-metric", Label)

            w_lbl.update(
                Text.assemble(("Workers: ", "dim"), (str(self.total_workers), "bold cyan"))
            )
            g_lbl.update(Text.assemble(("GPUs: ", "dim"), (str(self.total_gpus), "bold magenta")))
            o_lbl.update(
                Text.assemble(("Online: ", "dim"), (str(self.online_workers), "bold green"))
            )
        except Exception:
            pass
