"""Benchmark modal screen for GPU Fabric TUI."""

import math
from typing import Any, Dict, List, Optional

from rich.table import Table
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ProgressBar, Select, Static

from client.bench import PI_SAMPLE_STD, PI_SIGMAS
from client.client import GPUFabricClient


class BenchmarkModal(ModalScreen):
    """Modal screen for configuring and executing remote GPU benchmarks."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
    ]

    DEFAULT_CSS = """
    BenchmarkModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }
    .bench-dialog {
        width: 88%;
        height: 85%;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }
    .bench-title {
        text-style: bold;
        color: $accent;
        padding-bottom: 1;
        border-bottom: solid $accent;
    }
    .config-grid {
        height: auto;
        margin: 1 0;
        padding: 1;
        background: $panel;
        border: solid $panel-lighten-1;
    }
    .form-row {
        height: 3;
        align: left middle;
        margin-bottom: 1;
    }
    .form-label {
        width: 16;
        color: $text-muted;
    }
    .form-input {
        width: 1fr;
    }
    .results-scroll {
        height: 1fr;
        margin: 1 0;
    }
    .btn-row {
        height: 3;
        align: right middle;
    }
    .action-btn {
        margin-left: 1;
    }
    """

    BENCHMARK_TYPES = [
        ("STREAM Triad (Memory Bandwidth)", "triad"),
        ("Matrix Multiplication (GFLOPS)", "matmul"),
        ("Monte Carlo Pi (Accuracy & Throughput)", "monte_carlo_pi"),
    ]

    DEFAULT_SIZES = {
        "triad": "10000000",
        "matmul": "2000",
        "monte_carlo_pi": "50000000",
    }

    def __init__(
        self,
        online_workers: List[str],
        default_target: Optional[str] = None,
        default_device: int = 0,
        client_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.online_workers = online_workers
        self.default_target = default_target or (online_workers[0] if online_workers else "")
        self.default_device = default_device
        self.client_kwargs = dict(client_kwargs or {})
        self._bench_in_progress = False

    def compose(self) -> ComposeResult:
        with Container(classes="bench-dialog"):
            yield Static("GPU Benchmark Execution", classes="bench-title")

            with Vertical(classes="config-grid"):
                with Horizontal(classes="form-row"):
                    yield Label("Worker:", classes="form-label")
                    worker_options = [(w, w) for w in self.online_workers] or [
                        ("No workers available", "")
                    ]
                    yield Select(
                        worker_options,
                        value=self.default_target
                        if self.default_target in self.online_workers
                        else (self.online_workers[0] if self.online_workers else ""),
                        id="bench-worker-select",
                        classes="form-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Benchmark:", classes="form-label")
                    yield Select(
                        self.BENCHMARK_TYPES,
                        value="triad",
                        id="bench-type-select",
                        classes="form-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Device Index:", classes="form-label")
                    yield Input(
                        value=str(self.default_device),
                        id="bench-device-input",
                        placeholder="0",
                        classes="form-input",
                    )

                    yield Label("Repeats:", classes="form-label")
                    yield Input(
                        value="5",
                        id="bench-repeats-input",
                        placeholder="5",
                        classes="form-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Elements / Size:", classes="form-label")
                    yield Input(
                        value=self.DEFAULT_SIZES["triad"],
                        id="bench-size-input",
                        placeholder="10000000",
                        classes="form-input",
                    )

            with VerticalScroll(classes="results-scroll"):
                yield ProgressBar(id="bench-progress", show_percentage=False, show_eta=False)
                yield Static(id="bench-results-text")

            with Horizontal(classes="btn-row"):
                yield Button(
                    "Run Benchmark", variant="success", id="run-bench-btn", classes="action-btn"
                )
                yield Button(
                    "Close (Esc)", variant="default", id="close-bench-btn", classes="action-btn"
                )

    def on_mount(self) -> None:
        try:
            self.query_one("#bench-progress").display = False
            self.query_one("#bench-results-text", Static).update(
                "[dim]Configure benchmark parameters above and click 'Run Benchmark'.[/dim]"
            )
        except Exception:
            pass

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "bench-type-select" and event.value in self.DEFAULT_SIZES:
            try:
                self.query_one("#bench-size-input", Input).value = self.DEFAULT_SIZES[
                    str(event.value)
                ]
            except Exception:
                pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-bench-btn":
            self.dismiss()
        elif event.button.id == "run-bench-btn":
            self.start_benchmark()

    def start_benchmark(self) -> None:
        if self._bench_in_progress:
            return

        worker = str(self.query_one("#bench-worker-select", Select).value or "")
        if not worker:
            self.query_one("#bench-results-text", Static).update(
                "[bold red]No worker selected![/bold red]"
            )
            return

        bench_type = str(self.query_one("#bench-type-select", Select).value or "triad")
        try:
            device_idx = int(self.query_one("#bench-device-input", Input).value or "0")
            repeats = int(self.query_one("#bench-repeats-input", Input).value or "5")
            size = int(self.query_one("#bench-size-input", Input).value or "1000")
        except ValueError:
            self.query_one("#bench-results-text", Static).update(
                "[bold red]Device index, repeats, and size must be valid integers![/bold red]"
            )
            return

        self._bench_in_progress = True
        self.query_one("#run-bench-btn", Button).disabled = True
        self.query_one("#bench-progress").display = True
        self.query_one("#bench-results-text", Static).update(
            f"[bold yellow]Running {bench_type} on {worker} (Device {device_idx})...[/bold yellow]"
        )

        self._run_benchmark_worker(worker, bench_type, size, repeats, device_idx)

    @work(thread=True)
    def _run_benchmark_worker(
        self, worker: str, bench_type: str, size: int, repeats: int, device_idx: int
    ) -> None:
        try:
            # Parse host/port
            if ":" in worker:
                h, p = worker.split(":", 1)
                port = int(p)
            else:
                h, port = worker, 50051

            kwargs = dict(self.client_kwargs)
            kwargs.pop("port", None)

            with GPUFabricClient(host=h, port=port, **kwargs) as client:
                resp = client.run_benchmark(
                    bench_type, size=size, repeats=repeats, device_index=device_idx
                )
                self.app.call_from_thread(
                    self._handle_benchmark_success,
                    worker,
                    bench_type,
                    resp,
                    size,
                    repeats,
                    device_idx,
                )
        except Exception as e:
            self.app.call_from_thread(self._handle_benchmark_error, str(e))

    def _handle_benchmark_success(
        self, worker: str, bench_type: str, resp: Any, size: int, repeats: int, device_idx: int
    ) -> None:
        self._bench_in_progress = False
        try:
            self.query_one("#run-bench-btn", Button).disabled = False
            self.query_one("#bench-progress").display = False

            t = Table(title=f"Benchmark Results: {bench_type.upper()}", expand=True)
            t.add_column("Metric", style="cyan", width=26)
            t.add_column("Result", style="bold green", width=30)
            t.add_column("Status / Note", style="bold")

            t.add_row("Worker Target", worker, "-")
            t.add_row("GPU Device Index", str(device_idx), "-")
            t.add_row("Size / Elements", f"{size:,}", "-")
            t.add_row("Timed Repeats", str(repeats), "-")
            t.add_row("Fastest Run (best)", f"{resp.best_ms:.4f} ms", "-")
            t.add_row("Mean Run Time", f"{resp.mean_ms:.4f} ms", "-")

            if bench_type == "triad":
                bw = resp.bandwidth_gb_s
                peak = resp.peak_bandwidth_gb_s
                if peak > 0:
                    pct = 100.0 * bw / peak
                    t.add_row(
                        "GPU Memory Bandwidth",
                        f"{bw:.1f} GB/s",
                        f"[green]PASS ({pct:.0f}% of {peak:.1f} GB/s peak)[/green]"
                        if pct >= 50
                        else f"[yellow]{pct:.0f}% of {peak:.1f} GB/s peak[/yellow]",
                    )
                else:
                    t.add_row("GPU Memory Bandwidth", f"{bw:.1f} GB/s", "[dim]peak unknown[/dim]")
            elif bench_type == "matmul":
                t.add_row(
                    "Compute Throughput", f"{resp.gflops:,.1f} GFLOPS", "[green]PASS (INFO)[/green]"
                )
            elif bench_type == "monte_carlo_pi":
                total_samples = size * repeats
                limit = PI_SIGMAS * PI_SAMPLE_STD / math.sqrt(total_samples)
                error = abs(resp.pi_estimate - math.pi)
                status_str = (
                    "[green]PASS (<= 5 sigma)[/green]"
                    if error <= limit
                    else "[red]FAIL (> 5 sigma)[/red]"
                )
                t.add_row("Pi Estimate", f"{resp.pi_estimate:.7f}", f"error: {error:.2e}")
                t.add_row("Error Tolerance", f"{limit:.2e}", status_str)

            self.query_one("#bench-results-text", Static).update(t)
        except Exception:
            pass

    def _handle_benchmark_error(self, error_msg: str) -> None:
        self._bench_in_progress = False
        try:
            self.query_one("#run-bench-btn", Button).disabled = False
            self.query_one("#bench-progress").display = False
            self.query_one("#bench-results-text", Static).update(
                f"[bold red]Benchmark Failed:[/bold red]\n\n{error_msg}"
            )
        except Exception:
            pass
