"""Main Textual Application for GPU Fabric TUI."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Footer

from client.client import GPUFabricClient
from client.commands import resolve_workers
from client.tui.screens.benchmark import BenchmarkModal
from client.tui.screens.details import DetailsModal
from client.tui.widgets.gpu_card import GPUCard
from client.tui.widgets.header import GPUFabricHeader
from client.tui.widgets.worker_table import WorkerEntry, WorkerTable
from common.constants import (
    DEFAULT_DISCOVERY_TIMEOUT,
    DEFAULT_MAX_MESSAGE_MB,
    DEFAULT_PORT,
)


class GPUFabricApp(App):
    """Interactive Terminal User Interface for GPU Fabric."""

    TITLE = "GPU Fabric"
    SUB_TITLE = "Distributed GPU Monitoring & Control"

    BINDINGS = [
        Binding("d", "discover", "Discover LAN", show=True),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("enter", "details", "Details", show=True),
        Binding("b", "benchmark", "Benchmark", show=True),
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
    WorkerTable {
        width: 48%;
        margin-right: 1;
    }
    GPUCard {
        width: 52%;
    }
    Footer {
        dock: bottom;
        background: $surface;
    }
    """

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        timeout: float = 5.0,
        max_message_mb: int = DEFAULT_MAX_MESSAGE_MB,
        tls_dir: Optional[Union[str, Path]] = None,
        insecure: bool = False,
        seed_workers: Optional[List[str]] = None,
        auto_discover_on_mount: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.port = port
        self.timeout = timeout
        self.max_message_mb = max_message_mb
        self.tls_dir = tls_dir
        self.insecure = insecure
        self.auto_discover_on_mount = auto_discover_on_mount

        self.client_kwargs = {
            "port": port,
            "timeout": timeout,
            "max_message_mb": max_message_mb,
            "tls_dir": tls_dir,
            "insecure": insecure,
        }

        # Worker state tracking
        self.known_workers: Dict[str, WorkerEntry] = {}
        self.worker_gpu_responses: Dict[str, Any] = {}
        self.selected_target: Optional[str] = None
        self.selected_device: int = 0

        # Seed from arguments or env
        initial_workers = resolve_workers(seed_workers or [])

        for w in initial_workers:
            target = w if ":" in w else f"{w}:{port}"
            self.known_workers[target] = WorkerEntry(
                target=target,
                name=target.split(":")[0],
                worker_id="",
                status="CONNECTING",
                gpu_count=0,
                compute_ready=False,
            )

    def compose(self) -> ComposeResult:
        yield GPUFabricHeader(id="app-header")
        with Horizontal(classes="main-container"):
            yield WorkerTable(id="worker-table")
            yield GPUCard(id="gpu-card")
        yield Footer()

    def on_mount(self) -> None:
        # Populate initial workers if any
        if self.known_workers:
            self.query_one(WorkerTable).update_workers(list(self.known_workers.values()))
            self._update_header_metrics()

        # Telemetry interval
        self.set_interval(1.5, self._on_telemetry_timer)

        # Trigger startup discovery or refresh
        if self.auto_discover_on_mount:
            self.action_discover()
        elif self.known_workers:
            self.action_refresh()

    def _get_client_for_target(self, target: str) -> GPUFabricClient:
        if ":" in target:
            h, p = target.split(":", 1)
            port = int(p)
        else:
            h, port = target, self.port

        kwargs = dict(self.client_kwargs)
        kwargs.pop("port", None)
        return GPUFabricClient(host=h, port=port, **kwargs)

    # -------------------------------------------------------------------------
    # Actions & Background Workers
    # -------------------------------------------------------------------------

    def action_discover(self) -> None:
        """Scan LAN for GPU Fabric workers."""
        header = self.query_one(GPUFabricHeader)
        header.status_message = "Scanning LAN for workers..."
        self._discover_worker_task()

    @work(exclusive=True, thread=True)
    def _discover_worker_task(self) -> None:
        try:
            discovered = GPUFabricClient.discover_lan(
                timeout=DEFAULT_DISCOVERY_TIMEOUT,
                client_kwargs=self.client_kwargs,
            )
            self.app.call_from_thread(self._handle_discovered_workers, discovered)
        except Exception as e:
            self.app.call_from_thread(self._handle_discover_error, str(e))

    def _handle_discovered_workers(self, discovered: List[Any]) -> None:
        header = self.query_one(GPUFabricHeader)
        for cand in discovered:
            target = f"{cand.address}:{cand.port}"
            existing = self.known_workers.get(target)
            self.known_workers[target] = WorkerEntry(
                target=target,
                name=cand.name or (existing.name if existing else cand.address),
                worker_id=cand.worker_id or (existing.worker_id if existing else ""),
                status=cand.status,
                gpu_count=existing.gpu_count if existing else 0,
                compute_ready=existing.compute_ready if existing else False,
                version=cand.version,
                error=cand.error or "",
            )

        header.status_message = f"Discovery complete ({len(discovered)} found)"
        self.query_one(WorkerTable).update_workers(list(self.known_workers.values()))
        self._update_header_metrics()

        # Query GPU details for all discovered workers
        self._refresh_workers_task()

    def _handle_discover_error(self, err: str) -> None:
        header = self.query_one(GPUFabricHeader)
        header.status_message = f"Discovery error: {err}"
        self.notify(f"Discovery error: {err}", severity="error")

    def action_refresh(self) -> None:
        """Refresh inventory and health for all known workers."""
        header = self.query_one(GPUFabricHeader)
        header.status_message = "Refreshing workers..."
        self._refresh_workers_task()

    @work(exclusive=True, thread=True)
    def _refresh_workers_task(self) -> None:
        targets = list(self.known_workers.keys())
        results = {}

        for target in targets:
            try:
                with self._get_client_for_target(target) as client:
                    resp = client.list_gpus()
                    results[target] = (resp, None)
            except Exception as e:
                results[target] = (None, str(e))

        self.app.call_from_thread(self._handle_refresh_results, results)

    def _handle_refresh_results(self, results: Dict[str, tuple]) -> None:
        try:
            header = self.query_one(GPUFabricHeader)
            for target, (resp, err) in results.items():
                entry = self.known_workers.get(target)
                if not entry:
                    continue

                if resp is not None:
                    self.worker_gpu_responses[target] = resp
                    entry.status = "ONLINE"
                    entry.worker_id = resp.worker_id
                    entry.version = resp.version
                    entry.compute_ready = resp.compute_ready
                    entry.gpu_count = len(resp.gpus)
                    entry.health_state = getattr(resp, "health_state", "HEALTHY")
                    entry.cuda_version = getattr(resp, "cuda_version", "")
                    entry.hostname = getattr(resp, "hostname", "")
                    entry.error = ""
                else:
                    entry.status = "UNREACHABLE"
                    entry.health_state = "UNAVAILABLE"
                    entry.error = err or "Connection failed"

            header.status_message = "Ready"
            self.query_one(WorkerTable).update_workers(list(self.known_workers.values()))
            self._update_header_metrics()

            # Update selected GPU card
            if self.selected_target in results:
                resp, err = results[self.selected_target]
                self.query_one(GPUCard).set_worker(self.selected_target, resp, error=err or "")
        except Exception:
            pass

    def _on_telemetry_timer(self) -> None:
        if not self.selected_target:
            return
        entry = self.known_workers.get(self.selected_target)
        if not entry or entry.status != "ONLINE":
            return
        self._poll_selected_gpu_task(self.selected_target, self.selected_device)

    @work(thread=True)
    def _poll_selected_gpu_task(self, target: str, device_idx: int) -> None:
        try:
            with self._get_client_for_target(target) as client:
                status = client.get_status(device_index=device_idx)
                self.app.call_from_thread(self._update_gpu_card_status, target, status)
        except Exception:
            pass

    def _update_gpu_card_status(self, target: str, status: Any) -> None:
        if target == self.selected_target:
            try:
                self.query_one(GPUCard).update_telemetry(status)
            except Exception:
                pass

    def _update_header_metrics(self) -> None:
        try:
            header = self.query_one(GPUFabricHeader)
            workers = list(self.known_workers.values())
            header.total_workers = len(workers)
            header.online_workers = sum(1 for w in workers if w.status == "ONLINE")
            header.total_gpus = sum(w.gpu_count for w in workers if w.status == "ONLINE")
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # UI Event Handlers
    # -------------------------------------------------------------------------

    def on_worker_table_worker_selected(self, event: WorkerTable.WorkerSelected) -> None:
        self.selected_target = event.target
        try:
            gpu_card = self.query_one(GPUCard)
        except Exception:
            return

        if not event.target or not event.entry:
            gpu_card.set_worker("", None)
            return

        resp = self.worker_gpu_responses.get(event.target)
        gpu_card.set_worker(event.target, resp, error=event.entry.error)

    def on_worker_table_worker_activated(self, event: WorkerTable.WorkerActivated) -> None:
        self.action_details()

    def on_gpu_card_device_changed(self, event: GPUCard.DeviceChanged) -> None:
        self.selected_device = event.device_index
        if self.selected_target:
            self._poll_selected_gpu_task(self.selected_target, self.selected_device)

    def action_details(self) -> None:
        """Open detailed specs modal for selected worker."""
        if not self.selected_target:
            self.notify("Select a worker first.", severity="warning")
            return

        entry = self.known_workers.get(self.selected_target)
        resp = self.worker_gpu_responses.get(self.selected_target)
        self.push_screen(
            DetailsModal(
                target=self.selected_target,
                gpus_response=resp,
                error=entry.error if entry else "",
            )
        )

    def action_benchmark(self) -> None:
        """Open benchmark execution modal."""
        online = [t for t, w in self.known_workers.items() if w.status == "ONLINE"]
        if not online:
            self.notify("No online workers available for benchmarking.", severity="warning")
            return

        self.push_screen(
            BenchmarkModal(
                online_workers=online,
                default_target=self.selected_target
                if self.selected_target in online
                else online[0],
                default_device=self.selected_device,
                client_kwargs=self.client_kwargs,
            )
        )
