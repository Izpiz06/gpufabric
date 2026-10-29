# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Add worker modal screen for GPU Fabric TUI."""

from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label


class AddWorkerModal(ModalScreen[Optional[str]]):
    """Modal screen for manually connecting to a worker node by IP or hostname."""

    BINDINGS = [
        Binding("escape", "dismiss_modal", "Cancel", show=True),
        Binding("enter", "submit_worker", "Connect", show=True),
    ]

    DEFAULT_CSS = """
    AddWorkerModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }
    .add-worker-dialog {
        width: 64;
        height: auto;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }
    .modal-title {
        text-style: bold;
        color: $accent;
        padding-bottom: 1;
        border-bottom: solid $primary;
    }
    .form-desc {
        margin: 1 0;
        color: $text-muted;
    }
    .modal-footer {
        height: 3;
        margin-top: 1;
        align: right middle;
    }
    .btn {
        margin-left: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Container(classes="add-worker-dialog"):
            yield Label("Add Worker Node", classes="modal-title")
            yield Label(
                "Enter worker IP or hostname (e.g. 192.168.1.50 or 192.168.1.50:50051):",
                classes="form-desc",
            )
            yield Input(placeholder="10.9.102.184:50051", id="worker-address-input")
            with Horizontal(classes="modal-footer"):
                yield Button("Cancel", id="btn-cancel", classes="btn")
                yield Button("Connect", variant="primary", id="btn-connect", classes="btn")

    def on_mount(self) -> None:
        self.query_one("#worker-address-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-connect":
            self.action_submit_worker()
        elif event.button.id == "btn-cancel":
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.action_submit_worker()

    def action_dismiss_modal(self) -> None:
        self.dismiss(None)

    def action_submit_worker(self) -> None:
        val = self.query_one("#worker-address-input", Input).value.strip()
        if val:
            self.dismiss(val)
        else:
            self.dismiss(None)
