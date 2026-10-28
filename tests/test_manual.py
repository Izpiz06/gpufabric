# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the GPU Fabric CLI manual and interactive setup assistant."""

from unittest.mock import MagicMock, patch

from rich.console import Console

from client.cli import build_parser, main
from client.commands import cmd_manual
from client.manual import (
    SECTIONS,
    get_diagnostic_info,
    print_diagnostic_panel,
    run_interactive_setup,
    show_manual,
)


def test_get_diagnostic_info():
    """Verify system, GPU, and TLS diagnostic probe returns required structure."""
    info = get_diagnostic_info()
    assert "python" in info
    assert "platform" in info
    assert "nvidia_available" in info
    assert "gpu_count" in info
    assert "cupy_available" in info
    assert "tls_initialized" in info
    assert "tls_dir" in info


def test_get_diagnostic_info_fallbacks():
    """Verify diagnostic fallbacks when NVML, CuPy, or TLS directories fail."""
    with patch("pynvml.nvmlInit", side_effect=Exception("No driver")):
        with patch("client.manual.default_tls_dir", side_effect=Exception("Disk error")):
            info = get_diagnostic_info()
            assert info["nvidia_available"] is False
            assert info["tls_initialized"] is False


def test_print_diagnostic_panel():
    """Verify diagnostic table prints without errors."""
    c = Console(record=True)
    diag = {
        "python": "3.11.0",
        "platform": "Linux test-box",
        "nvidia_available": True,
        "gpu_count": 1,
        "gpu_names": ["RTX 4060"],
        "driver_version": "535.104.05",
        "cupy_available": True,
        "cupy_version": "13.0.0",
        "tls_dir": "/tmp/tls",
        "tls_initialized": True,
        "tls_files": ["ca.crt", "server.crt", "server.key"],
        "has_ca": True,
        "has_server": True,
        "has_client": False,
    }
    print_diagnostic_panel(diag, console_out=c)
    out = c.export_text()
    assert "RTX 4060" in out
    assert "CuPy" in out
    assert "mTLS" in out


def test_show_manual_sections():
    """Verify all individual manual sections render without errors."""
    c = Console(record=True)
    for section in SECTIONS:
        show_manual(section=section, console_out=c)
    out = c.export_text()
    assert len(out) > 0


def test_show_manual_default_and_all():
    """Verify default menu, 'all' chapter view, and invalid section handling."""
    c = Console(record=True)
    # Default menu
    show_manual(section=None, console_out=c)
    out = c.export_text()
    assert "Manual Chapters" in out

    # All chapters
    c2 = Console(record=True)
    show_manual(section="all", console_out=c2)
    out2 = c2.export_text()
    assert "First-Time Setup Guide" in out2

    # Invalid section
    c3 = Console(record=True)
    show_manual(section="nonexistent_sec", console_out=c3)
    out3 = c3.export_text()
    assert "Unknown manual section" in out3


def test_run_interactive_setup_worker_role():
    """Test interactive setup assistant when user selects Worker role."""
    c = Console(record=True)
    with patch("rich.prompt.Prompt.ask", return_value="1"):
        with patch("rich.prompt.Confirm.ask", return_value=False):
            run_interactive_setup(console_out=c)
    out = c.export_text()
    assert "Configuring as GPU Worker" in out
    assert "gpufabric-worker" in out


def test_run_interactive_setup_client_role():
    """Test interactive setup assistant when user selects Client role."""
    c = Console(record=True)
    with patch("rich.prompt.Prompt.ask", return_value="2"):
        run_interactive_setup(console_out=c)
    out = c.export_text()
    assert "Configuring as GPU Fabric Client" in out
    assert "gpufabric discover" in out


def test_run_interactive_setup_standalone_role():
    """Test interactive setup assistant when user selects Standalone role."""
    c = Console(record=True)
    with patch("rich.prompt.Prompt.ask", return_value="3"):
        with patch("rich.prompt.Confirm.ask", return_value=False):
            run_interactive_setup(console_out=c)
    out = c.export_text()
    assert "Configuring Standalone / Local Test Mode" in out
    assert "gpufabric-worker" in out


def test_cmd_manual_dispatcher():
    """Test cmd_manual forwarding args to show_manual."""
    args = MagicMock()
    args.section = "setup"
    args.setup = False
    args.interactive = False

    with patch("client.manual.show_manual") as mock_show:
        cmd_manual(args)
        mock_show.assert_called_once_with(section="setup", interactive=False)


def test_cli_parser_manual_and_setup_commands():
    """Verify build_parser supports manual, guide, and setup subcommands."""
    parser = build_parser()

    # manual
    args = parser.parse_args(["manual"])
    assert args.command == "manual"
    assert args.section is None
    assert args.setup is False

    # manual setup
    args = parser.parse_args(["manual", "setup"])
    assert args.command == "manual"
    assert args.section == "setup"

    # manual --setup
    args = parser.parse_args(["manual", "--setup"])
    assert args.command == "manual"
    assert args.setup is True

    # manual -i
    args = parser.parse_args(["manual", "-i"])
    assert args.command == "manual"
    assert args.interactive is True

    # guide alias
    args = parser.parse_args(["guide", "commands"])
    assert args.command == "guide"
    assert args.section == "commands"

    # setup command
    args = parser.parse_args(["setup"])
    assert args.command == "setup"


def test_cli_main_dispatches_manual():
    """Verify main() properly executes cmd_manual when invoked."""
    with patch("sys.argv", ["gpufabric", "manual", "commands"]):
        with patch("client.cli.cmd_manual") as mock_cmd:
            main()
            mock_cmd.assert_called_once()

    with patch("sys.argv", ["gpufabric", "setup"]):
        with patch("client.cli.cmd_manual") as mock_cmd:
            main()
            mock_cmd.assert_called_once()
