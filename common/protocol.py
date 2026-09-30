"""Common protocol constants and helper utilities."""

DEFAULT_PORT = 8000
API_VERSION = "0.1.0"

# API Endpoints
HEALTH_ENDPOINT = "/health"
GPU_INFO_ENDPOINT = "/gpu"
STATUS_ENDPOINT = "/status"
EXECUTE_ENDPOINT = "/execute"


def bytes_to_human(n_bytes: int) -> str:
    """Format bytes into human-readable string (KiB, MiB, GiB, TiB)."""
    if n_bytes < 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    value = float(n_bytes)
    unit_idx = 0
    while value >= 1024.0 and unit_idx < len(units) - 1:
        value /= 1024.0
        unit_idx += 1
    return f"{value:.2f} {units[unit_idx]}"
