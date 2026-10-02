"""Shared gRPC channel and server options."""

from typing import List, Tuple

from common.constants import MAX_MESSAGE_MB_LIMIT


def message_size_options(max_message_mb: int) -> List[Tuple[str, int]]:
    """Build gRPC options that set the max send and receive message size."""
    if not 1 <= max_message_mb <= MAX_MESSAGE_MB_LIMIT:
        raise ValueError(
            f"max_message_mb must be between 1 and {MAX_MESSAGE_MB_LIMIT}, got {max_message_mb}"
        )
    max_bytes = max_message_mb * 1024 * 1024
    return [
        ("grpc.max_send_message_length", max_bytes),
        ("grpc.max_receive_message_length", max_bytes),
    ]
