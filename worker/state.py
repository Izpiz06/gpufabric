"""Runtime state tracker for the worker node."""

import socket
import threading
from typing import Optional


class WorkerState:
    """Thread-safe state holder for worker ID and active tasks."""

    def __init__(self, worker_id: Optional[str] = None):
        self.worker_id = worker_id or f"worker-{socket.gethostname()}"
        self._active_tasks = 0
        self._lock = threading.Lock()

    @property
    def active_tasks(self) -> int:
        with self._lock:
            return self._active_tasks

    def increment_tasks(self):
        with self._lock:
            self._active_tasks += 1

    def decrement_tasks(self):
        with self._lock:
            self._active_tasks = max(0, self._active_tasks - 1)
