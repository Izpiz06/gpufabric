"""Runtime state tracker for the worker node."""

import socket
import threading
import time
from typing import Dict, Optional


class WorkerState:
    """Thread-safe state holder for worker ID, active tasks, heartbeats, and workloads."""

    def __init__(self, worker_id: Optional[str] = None):
        self.hostname = socket.gethostname()
        self.worker_id = worker_id or f"worker-{self.hostname}"
        self._active_tasks = 0
        self._start_time = time.time()
        self._last_heartbeat = self._start_time
        self._device_tasks: Dict[int, int] = {}
        self._device_workloads: Dict[int, str] = {}
        self._lock = threading.Lock()

    @property
    def active_tasks(self) -> int:
        with self._lock:
            return self._active_tasks

    @property
    def last_heartbeat(self) -> float:
        with self._lock:
            return self._last_heartbeat

    @property
    def uptime(self) -> float:
        return time.time() - self._start_time

    def heartbeat(self) -> float:
        with self._lock:
            self._last_heartbeat = time.time()
            return self._last_heartbeat

    def increment_tasks(self, device_index: int = 0, workload: str = "COMPUTE"):
        with self._lock:
            self._active_tasks += 1
            self._device_tasks[device_index] = self._device_tasks.get(device_index, 0) + 1
            self._device_workloads[device_index] = workload
            self._last_heartbeat = time.time()

    def decrement_tasks(self, device_index: int = 0):
        with self._lock:
            self._active_tasks = max(0, self._active_tasks - 1)
            remaining = max(0, self._device_tasks.get(device_index, 1) - 1)
            if remaining == 0:
                self._device_tasks.pop(device_index, None)
                self._device_workloads.pop(device_index, None)
            else:
                self._device_tasks[device_index] = remaining
            self._last_heartbeat = time.time()

    def get_device_workload(self, device_index: int = 0) -> str:
        with self._lock:
            return self._device_workloads.get(device_index, "IDLE")
