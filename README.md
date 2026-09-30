# GPU Fabric (Phase 1)

Easily discover, access, and control GPUs across machines on the same local network, allowing multiple machines to behave like a manageable GPU cluster.

---

## Phase 1 Architecture

Phase 1 implements a lightweight, high-performance client-worker prototype for remote GPU access over LAN.

```text
Machine A (Client)
        |
        | HTTP / JSON over LAN
        v
Machine B (GPU Worker)
        |
        +--> GPU Manager (pynvml / NVML)
        +--> GPU Executor (CuPy / PyCUDA / CUDA Kernel)
        |
        v
   NVIDIA GPU
```

---

## Directory Structure

```text
gpufabric/
├── client/
│   ├── __init__.py      # Exports GPUFabricClient
│   ├── client.py        # Python client library
│   ├── cli.py           # Rich CLI interface
│   └── __main__.py      # python -m client entrypoint
├── worker/
│   ├── __init__.py      # Exports create_app, app, GPUManager, GPUExecutor
│   ├── app.py           # FastAPI worker application
│   ├── gpu.py           # NVML device discovery and hardware telemetry
│   ├── executor.py      # Real GPU kernel execution (CuPy/PyCUDA/CUDA)
│   ├── server.py        # Worker CLI runner
│   └── __main__.py      # python -m worker entrypoint
├── common/
│   ├── __init__.py
│   ├── models.py        # Pydantic v2 data models for requests/responses
│   └── protocol.py      # Constants, endpoints, formatting helpers
├── tests/
│   ├── test_common.py
│   ├── test_gpu_manager.py
│   ├── test_executor.py
│   ├── test_worker_api.py
│   └── test_client.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Worker Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Returns worker status, worker ID, version, and GPU availability |
| `GET` | `/gpu` | Returns GPU model name, total/free/used VRAM, compute capability, and driver version |
| `GET` | `/status` | Returns real-time GPU core & memory utilization %, temperature, VRAM usage, and active tasks |
| `POST` | `/execute` | Executes real GPU workloads (vector addition: $C = A + B$) directly on device memory |

---

## Installation

### Prerequisites
- Python 3.9+
- NVIDIA GPU with drivers installed (for Worker nodes)
- CUDA Toolkit (matching your CUDA driver version)

### Install Dependencies

```bash
git clone https://github.com/your-username/gpufabric.git
cd gpufabric

pip install -r requirements.txt
```

#### GPU Acceleration Backend
On the **Worker machine**, install your preferred GPU runtime:
- **CuPy** (Recommended):
  ```bash
  pip install cupy-cuda12x  # For CUDA 12.x
  # or
  pip install cupy-cuda11x  # For CUDA 11.x
  ```
- **PyCUDA**:
  ```bash
  pip install pycuda
  ```

---

## Quickstart

### 1. Start the Worker (Machine B with GPU)

Run the worker server on the machine with the GPU:

```bash
python -m worker --host 0.0.0.0 --port 8000
```

Options:
- `--host`: Host IP to bind (default: `0.0.0.0`)
- `--port`: Port number (default: `8000`)
- `--worker-id`: Custom worker identifier (defaults to hostname)
- `--log-level`: Logging verbosity (`debug`, `info`, `warning`, `error`)

### 2. Use the Client CLI (Machine A)

From another machine on the LAN (or locally):

#### Discover / Health Check
```bash
python -m client discover <worker-ip>
# Example:
python -m client discover 192.168.1.50
```

#### Query GPU Hardware Specifications
```bash
python -m client gpu <worker-ip>
# Example:
python -m client gpu 192.168.1.50 --device 0
```

#### Query Real-Time GPU Status & Telemetry
```bash
python -m client status <worker-ip>
# Example:
python -m client status 192.168.1.50
```

#### Submit Vector Addition Workload ($C = A + B$)
Execute custom vectors:
```bash
python -m client execute <worker-ip> --a 1.0 2.0 3.0 4.0 --b 10.0 20.0 30.0 40.0
```

Or benchmark with generated vectors:
```bash
python -m client execute <worker-ip> --size 1000000
```

---

## Python API Usage

You can also interact with the worker programmatically in Python:

```python
from client import GPUFabricClient

# Connect to the remote worker
with GPUFabricClient(host="192.168.1.50", port=8000) as client:
    # 1. Check worker health
    health = client.health()
    print(f"Worker {health.worker_id} status: {health.status}")

    # 2. Query GPU specs
    gpu_info = client.get_gpu_info(device_index=0)
    print(f"GPU: {gpu_info.name}, Total VRAM: {gpu_info.total_vram_human}")

    # 3. Query dynamic status
    status = client.get_status(device_index=0)
    print(f"GPU Utilization: {status.gpu_utilization_pct}%, Free VRAM: {status.free_vram_human}")

    # 4. Execute vector addition on the remote GPU
    response = client.execute_vector_add(
        a=[1.5, 2.5, 3.5],
        b=[10.0, 20.0, 30.0],
        device_index=0,
    )
    print(f"Result: {response.result}")
    print(f"Execution time on GPU: {response.execution_time_ms} ms")
```

---

## Running Tests & Quality Checks

Run the full automated test suite:

```bash
pytest -v
```

Run code formatting and linting:

```bash
ruff check .
```

---

## Continuous Integration & Prebuilds

This project includes automated GitHub Actions workflows:

* **CI Matrix (`.github/workflows/ci.yml`)**: Automatically runs the test suite across Python 3.9, 3.10, 3.11, 3.12, and 3.13 on every push and pull request, in addition to Ruff linting.
* **Prebuilds (`.github/workflows/build.yml`)**: Builds standard Python wheels (`.whl`) and source distributions (`.tar.gz`), validates them with Twine, and uploads distribution artifacts for easy installation and releases.

