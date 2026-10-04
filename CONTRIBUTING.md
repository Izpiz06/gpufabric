# Contributing to GPU Fabric

Thank you for contributing to **GPU Fabric**! We welcome bug fixes, documentation improvements, and new features from the community.

Please take a moment to review this guide to ensure a smooth and effective contribution process.

---

## 📜 Table of Contents

- [Code of Conduct & Core Principles](#-core-principles)
- [Developer Certificate of Origin (DCO) & Commit Signing](#-developer-certificate-of-origin-dco--commit-signing)
- [Copyright & License Headers](#-copyright--license-headers)
- [Development Setup](#-development-setup)
- [Code Style & Verification](#-code-style--verification)
- [Pull Request Guidelines](#-pull-request-guidelines)
- [Current Maintainers & Contributors](#-maintainers--contributors)

---

## 🎯 Core Principles

1. **Keep it Modular & Small**: Avoid monolithic files; divide functionality into clean, single-responsibility modules.
2. **Layered Architecture**:
   - `common/`: Shared protobuf definitions, constants, discovery, and TLS utilities.
   - `worker/`: NVML telemetry, gRPC server, and CuPy GPU execution.
   - `client/`: Python SDK (`GPUFabricClient`), CLI subcommands, and interactive TUI.
3. **Security First**: Mutual TLS (mTLS) is enabled by default. Never introduce unauthenticated or plaintext pathways without explicit opt-in flags.

---

## ✍️ Developer Certificate of Origin (DCO) & Commit Signing

All contributions must adhere to the **Developer Certificate of Origin (DCO)**. To certify compliance, **all commits must be signed off** using the `-s` or `--signoff` flag:

```bash
git commit -s -m "feat(module): add new capability"
```

This appends a `Signed-off-by` trailer with your git committer name and email:

```text
Signed-off-by: Your Name <your.email@example.com>
```

### Git Setup
Ensure your git environment is properly configured:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

## 🏷️ Copyright & License Headers

Every source file must include a standard Apache-2.0 copyright and license header at the very top.

### Rules:
- **File Creator Attribution**: Whoever creates/starts a new file must place their name in the copyright header.
- **Python Files (`.py`)**:
  ```python
  # Copyright (c) 2026 <Author Name>
  # SPDX-License-Identifier: Apache-2.0
  ```
- **Protocol Buffer Files (`.proto`)**:
  ```protobuf
  // Copyright (c) 2026 <Author Name>
  // SPDX-License-Identifier: Apache-2.0
  ```
- **Verification Tool**:
  Run the automated header check script before submitting changes:
  ```bash
  python scripts/check_headers.py
  ```

---

## 🛠️ Development Setup

1. **Clone Repository & Set Up Virtual Environment**:
   ```bash
   git clone https://github.com/Izpiz06/gpufabric.git
   cd gpufabric
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

---

## 🧪 Code Style & Verification

All code must pass formatting, linting, copyright verification, and automated tests.

### 1. Run Copyright Header Check
```bash
python scripts/check_headers.py
```

### 2. Run Ruff Linter & Formatter
```bash
# Check code style
ruff check .

# Check formatting
ruff format --check .

# Automatically apply fixes
ruff check --fix .
ruff format .
```

### 3. Run Test Suite
```bash
# Run all unit tests
pytest -v

# Run with coverage
pytest --cov=client --cov=worker --cov=common tests/
```

### 4. Regenerate Protocol Buffers (if modifying `proto/gpufabric.proto`)
```bash
python scripts/gen_proto.py
```

---

## 🚀 Pull Request Guidelines

1. **Create a Topic Branch**:
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. **Make Focused, Atomic Commits**:
   Keep commit diffs small, well-described, and signed-off (`git commit -s`).
3. **Verify Locally**:
   Ensure `pytest -v`, `ruff check .`, `ruff format --check .`, and `python scripts/check_headers.py` all pass.
4. **Open a Pull Request**:
   Describe your changes clearly, linking any relevant issues.

---

## 👥 Maintainers & Contributors

- **Mohammad Izaan** ([@Izpiz06](https://github.com/Izpiz06))
- **Shreyas Mene** ([@shreyasmene](https://github.com/shreyasmene))
