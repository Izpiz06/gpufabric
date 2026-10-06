# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Tests for CUDA runtime diagnostics and missing library probing."""

from unittest.mock import MagicMock, patch

from worker.diagnostics import (
    CudaDiagnostics,
    _extract_missing_library_names,
    probe_cuda_environment,
)


def test_extract_missing_library_names():
    err1 = "libcublas.so.12: cannot open shared object file: No such file or directory"
    assert _extract_missing_library_names(err1) == ["libcublas.so.12"]

    err2 = "Could not load dynamic library 'cublas64_12.dll'"
    assert _extract_missing_library_names(err2) == ["cublas64_12.dll"]

    err3 = "libcudart.so.11.0: cannot open shared object file"
    assert _extract_missing_library_names(err3) == ["libcudart.so.11.0"]

    err4 = "Internal unknown failure"
    assert _extract_missing_library_names(err4) == []


def test_diagnostics_status_and_properties():
    # Healthy case
    d_healthy = CudaDiagnostics(
        gpu_detected=True,
        gpu_name="NVIDIA RTX 4090",
        gpu_count=1,
        nvml_available=True,
        cupy_available=True,
        cudart_available=True,
        cudart_version="12.4",
        cublas_available=True,
        cublas_version="12.4",
    )
    assert d_healthy.is_healthy is True
    assert d_healthy.status == "HEALTHY"
    summary = d_healthy.summary()
    assert "Worker Status: HEALTHY" in summary
    assert "NVIDIA RTX 4090" in summary
    assert "CUDA runtime: AVAILABLE (12.4)" in summary
    assert "CuPy: AVAILABLE" in summary
    assert "cuBLAS: AVAILABLE (12.4)" in summary

    # Degraded case: missing cuBLAS
    d_degraded = CudaDiagnostics(
        gpu_detected=True,
        gpu_name="NVIDIA RTX 4090",
        gpu_count=1,
        nvml_available=True,
        cupy_available=True,
        cudart_available=True,
        cudart_version="12.4",
        cublas_available=False,
        missing_libraries=["libcublas.so.12"],
        warnings=["Some operations may be unavailable."],
        remediation_hints=["Install nvidia-cublas-cu12."],
    )
    assert d_degraded.is_healthy is False
    assert d_degraded.status == "DEGRADED"
    summary = d_degraded.summary()
    assert "Worker Status: DEGRADED" in summary
    assert "cuBLAS: MISSING (libcublas.so.12 not found)" in summary
    assert "Warning: Some operations may be unavailable." in summary
    assert "Hint: Install nvidia-cublas-cu12." in summary

    # Unavailable case: no GPU and no CuPy
    d_unavail = CudaDiagnostics(
        gpu_detected=False,
        cupy_available=False,
    )
    assert d_unavail.is_healthy is False
    assert d_unavail.status == "UNAVAILABLE"


def test_probe_cuda_environment_with_missing_cublas():
    mock_gpu = MagicMock()
    mock_gpu.is_available.return_value = True
    mock_gpu.device_count.return_value = 1
    mock_gpu.get_driver_version.return_value = "550.54"
    mock_gpu.get_cuda_version.return_value = "12.4"
    mock_gpu.get_info.return_value = {"name": "NVIDIA RTX 4090"}

    mock_cp = MagicMock()
    mock_cp.cuda.runtime.getDeviceCount.return_value = 1
    mock_cp.cuda.runtime.runtimeGetVersion.return_value = 12040

    mock_cublas = MagicMock()
    mock_cublas.create.side_effect = OSError(
        "libcublas.so.12: cannot open shared object file: No such file or directory"
    )
    mock_cp.cuda.cublas = mock_cublas

    with (
        patch.dict("sys.modules", {"cupy": mock_cp, "cupy.cuda.cublas": mock_cublas}),
        patch("ctypes.util.find_library", return_value=None),
    ):
        diag = probe_cuda_environment(gpu_manager=mock_gpu)

    assert diag.gpu_detected is True
    assert diag.gpu_name == "NVIDIA RTX 4090"
    assert diag.cupy_available is True
    assert diag.cudart_available is True
    assert diag.cudart_version == "12.4"
    assert diag.cublas_available is False
    assert "libcublas.so.12" in diag.missing_libraries
    assert diag.status == "DEGRADED"
    assert any("nvidia-cublas-cu12" in hint for hint in diag.remediation_hints)


def test_probe_cuda_environment_healthy():
    mock_gpu = MagicMock()
    mock_gpu.is_available.return_value = True
    mock_gpu.device_count.return_value = 1
    mock_gpu.get_driver_version.return_value = "550.54"
    mock_gpu.get_cuda_version.return_value = "12.4"
    mock_gpu.get_info.return_value = {"name": "NVIDIA RTX 4090"}

    mock_cp = MagicMock()
    mock_cp.cuda.runtime.getDeviceCount.return_value = 1
    mock_cp.cuda.runtime.runtimeGetVersion.return_value = 12040

    mock_cublas = MagicMock()
    mock_cublas.create.return_value = 12345
    mock_cublas.getVersion.return_value = 12040

    with patch.dict("sys.modules", {"cupy": mock_cp, "cupy.cuda.cublas": mock_cublas}):
        diag = probe_cuda_environment(gpu_manager=mock_gpu)

    assert diag.gpu_detected is True
    assert diag.cupy_available is True
    assert diag.cudart_available is True
    assert diag.cublas_available is True
    assert diag.status == "HEALTHY"
    assert diag.is_healthy is True
