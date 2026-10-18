"""Device identity/routing contracts, independent of GPU availability."""
from dataclasses import replace
import importlib
import os
import subprocess
import sys

import pytest

import tensorcx as cx
from tensorcx import backend
from tensorcx.device import Device, _normalize_device


def test_indexed_registry_canonicalization_and_display_names(monkeypatch):
    routes = dict(backend._BACKENDS)
    base = replace(routes["cpu"], name="mockgpu", indexed=True,
                   device_name=lambda: "First GPU")
    routes["mockgpu"] = base
    routes["mockgpu:1"] = replace(base, name="mockgpu:1", device_name=lambda: "Second GPU")
    monkeypatch.setattr(backend, "_BACKENDS", routes)
    assert _normalize_device("mockgpu:0") == "mockgpu"
    assert _normalize_device(Device("mockgpu", 1)) == "mockgpu:1"
    device_module = importlib.import_module("tensorcx.device")
    second = device_module.device("mockgpu:1")
    assert (second.type, second.index, second.name) == ("mockgpu", 1, "Second GPU")
    assert second == Device("mockgpu", 1, "A different display label")
    assert hash(second) == hash(Device("mockgpu", 1))
    assert second != Device("mockgpu", 0)
    with pytest.raises(ValueError, match="non-negative"):
        _normalize_device("mockgpu:-1")
    with pytest.raises(ValueError, match="not available"):
        device_module.device("mockgpu:2")
    for value in (Device("mockgpu", True), Device("mockgpu", 1.5)):
        with pytest.raises(ValueError, match="integer"):
            _normalize_device(value)
    with pytest.raises(ValueError, match="index 0"):
        _normalize_device("cpu:1")


def test_vulkan_without_a_driver_preserves_cpu(tmp_path):
    environment = dict(os.environ, VK_DRIVER_FILES=str(tmp_path / "missing-icd.json"),
                       VK_ICD_FILENAMES=str(tmp_path / "missing-icd.json"))
    environment.pop("TENSORCX_VULKAN_VALIDATION", None)
    result = subprocess.run([sys.executable, "-c", """
import tensorcx as cx
assert not cx.is_available('vulkan')
assert not any(d.startswith('vulkan') for d in cx.devices())
assert (cx.ones((3,), device='cpu') + 2).numpy().tolist() == [3, 3, 3]
assert cx.diagnostics()['backends']['vulkan']['available'] is False
"""], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_indexed_diagnostics_use_family_build_flag(monkeypatch):
    diag = importlib.import_module("tensorcx.diagnostics")
    routes = dict(backend._BACKENDS)
    routes["vulkan:1"] = replace(routes["cpu"], name="vulkan:1", indexed=True)
    monkeypatch.setattr(backend, "_BACKENDS", routes)
    monkeypatch.setattr(backend, "_BACKEND_ORDER", ["cpu", "vulkan:1"])
    monkeypatch.setattr(cx._core, "_build_info", lambda: {"cpu": True, "vulkan": True})
    assert diag.diagnostics()["backends"]["vulkan:1"]["compiled"] is True


def test_vulkan_import_without_loader():
    shim = os.environ.get("TENSORCX_TEST_VULKAN_NO_LOADER")
    if not shim:
        pytest.skip("no-loader interposer is built by the native Vulkan acceptance target")
    environment = dict(os.environ, LD_PRELOAD=shim)
    environment.pop("TENSORCX_VULKAN_VALIDATION", None)
    result = subprocess.run([sys.executable, "-c", """
import tensorcx as cx
assert cx._core._build_info()['vulkan']
assert not cx.is_available('vulkan')
assert 'loader' in cx._core._vulkan_discovery_error()
assert (cx.ones((3,), device='cpu') + 2).numpy().tolist() == [3, 3, 3]
"""], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_vulkan_incomplete_loader_preserves_cpu():
    shim = os.environ.get("TENSORCX_TEST_VULKAN_NO_LOADER")
    fake = os.environ.get("TENSORCX_TEST_VULKAN_INCOMPLETE_LIBRARY")
    if not shim or not fake:
        pytest.skip("incomplete loader fixture is built by the native Vulkan acceptance target")
    environment = dict(os.environ, LD_PRELOAD=shim, TENSORCX_TEST_VULKAN_INCOMPLETE_LOADER=fake)
    environment.pop("TENSORCX_VULKAN_VALIDATION", None)
    result = subprocess.run([sys.executable, "-c", """
import tensorcx as cx
assert cx._core._build_info()['vulkan']
assert not cx.is_available('vulkan')
assert 'incomplete' in cx._core._vulkan_discovery_error()
assert (cx.ones((3,), device='cpu') + 2).numpy().tolist() == [3, 3, 3]
"""], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_vulkan_registry_callbacks_keep_concrete_index(monkeypatch):
    monkeypatch.setattr(backend, "_BACKENDS", {})
    monkeypatch.setattr(backend, "_BACKEND_ORDER", [])
    assert backend.devices() == []
    monkeypatch.setattr(cx._core, "is_available", lambda name: name in {"vulkan", "vulkan:1"})
    monkeypatch.setattr(cx._core, "device_name", lambda name: name)
    monkeypatch.setattr(cx._core, "cpu_to_vulkan", lambda value, index: (value, index), raising=False)
    monkeypatch.setattr(cx._core, "_vulkan_empty", lambda shape, dtype, index: (shape, dtype, index), raising=False)
    monkeypatch.setattr(cx._core, "fill", lambda shape, dtype, value, device: (shape, dtype, value, device))
    backend._register_vulkan_device(0)
    assert backend.devices() == ["vulkan"]
    backend._register_vulkan_device(1)
    assert backend.devices() == ["vulkan", "vulkan:1"]
    assert backend.best_device() == "vulkan"
    second = backend.require_backend("vulkan:1")
    assert second.copy_from_cpu("payload") == ("payload", 1)
    assert second.empty((3,), "int32") == ((3,), "int32", 1)
    assert second.fill((3,), "float32", 2) == ((3,), "float32", 2, "vulkan:1")


def test_software_vulkan_is_not_listed_as_a_gpu():
    from pathlib import Path
    manifests = sorted(Path('/usr/share/vulkan/icd.d').glob('lvp_icd*.json'))
    if not manifests:
        pytest.skip('lavapipe ICD unavailable for CPU-device filtering test')
    environment = dict(os.environ, VK_DRIVER_FILES=str(manifests[0]), VK_ICD_FILENAMES=str(manifests[0]))
    environment.pop('TENSORCX_VULKAN_VALIDATION', None)
    result = subprocess.run([sys.executable, '-c', """
import tensorcx as cx
assert not cx.is_available('vulkan')
assert not any(d.startswith('vulkan') for d in cx.devices())
assert cx.ones((2,), device='cpu').numpy().tolist() == [1, 1]
"""], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_partial_backend_gates_preserve_legacy_contracts(monkeypatch):
    from backend_support import require_operations
    monkeypatch.setattr(cx, 'capabilities', lambda device: {})
    for device in ('cpu', 'metal', 'cuda'):
        with pytest.raises(pytest.fail.Exception, match='lost required'):
            require_operations(device, 'astype')
    with pytest.raises(pytest.skip.Exception, match='does not declare'):
        require_operations('vulkan', 'astype')
