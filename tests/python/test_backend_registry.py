import pytest

import tensorcx as cx
from tensorcx import backend as _backend


def test_python_backend_registry_lists_registered_and_available_backends():
    registered = _backend.backend_names(include_unavailable=True)

    assert registered[0] == "cpu"
    assert "metal" in registered
    assert _backend.backend_names() == cx.devices()
    assert _backend.is_available("cpu")
    assert _backend.is_available("cuda") == ("cuda" in cx.devices())


def test_python_backend_registry_reports_unavailable_devices_consistently():
    with pytest.raises(ValueError, match="device is not available: missing"):
        _backend.require_backend("missing")

    if not cx.is_available("metal"):
        with pytest.raises(ValueError, match="Metal is not available on this system"):
            _backend.require_backend("metal")
    if not cx.is_available("cuda"):
        with pytest.raises(ValueError, match="device is not available: cuda"):
            _backend.require_backend("cuda")


def test_public_device_helpers_route_through_registry():
    assert cx.devices() == _backend.devices()
    assert cx.best_device() == _backend.best_device()
    assert cx.device_name("cpu") == "CPU"
    assert cx.device("cpu").name == _backend.device_name("cpu")


def test_public_matmul_backend_options_require_available_device():
    assert cx.matmul_backends("cpu") == ["auto", "cpu", "reference"]

    if not cx.is_available("metal"):
        with pytest.raises(ValueError, match="Metal is not available on this system"):
            cx.matmul_backends("metal")


def test_native_backend_route_table_covers_cpu_and_unknown_devices():
    assert cx._core.is_available("cpu")
    assert cx._core.device_name("cpu") == "CPU"
    assert list(cx._core.devices()) == cx.devices()
    assert list(cx._core.matmul_backends("cpu")) == ["auto", "cpu", "reference"]

    with pytest.raises(ValueError, match="device is not available: missing"):
        cx._core.device_name("missing")
    with pytest.raises(ValueError, match="device is not available: missing"):
        cx._core.fill((1,), dtype=cx.float32, value=0.0, device="missing")
