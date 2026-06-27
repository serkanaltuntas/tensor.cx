"""Backend registry used by the public Python API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from . import _core


@dataclass(frozen=True, slots=True)
class Backend:
    """String-keyed backend route for Python-level device dispatch."""

    name: str
    priority: int
    is_available: Callable[[], bool]
    device_name: Callable[[], str]
    unavailable_message: str | None = None
    copy_from_cpu: Callable[[Any], Any] | None = None
    copy_to_cpu: Callable[[Any], Any] | None = None
    fill: Callable[[Any, str, float], Any] | None = None
    matmul_backends: Callable[[], list[str]] | None = None


_BACKENDS: dict[str, Backend] = {}
_BACKEND_ORDER: list[str] = []


def _register_backend(backend: Backend) -> None:
    if not backend.name:
        raise ValueError("backend name must be non-empty")
    if backend.name in _BACKENDS:
        raise ValueError(f"backend is already registered: {backend.name}")
    _BACKENDS[backend.name] = backend
    _BACKEND_ORDER.append(backend.name)


def backend_names(*, include_unavailable: bool = False) -> list[str]:
    """Return registered backend names in stable registration order."""

    if include_unavailable:
        return list(_BACKEND_ORDER)
    return devices()


def _default_unavailable_message(name: str) -> str:
    return f"device is not available: {name}"


def is_available(name: str) -> bool:
    backend = _BACKENDS.get(name)
    return bool(backend is not None and backend.is_available())


def devices() -> list[str]:
    return [name for name in _BACKEND_ORDER if is_available(name)]


def require_backend(name: str) -> Backend:
    backend = _BACKENDS.get(name)
    if backend is None:
        raise ValueError(_default_unavailable_message(name))
    if not backend.is_available():
        raise ValueError(backend.unavailable_message or _default_unavailable_message(name))
    return backend


def device_name(name: str) -> str:
    return require_backend(name).device_name()


def best_device() -> str:
    available = [require_backend(name) for name in devices()]
    if not available:
        raise RuntimeError("no runtime backend is available")
    return max(available, key=lambda backend: backend.priority).name


def copy_tensor(impl: Any, source: str, target: str) -> Any:
    if target == source:
        return impl

    target_backend = require_backend(target)
    if target == "cpu":
        source_backend = require_backend(source)
        if source_backend.copy_to_cpu is None:
            raise ValueError(f"cannot copy tensor from device {source!r} to CPU")
        return source_backend.copy_to_cpu(impl)

    if source != "cpu":
        raise ValueError(f"unsupported device transfer: {source!r} -> {target!r}")
    if target_backend.copy_from_cpu is None:
        raise ValueError(f"unsupported device transfer: {source!r} -> {target!r}")
    return target_backend.copy_from_cpu(impl)


def fill(name: str, shape: Any, dtype: str, value: float) -> Any:
    backend = require_backend(name)
    if backend.fill is None:
        raise ValueError(f"backend does not support fill: {name}")
    return backend.fill(shape, dtype, value)


def matmul_backends(name: str) -> list[str]:
    backend = require_backend(name)
    if backend.matmul_backends is None:
        raise ValueError(f"backend does not support matmul: {name}")
    return backend.matmul_backends()


def _cpu_available() -> bool:
    return True


def _cpu_name() -> str:
    return "CPU"


def _cpu_fill(shape: Any, dtype: str, value: float) -> Any:
    return _core.fill(shape, dtype=dtype, value=value, device="cpu")


def _cpu_matmul_backends() -> list[str]:
    return list(_core.matmul_backends("cpu"))


def _metal_available() -> bool:
    return bool(_core.is_available("metal"))


def _metal_name() -> str:
    return _core.device_name("metal")


def _metal_from_cpu(impl: Any) -> Any:
    return _core.cpu_to_metal(impl)


def _metal_to_cpu(impl: Any) -> Any:
    return _core.metal_to_cpu(impl)


def _metal_fill(shape: Any, dtype: str, value: float) -> Any:
    return _core.fill(shape, dtype=dtype, value=value, device="metal")


def _metal_matmul_backends() -> list[str]:
    return list(_core.matmul_backends("metal"))


_register_backend(
    Backend(
        name="cpu",
        priority=0,
        is_available=_cpu_available,
        device_name=_cpu_name,
        copy_to_cpu=lambda impl: impl,
        fill=_cpu_fill,
        matmul_backends=_cpu_matmul_backends,
    )
)
_register_backend(
    Backend(
        name="metal",
        priority=100,
        is_available=_metal_available,
        device_name=_metal_name,
        unavailable_message="Metal is not available on this system",
        copy_from_cpu=_metal_from_cpu,
        copy_to_cpu=_metal_to_cpu,
        fill=_metal_fill,
        matmul_backends=_metal_matmul_backends,
    )
)
