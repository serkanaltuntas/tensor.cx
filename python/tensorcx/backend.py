"""Backend registry used by the public Python API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from . import _core
from ._capabilities import CPU_OPERATIONS, CUDA_OPERATIONS, METAL_OPERATIONS, VULKAN_OPERATIONS


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
    operations: dict[str, tuple[str, ...]] | None = None
    diagnostic_info: Callable[[], dict[str, Any]] | None = None
    memory_pool: Callable[[int, int], Any] | None = None
    empty: Callable[[Any, str], Any] | None = None
    indexed: bool = False
    clone: Callable[[Any], Any] | None = None


_BACKENDS: dict[str, Backend] = {}
_BACKEND_ORDER: list[str] = []


@dataclass(frozen=True, slots=True)
class StorageExchange:
    """Explicit shared allocation route, ordered by the physical allocator."""

    validate: Callable[[], None]
    empty: Callable[[Any, str], tuple[Any, Any]]
    tensor: Callable[[Any], tuple[Any, Any]]


_STORAGE_EXCHANGES: dict[tuple[str, str], StorageExchange] = {}


def shared_route(first: str, second: str) -> StorageExchange | None:
    require_backend(first)
    require_backend(second)
    if first == second:
        return None
    route = _STORAGE_EXCHANGES.get((first, second))
    if route is None:
        raise ValueError(f"shared storage is unsupported for devices {first!r}, {second!r}")
    route.validate()
    return route


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


def allows_device_index(name: str) -> bool:
    backend = _BACKENDS.get(name)
    return bool(backend is not None and backend.indexed)


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


def copy_tensor(impl: Any, source: str, target: str, *, copy: bool = False) -> Any:
    """Synchronous explicit placement; accelerator pairs stage through CPU."""
    if type(copy) is not bool:
        raise TypeError("copy must be bool")
    if target == source and not copy:
        return impl

    target_backend = require_backend(target)
    source_backend = require_backend(source)
    if target == source:
        if target_backend.clone is None:
            raise ValueError(f"device does not support an independent copy: {target!r}")
        return target_backend.clone(impl)
    if target == "cpu":
        if source_backend.copy_to_cpu is None:
            raise ValueError(f"cannot copy tensor from device {source!r} to CPU")
        return source_backend.copy_to_cpu(impl)

    if target_backend.copy_from_cpu is None:
        raise ValueError(f"unsupported device transfer: {source!r} -> {target!r}")
    if source == "cpu":
        return target_backend.copy_from_cpu(impl)
    if source_backend.copy_to_cpu is None:
        raise ValueError(f"unsupported device transfer: {source!r} -> {target!r}")
    # Both legs complete before returning. Keep staging alive through upload;
    # never replace an allocation/copy failure with another device or an alias.
    staging = source_backend.copy_to_cpu(impl)
    return target_backend.copy_from_cpu(staging)


def _clone_contiguous(impl: Any) -> Any:
    # Existing identity transpose returns independent contiguous storage and
    # preserves dtype/shape/bits. Vulkan provides its direct device-copy route.
    return _core.transpose(impl, tuple(range(len(impl.shape))))


def empty(name: str, shape: Any, dtype: str) -> Any:
    backend = require_backend(name)
    if backend.empty is not None:
        return backend.empty(shape, dtype)
    # Preserve the existing CPU factory and copy route for other backends.
    host = _core.empty(shape, dtype=dtype, device="cpu")
    return host if name == "cpu" else copy_tensor(host, "cpu", name)


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


def _cpu_diagnostic_info() -> dict[str, Any]:
    from .diagnostics import _host_memory
    return {"name": "CPU", "memory": _host_memory()}


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
        operations=CPU_OPERATIONS,
        diagnostic_info=_cpu_diagnostic_info,
        clone=_clone_contiguous,
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
        operations=METAL_OPERATIONS,
        clone=_clone_contiguous,
    )
)

# Register the optional accelerator only when the native runtime can use it.
# Device selection and copies continue through the same string-keyed registry.
if _core.is_available("cuda"):
    _register_backend(
        Backend(
            name="cuda",
            priority=100,
            is_available=lambda: bool(_core.is_available("cuda")),
            device_name=lambda: _core.device_name("cuda"),
            copy_from_cpu=lambda impl: _core.cpu_to_cuda(impl),
            copy_to_cpu=lambda impl: _core.cuda_to_cpu(impl),
            fill=lambda shape, dtype, value: _core.fill(
                shape, dtype=dtype, value=value, device="cuda"
            ),
            matmul_backends=lambda: list(_core.matmul_backends("cuda")),
            operations=CUDA_OPERATIONS,
            diagnostic_info=lambda: dict(_core._cuda_device_info()),
            memory_pool=lambda size, blocks: _core._CudaMemoryPool.create(size, blocks),
            empty=lambda shape, dtype: _core._cuda_empty(shape, dtype=dtype),
            clone=_clone_contiguous,
        )
    )


def _register_vulkan_device(index: int) -> None:
    name = "vulkan" if index == 0 else f"vulkan:{index}"
    _register_backend(
        Backend(
            name=name,
            priority=50,
            indexed=True,
            is_available=lambda: bool(_core.is_available(name)),
            device_name=lambda: _core.device_name(name),
            copy_from_cpu=lambda impl: _core.cpu_to_vulkan(impl, index=index),
            copy_to_cpu=lambda impl: _core.vulkan_to_cpu(impl),
            fill=lambda shape, dtype, value: _core.fill(
                shape, dtype=dtype, value=value, device=name
            ),
            empty=lambda shape, dtype: _core._vulkan_empty(shape, dtype=dtype, index=index),
            matmul_backends=lambda: list(_core.matmul_backends(name)),
            operations=VULKAN_OPERATIONS,
            diagnostic_info=lambda: dict(_core._vulkan_device_info(index=index)),
            memory_pool=lambda size, blocks: _core._VulkanMemoryPool.create(index, size, blocks),
            clone=lambda impl: _core._vulkan_clone(impl),
        )
    )
    if "cuda" in _BACKENDS and hasattr(_core, "_shared_cuda_vulkan_support"):
        for cuda_origin in (False, True):
            pair = ("cuda", name) if cuda_origin else (name, "cuda")
            _STORAGE_EXCHANGES[pair] = StorageExchange(
                validate=lambda cuda_origin=cuda_origin: _core._shared_cuda_vulkan_support(index, cuda_origin),
                empty=lambda shape, dtype, cuda_origin=cuda_origin: _core._shared_cuda_vulkan_empty(
                    shape, dtype, index, cuda_origin),
                tensor=lambda host, cuda_origin=cuda_origin: _core._shared_cuda_vulkan_tensor(
                    host, index, cuda_origin),
            )


if _core._build_info().get("vulkan", False):
    # Keep index-zero discovery visible to diagnostics even without a loader/ICD.
    for _index in range(max(1, _core._vulkan_device_count())):
        _register_vulkan_device(_index)
