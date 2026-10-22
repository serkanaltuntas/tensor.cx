"""Explicit contiguous shared allocations across supported device pairs."""

from __future__ import annotations

from . import backend as _backend
from .device import Device, _normalize_device
from .tensor import Tensor, empty, tensor


def _route(devices):
    if not isinstance(devices, (tuple, list)) or len(devices) != 2:
        raise ValueError("devices must contain exactly two explicit devices")
    if any(not isinstance(value, (str, Device)) for value in devices):
        raise TypeError("shared devices must be strings or Device objects")
    first, second = (_normalize_device(value) for value in devices)
    return first, _backend.shared_route(first, second)


def shared_empty(shape, dtype: str = "float32", *, devices) -> tuple[Tensor, Tensor]:
    """Allocate two aliases; the first device selects the physical allocator.

    Contents are unspecified. Unsupported pairs fail instead of copying through
    CPU. A repeated device produces two wrappers sharing one ordinary allocation.
    """
    first, route = _route(devices)
    if route is None:
        result = empty(shape, dtype=dtype, device=first)
        return result, Tensor(result._impl)
    left, right = route.empty(shape, dtype)
    return Tensor(left), Tensor(right)


def shared_tensor(data, dtype: str | None = None, *, devices) -> tuple[Tensor, Tensor]:
    """Initialize shared storage from CPU data and return aliases in device order.

    Tensor creation and handoffs are synchronous. Foreign CUDA consumers must
    complete their writes before a tensor.cx operation accesses either alias.
    """
    first, route = _route(devices)
    if route is None:
        result = tensor(data, dtype=dtype, device=first)
        return result, Tensor(result._impl)
    host = tensor(data, dtype=dtype, device="cpu")
    left, right = route.tensor(host._impl)
    return Tensor(left), Tensor(right)
