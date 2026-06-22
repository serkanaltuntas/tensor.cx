"""Tensor API wrappers for Cortex Runtime."""

from __future__ import annotations

from typing import Any

from . import _core
from .device import Device, _normalize_device


class Tensor:
    """Public tensor wrapper over backend-specific native tensor objects."""

    def __init__(self, impl: Any):
        self._impl = impl

    @property
    def shape(self) -> tuple[int, ...]:
        return self._impl.shape

    @property
    def strides(self) -> tuple[int, ...]:
        return self._impl.strides

    @property
    def dtype(self) -> str:
        return self._impl.dtype

    @property
    def device(self) -> str:
        return self._impl.device

    @property
    def nbytes(self) -> int:
        return self._impl.nbytes

    def numpy(self):
        return self.cpu()._impl.numpy()

    def cpu(self) -> "Tensor":
        if self.device == "cpu":
            return self
        if self.device == "metal" and hasattr(_core, "metal_to_cpu"):
            return Tensor(_core.metal_to_cpu(self._impl))
        raise ValueError(f"cannot copy tensor from device {self.device!r} to CPU")

    def to(self, device: str | Device) -> "Tensor":
        target = _normalize_device(device)
        if target == self.device:
            return self
        if target == "cpu":
            return self.cpu()
        if target == "metal" and self.device == "cpu":
            if not _core.is_available("metal"):
                raise ValueError("Metal is not available on this system")
            if hasattr(_core, "cpu_to_metal"):
                return Tensor(_core.cpu_to_metal(self._impl))
        raise ValueError(f"unsupported device transfer: {self.device!r} -> {target!r}")

    def __add__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            return NotImplemented
        if self.device != other.device:
            raise ValueError("device mismatch for binary operation")
        return Tensor(_core.add(self._impl, other._impl))

    def __mul__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            return NotImplemented
        if self.device != other.device:
            raise ValueError("device mismatch for binary operation")
        return Tensor(_core.multiply(self._impl, other._impl))


def tensor(data, dtype: str | None = None, device: str | Device | None = None) -> Tensor:
    target = _normalize_device(device)
    cpu_tensor = Tensor(_core.tensor(data, dtype=dtype, device="cpu"))
    return cpu_tensor if target == "cpu" else cpu_tensor.to(target)


def empty(shape, dtype: str = "float32", device: str | Device = "cpu") -> Tensor:
    target = _normalize_device(device)
    cpu_tensor = Tensor(_core.empty(shape, dtype=dtype, device="cpu"))
    return cpu_tensor if target == "cpu" else cpu_tensor.to(target)


def zeros(shape, dtype: str = "float32", device: str | Device = "cpu") -> Tensor:
    target = _normalize_device(device)
    if target == "metal" and hasattr(_core, "fill"):
        if not _core.is_available("metal"):
            raise ValueError("Metal is not available on this system")
        return Tensor(_core.fill(shape, dtype=dtype, value=0.0, device=target))
    cpu_tensor = Tensor(_core.zeros(shape, dtype=dtype, device="cpu"))
    return cpu_tensor if target == "cpu" else cpu_tensor.to(target)


def ones(shape, dtype: str = "float32", device: str | Device = "cpu") -> Tensor:
    target = _normalize_device(device)
    if target == "metal" and hasattr(_core, "fill"):
        if not _core.is_available("metal"):
            raise ValueError("Metal is not available on this system")
        return Tensor(_core.fill(shape, dtype=dtype, value=1.0, device=target))
    cpu_tensor = Tensor(_core.ones(shape, dtype=dtype, device="cpu"))
    return cpu_tensor if target == "cpu" else cpu_tensor.to(target)
