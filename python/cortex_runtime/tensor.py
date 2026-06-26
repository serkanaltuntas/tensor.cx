"""Tensor API wrappers for Cortex Runtime."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

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

    def __matmul__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            return NotImplemented
        return matmul(self, other)

    def sum(self, axis: int) -> "Tensor":
        return sum(self, axis=axis)

    def max(self, axis: int) -> "Tensor":
        return max(self, axis=axis)

    def mean(self, axis: int) -> "Tensor":
        return mean(self, axis=axis)

    def exp(self) -> "Tensor":
        return exp(self)

    def gelu(self) -> "Tensor":
        return gelu(self)

    def silu(self) -> "Tensor":
        return silu(self)

    def softmax(self, axis: int) -> "Tensor":
        return softmax(self, axis=axis)

    def rmsnorm(self, axis: int, eps: float = 1.0e-5) -> "Tensor":
        return rmsnorm(self, axis=axis, eps=eps)

    def layernorm(self, axis: int, eps: float = 1.0e-5) -> "Tensor":
        return layernorm(self, axis=axis, eps=eps)


def _normalize_shape(shape) -> tuple[int, ...]:
    try:
        return (_normalize_shape_dim(shape),)
    except TypeError:
        pass

    try:
        iterator = iter(shape)
    except TypeError:
        raise ValueError("shape must be an int or an iterable of ints") from None

    dims = []
    for dim in iterator:
        try:
            dims.append(_normalize_shape_dim(dim))
        except TypeError:
            raise ValueError("shape dimensions must be integers") from None
    return tuple(dims)


def _normalize_shape_dim(dim) -> int:
    if isinstance(dim, (bool, np.bool_)):
        raise ValueError("shape dimensions must be integers")
    try:
        value = operator.index(dim)
    except TypeError:
        raise TypeError from None
    if value < 0:
        raise ValueError("shape dimensions must be non-negative")
    return int(value)


def _normalize_axis(axis) -> int:
    if isinstance(axis, (bool, np.bool_)):
        raise ValueError("axis must be an integer")
    try:
        value = operator.index(axis)
    except TypeError:
        raise ValueError("axis must be an integer") from None
    if value < -(2**63) or value > 2**63 - 1:
        raise ValueError("axis is out of range")
    return int(value)


def tensor(data, dtype: str | None = None, device: str | Device | None = None) -> Tensor:
    target = _normalize_device(device)
    array = np.asarray(data)
    if array.ndim == 1:
        cpu_tensor = Tensor(_core.tensor(array.reshape(-1).tolist(), dtype=dtype, device="cpu"))
    else:
        # ndim == 0 (scalar) and ndim >= 2 both route through the flat factory so
        # the original shape is preserved -- a scalar stays rank-0 instead of
        # being silently promoted to (1,).
        actual_dtype = dtype
        if actual_dtype is None:
            actual_dtype = "float32" if np.issubdtype(array.dtype, np.floating) else "int32"
        cpu_tensor = Tensor(
            _core.tensor_from_flat(
                array.reshape(-1).tolist(),
                shape=tuple(int(dim) for dim in array.shape),
                dtype=actual_dtype,
                device="cpu",
            )
        )
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


def randn(
    shape,
    dtype: str = "float32",
    device: str | Device = "cpu",
    seed: int | None = None,
) -> Tensor:
    if dtype != "float32":
        raise ValueError("randn only supports float32")
    dims = _normalize_shape(shape)
    values = np.random.default_rng(seed).standard_normal(dims).astype(np.float32)
    return tensor(values, dtype=dtype, device=device)


def matmul(lhs: Tensor, rhs: Tensor, backend: str = "auto") -> Tensor:
    if not isinstance(lhs, Tensor) or not isinstance(rhs, Tensor):
        raise TypeError("matmul expects Tensor arguments")
    if lhs.device != rhs.device:
        raise ValueError("device mismatch for matmul")
    return Tensor(_core.matmul(lhs._impl, rhs._impl, backend=backend))


def sum(input: Tensor, axis: int) -> Tensor:
    if not isinstance(input, Tensor):
        raise TypeError("sum expects a Tensor argument")
    return Tensor(_core.sum(input._impl, axis=_normalize_axis(axis)))


def max(input: Tensor, axis: int) -> Tensor:
    if not isinstance(input, Tensor):
        raise TypeError("max expects a Tensor argument")
    return Tensor(_core.max(input._impl, axis=_normalize_axis(axis)))


def mean(input: Tensor, axis: int) -> Tensor:
    if not isinstance(input, Tensor):
        raise TypeError("mean expects a Tensor argument")
    return Tensor(_core.mean(input._impl, axis=_normalize_axis(axis)))


def exp(input: Tensor) -> Tensor:
    if not isinstance(input, Tensor):
        raise TypeError("exp expects a Tensor argument")
    return Tensor(_core.exp(input._impl))


def gelu(input: Tensor) -> Tensor:
    if not isinstance(input, Tensor):
        raise TypeError("gelu expects a Tensor argument")
    return Tensor(_core.gelu(input._impl))


def silu(input: Tensor) -> Tensor:
    if not isinstance(input, Tensor):
        raise TypeError("silu expects a Tensor argument")
    return Tensor(_core.silu(input._impl))


def softmax(input: Tensor, axis: int) -> Tensor:
    if not isinstance(input, Tensor):
        raise TypeError("softmax expects a Tensor argument")
    return Tensor(_core.softmax(input._impl, axis=_normalize_axis(axis)))


def rmsnorm(input: Tensor, axis: int, eps: float = 1.0e-5) -> Tensor:
    if not isinstance(input, Tensor):
        raise TypeError("rmsnorm expects a Tensor argument")
    return Tensor(_core.rmsnorm(input._impl, axis=_normalize_axis(axis), eps=eps))


def layernorm(input: Tensor, axis: int, eps: float = 1.0e-5) -> Tensor:
    if not isinstance(input, Tensor):
        raise TypeError("layernorm expects a Tensor argument")
    return Tensor(_core.layernorm(input._impl, axis=_normalize_axis(axis), eps=eps))


def matmul_backends(device: str | Device = "cpu") -> list[str]:
    return list(_core.matmul_backends(_normalize_device(device)))
