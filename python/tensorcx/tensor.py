"""Tensor API wrappers for tensor.cx."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

from . import _core
from . import backend as _backend
from .device import Device, _normalize_device


class Tensor:
    """Public tensor wrapper over backend-specific native tensor objects."""

    # Opt out of NumPy object-ufunc broadcasting; NumPy scalar operators still
    # defer to our reflected arithmetic methods.
    __array_ufunc__ = None

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
        return Tensor(_backend.copy_tensor(self._impl, self.device, "cpu"))

    def to(self, device: str | Device) -> "Tensor":
        target = _normalize_device(device)
        if target == self.device:
            return self
        return Tensor(_backend.copy_tensor(self._impl, self.device, target))

    def __add__(self, other: "Tensor") -> "Tensor":
        return _arithmetic(self, other, "add")

    def __radd__(self, other) -> "Tensor":
        return _arithmetic(self, other, "add", scalar_left=True)

    def __sub__(self, other) -> "Tensor":
        return _arithmetic(self, other, "subtract")

    def __rsub__(self, other) -> "Tensor":
        return _arithmetic(self, other, "subtract", scalar_left=True)

    def __mul__(self, other: "Tensor") -> "Tensor":
        return _arithmetic(self, other, "multiply")

    def __rmul__(self, other) -> "Tensor":
        return _arithmetic(self, other, "multiply", scalar_left=True)

    def __truediv__(self, other) -> "Tensor":
        return _arithmetic(self, other, "divide")

    def __rtruediv__(self, other) -> "Tensor":
        return _arithmetic(self, other, "divide", scalar_left=True)

    def __neg__(self) -> "Tensor":
        return Tensor(_core.negative(self._impl))

    def reshape(self, shape) -> "Tensor":
        return reshape(self, shape)

    def astype(self, dtype: str, *, copy: bool = True) -> "Tensor":
        return astype(self, dtype, copy=copy)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            return NotImplemented
        return matmul(self, other)

    def sum(self, axis: int, keepdims: bool = False) -> "Tensor":
        return sum(self, axis=axis, keepdims=keepdims)

    def max(self, axis: int, keepdims: bool = False) -> "Tensor":
        return max(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: int, keepdims: bool = False) -> "Tensor":
        return mean(self, axis=axis, keepdims=keepdims)

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


def _arithmetic(input: Tensor, other, name: str, *, scalar_left: bool = False):
    if isinstance(other, Tensor):
        if input.device != other.device:
            raise ValueError("device mismatch for binary operation")
        lhs, rhs = (other, input) if scalar_left else (input, other)
        return Tensor(getattr(_core, name)(lhs._impl, rhs._impl))
    if isinstance(other, (bool, np.bool_)):
        raise TypeError("bool arithmetic scalars are not supported")
    if isinstance(other, np.ndarray):
        raise TypeError("NumPy arrays are not arithmetic scalars; use tensor() explicitly")
    if not isinstance(other, (int, float, np.integer, np.floating)):
        return NotImplemented
    if input.dtype == "int32":
        if not isinstance(other, (int, np.integer)):
            raise ValueError("int32 arithmetic requires an integer scalar")
        value = int(other)
        if not -(2**31) <= value < 2**31:
            raise ValueError("integer scalar is out of range for int32")
    else:
        try:
            value = float(other)
        except OverflowError as exc:
            raise ValueError("scalar is out of range for float32 conversion") from exc
    return Tensor(getattr(_core, name + "_scalar")(
        input._impl, value, scalar_left=scalar_left
    ))


def reshape(input: Tensor, shape) -> Tensor:
    """Return a contiguous view sharing storage; one dimension may be -1."""
    if not isinstance(input, Tensor):
        raise TypeError("reshape expects a Tensor argument")
    if shape is None:
        raise ValueError("shape must be an int or an iterable of ints")
    return Tensor(_core.reshape(input._impl, shape))


def astype(input: Tensor, dtype: str, *, copy: bool = True) -> Tensor:
    """Convert explicitly on the same device; float-to-int truncates toward zero.

    Non-finite or out-of-range float-to-int values raise ValueError. By default
    the result owns a new buffer, including same-dtype casts. With copy=False,
    a same-dtype cast returns the input; changing dtype still allocates.
    """
    if not isinstance(input, Tensor):
        raise TypeError("astype expects a Tensor argument")
    if not isinstance(dtype, str) or dtype not in ("float32", "int32"):
        raise ValueError("astype dtype must be 'float32' or 'int32'")
    if not isinstance(copy, (bool, np.bool_)):
        raise TypeError("copy must be a boolean")
    if not copy and input.dtype == dtype:
        return input
    return Tensor(_core.astype(input._impl, dtype))


def _validate_creation_device(target: str) -> None:
    _backend.require_backend(target)


def _contains_bool_data(data) -> bool:
    if isinstance(data, (bool, np.bool_)):
        return True
    if isinstance(data, np.ndarray):
        if data.dtype.kind == "b":
            return True
        if data.dtype.kind == "O":
            return any(_contains_bool_data(item) for item in data.flat)
        return False
    if isinstance(data, (list, tuple)):
        return any(_contains_bool_data(item) for item in data)
    return False


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
    _validate_creation_device(target)
    if _contains_bool_data(data):
        raise ValueError("bool tensor data is not supported")
    array = np.asarray(data)
    # Infer the dtype from the array once, for every rank. Resolving it here
    # (rather than letting the C++ 1-D factory infer from Python element types)
    # keeps an empty float array float32 instead of defaulting to int32 when
    # there are no elements to inspect.
    actual_dtype = dtype
    if actual_dtype is None:
        actual_dtype = "float32" if np.issubdtype(array.dtype, np.floating) else "int32"
    if array.ndim == 1:
        cpu_tensor = Tensor(
            _core.tensor(array.reshape(-1).tolist(), dtype=actual_dtype, device="cpu")
        )
    else:
        # ndim == 0 (scalar) and ndim >= 2 both route through the flat factory so
        # the original shape is preserved -- a scalar stays rank-0 instead of
        # being silently promoted to (1,).
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
    _validate_creation_device(target)
    cpu_tensor = Tensor(_core.empty(shape, dtype=dtype, device="cpu"))
    return cpu_tensor if target == "cpu" else cpu_tensor.to(target)


def zeros(shape, dtype: str = "float32", device: str | Device = "cpu") -> Tensor:
    target = _normalize_device(device)
    _validate_creation_device(target)
    return Tensor(_backend.fill(target, shape, dtype, 0.0))


def ones(shape, dtype: str = "float32", device: str | Device = "cpu") -> Tensor:
    target = _normalize_device(device)
    _validate_creation_device(target)
    return Tensor(_backend.fill(target, shape, dtype, 1.0))


def randn(
    shape,
    dtype: str = "float32",
    device: str | Device = "cpu",
    seed: int | None = None,
) -> Tensor:
    if dtype != "float32":
        raise ValueError("randn only supports float32")
    target = _normalize_device(device)
    _validate_creation_device(target)
    dims = _normalize_shape(shape)
    values = np.random.default_rng(seed).standard_normal(dims).astype(np.float32)
    return tensor(values, dtype=dtype, device=target)


def matmul(lhs: Tensor, rhs: Tensor, backend: str = "auto") -> Tensor:
    if not isinstance(lhs, Tensor) or not isinstance(rhs, Tensor):
        raise TypeError("matmul expects Tensor arguments")
    if lhs.device != rhs.device:
        raise ValueError("device mismatch for matmul")
    return Tensor(_core.matmul(lhs._impl, rhs._impl, backend=backend))


def _reduce(input: Tensor, axis: int, keepdims: bool, name: str) -> Tensor:
    if not isinstance(input, Tensor):
        raise TypeError(f"{name} expects a Tensor argument")
    if not isinstance(keepdims, (bool, np.bool_)):
        raise TypeError("keepdims must be a boolean")
    axis = _normalize_axis(axis)
    result = Tensor(getattr(_core, name)(input._impl, axis=axis))
    if keepdims and input.shape:
        shape = list(input.shape)
        shape[axis] = 1
        return result.reshape(shape)
    return result


def sum(input: Tensor, axis: int, keepdims: bool = False) -> Tensor:
    return _reduce(input, axis, keepdims, "sum")


def max(input: Tensor, axis: int, keepdims: bool = False) -> Tensor:
    return _reduce(input, axis, keepdims, "max")


def mean(input: Tensor, axis: int, keepdims: bool = False) -> Tensor:
    return _reduce(input, axis, keepdims, "mean")


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
    return _backend.matmul_backends(_normalize_device(device))
