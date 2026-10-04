"""Tensor API wrappers for tensor.cx."""

from __future__ import annotations

import builtins
import operator
import math
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

    def __bool__(self) -> bool:
        if math.prod(self.shape) != 1:
            raise ValueError("tensor truth value requires exactly one element; use any() or all()")
        return builtins.bool(self.reshape(()).numpy().item())

    def any(self, axis=None, keepdims=False) -> "Tensor":
        return any(self, axis=axis, keepdims=keepdims)

    def all(self, axis=None, keepdims=False) -> "Tensor":
        return all(self, axis=axis, keepdims=keepdims)

    def where(self, condition, other) -> "Tensor":
        return where(condition, self, other)

    def __eq__(self, other) -> "Tensor":
        return equal(self, other)

    def __ne__(self, other) -> "Tensor":
        return not_equal(self, other)

    def __lt__(self, other) -> "Tensor":
        return less(self, other)

    def __le__(self, other) -> "Tensor":
        return less_equal(self, other)

    def __gt__(self, other) -> "Tensor":
        return greater(self, other)

    def __ge__(self, other) -> "Tensor":
        return greater_equal(self, other)

    def __and__(self, other) -> "Tensor":
        return logical_and(self, other)

    def __or__(self, other) -> "Tensor":
        return logical_or(self, other)

    def __xor__(self, other) -> "Tensor":
        return logical_xor(self, other)

    def __rand__(self, other) -> "Tensor":
        return logical_and(other, self)

    def __ror__(self, other) -> "Tensor":
        return logical_or(other, self)

    def __rxor__(self, other) -> "Tensor":
        return logical_xor(other, self)

    def __invert__(self) -> "Tensor":
        return logical_not(self)

    def __getitem__(self, key) -> "Tensor":
        return _getitem(self, key)

    def split(self, indices_or_sections, axis=0) -> tuple["Tensor", ...]:
        return split(self, indices_or_sections, axis)

    def reshape(self, shape) -> "Tensor":
        return reshape(self, shape)

    def transpose(self, axes=None) -> "Tensor":
        return transpose(self, axes)

    @property
    def T(self) -> "Tensor":
        return transpose(self)

    def squeeze(self, axis=None) -> "Tensor":
        return squeeze(self, axis)

    def expand_dims(self, axis) -> "Tensor":
        return expand_dims(self, axis)

    def astype(self, dtype: str, *, copy: bool = True) -> "Tensor":
        return astype(self, dtype, copy=copy)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            return NotImplemented
        return matmul(self, other)

    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        return sum(self, axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims: bool = False) -> "Tensor":
        return max(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        return mean(self, axis=axis, keepdims=keepdims)

    def log(self) -> "Tensor":
        return log(self)

    def sqrt(self) -> "Tensor":
        return sqrt(self)

    def abs(self) -> "Tensor":
        return abs(self)

    def __abs__(self) -> "Tensor":
        return abs(self)

    def min(self, axis=None, keepdims=False) -> "Tensor":
        return min(self, axis=axis, keepdims=keepdims)

    def argmax(self, axis=None, keepdims=False) -> "Tensor":
        return argmax(self, axis=axis, keepdims=keepdims)

    def clip(self, lower=None, upper=None) -> "Tensor":
        return clip(self, lower, upper)

    def topk(self, k, axis=-1, largest=True, sorted=True):
        return topk(self, k, axis=axis, largest=largest, sorted=sorted)

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


def _axis_sequence(axes) -> tuple[tuple[int, ...], bool]:
    """Return integer axes and whether the argument was one integer."""
    if isinstance(axes, (bool, np.bool_, str, bytes)):
        raise ValueError("axis must be an integer or iterable of integers")
    try:
        value = operator.index(axes)
    except TypeError:
        try:
            values = tuple(axes)
        except TypeError:
            raise ValueError("axis must be an integer or iterable of integers") from None
        return tuple(_normalize_axis(value) for value in values), False
    return (_normalize_axis(value),), True


def _axes_in_rank(axes: tuple[int, ...], rank: int, *, label: str = "axis") -> tuple[int, ...]:
    normalized = tuple(axis + rank if axis < 0 else axis for axis in axes)
    if builtins.any(axis < 0 or axis >= rank for axis in normalized):
        raise ValueError(f"{label} is out of range")
    if len(set(normalized)) != len(normalized):
        raise ValueError("axes must not repeat")
    return normalized


def transpose(input: Tensor, axes=None) -> Tensor:
    """Permute axes into a new contiguous buffer on the same device.

    None reverses all axes; otherwise axes must be a full permutation. This
    operation copies even for an identity permutation or a scalar.
    """
    if not isinstance(input, Tensor):
        raise TypeError("transpose expects a Tensor argument")
    rank = len(input.shape)
    if axes is None:
        permutation = tuple(reversed(range(rank)))
    else:
        values, _ = _axis_sequence(axes)
        if len(values) != rank:
            raise ValueError("transpose axes must be a full permutation")
        permutation = _axes_in_rank(values, rank)
    return Tensor(_core.transpose(input._impl, permutation))


def squeeze(input: Tensor, axis=None) -> Tensor:
    """Remove singleton axes and return a view sharing the input buffer."""
    if not isinstance(input, Tensor):
        raise TypeError("squeeze expects a Tensor argument")
    shape = input.shape
    if axis is None:
        selected = {i for i, size in enumerate(shape) if size == 1}
    else:
        values, single = _axis_sequence(axis)
        # Match NumPy's scalar squeeze(axis=0/-1) convention; an explicit
        # axis sequence must name real axes (the empty tuple is valid).
        if not shape and single and values[0] in (0, -1):
            return reshape(input, ())
        selected = set(_axes_in_rank(values, len(shape)))
        if builtins.any(shape[i] != 1 for i in selected):
            raise ValueError("cannot squeeze an axis whose size is not one")
    return reshape(input, tuple(size for i, size in enumerate(shape) if i not in selected))


def expand_dims(input: Tensor, axis) -> Tensor:
    """Insert singleton axes at positions in the final shape, sharing storage."""
    if not isinstance(input, Tensor):
        raise TypeError("expand_dims expects a Tensor argument")
    values, _ = _axis_sequence(axis)
    rank = len(input.shape) + len(values)
    selected = set(_axes_in_rank(values, rank))
    source = iter(input.shape)
    shape = tuple(1 if i in selected else next(source) for i in range(rank))
    return reshape(input, shape)


def astype(input: Tensor, dtype: str, *, copy: bool = True) -> Tensor:
    """Convert explicitly on the same device; float-to-int truncates toward zero.

    Non-finite or out-of-range float-to-int values raise ValueError. By default
    the result owns a new buffer, including same-dtype casts. With copy=False,
    a same-dtype cast returns the input; changing dtype still allocates.
    """
    if not isinstance(input, Tensor):
        raise TypeError("astype expects a Tensor argument")
    if not isinstance(dtype, str) or dtype not in ("float32", "int32", "bool"):
        raise ValueError("astype dtype must be 'float32', 'int32' or 'bool'")
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
            return builtins.any(_contains_bool_data(item) for item in data.flat)
        return False
    if isinstance(data, (list, tuple)):
        return builtins.any(_contains_bool_data(item) for item in data)
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
    array = np.asarray(data)
    if dtype is None and array.dtype.kind != "b" and _contains_bool_data(data):
        raise ValueError("mixed boolean and numeric data requires an explicit dtype")
    # Infer the dtype from the array once, for every rank. Resolving it here
    # (rather than letting the C++ 1-D factory infer from Python element types)
    # keeps an empty float array float32 instead of defaulting to int32 when
    # there are no elements to inspect.
    actual_dtype = dtype
    if actual_dtype is None:
        actual_dtype = "bool" if array.dtype.kind == "b" else "float32" if np.issubdtype(array.dtype, np.floating) else "int32"
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


def _reduce(input: Tensor, axis, keepdims: bool, name: str) -> Tensor:
    if not isinstance(input, Tensor):
        raise TypeError(f"{name} expects a Tensor argument")
    if not isinstance(keepdims, (bool, np.bool_)):
        raise TypeError("keepdims must be a boolean")
    rank = len(input.shape)
    if axis is None:
        axes = tuple(range(rank))
        scalar_reduction = rank == 0
    else:
        values, single = _axis_sequence(axis)
        scalar_reduction = rank == 0 and single and values[0] in (0, -1)
        axes = () if scalar_reduction else _axes_in_rank(values, rank, label="reduction axis")
    # Preserve the original one-axis path, including scalar 0/-1 semantics.
    if name not in ("any", "all", "min") and (scalar_reduction or len(axes) == 1):
        result = Tensor(getattr(_core, name)(input._impl, axis=axes[0] if axes else 0))
    else:
        result = Tensor(getattr(_core, f"_{name}_axes")(input._impl, axes))
    if keepdims and rank:
        selected = set(axes)
        return result.reshape(tuple(1 if i in selected else size for i, size in enumerate(input.shape)))
    return result


def sum(input: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    """Sum over all axes (None), one axis, or an axis sequence; retain dtype."""
    return _reduce(input, axis, keepdims, "sum")


def max(input: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    """Maximum over selected axes; reducing a zero-size axis is an error."""
    return _reduce(input, axis, keepdims, "max")


def mean(input: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    """Float32 mean over selected axes, dividing once by their element count."""
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


def _index_integer(value, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be an integer, not a boolean")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"{label} must be an integer") from error


def _getitem(x: Tensor, key) -> Tensor:
    if isinstance(key, Tensor):
        return masked_select(x, key)
    if isinstance(key, tuple) and len(key) == 1 and isinstance(key[0], Tensor):
        return masked_select(x, key[0])
    keys = key if isinstance(key, tuple) else (key,)
    ellipses = builtins.sum(item is Ellipsis for item in keys)
    if ellipses > 1:
        raise IndexError("an index can contain only one ellipsis")
    consumed = builtins.sum(item is not None and item is not Ellipsis for item in keys)
    if consumed > len(x.shape):
        raise IndexError("too many indices for tensor")
    missing = len(x.shape) - consumed
    expanded = []
    for item in keys:
        expanded.extend([slice(None)] * missing if item is Ellipsis else [item])
    if not ellipses:
        expanded.extend([slice(None)] * missing)
    starts, steps, lengths, output_shape = [], [], [], []
    axis = 0
    for item in expanded:
        if item is None:
            output_shape.append(1)
            continue
        extent = x.shape[axis]
        axis += 1
        if isinstance(item, slice):
            # Reject masks and non-integral bounds without NumPy's deprecated
            # boolean-as-integer coercion. Python clips arbitrarily large ints.
            bounds = [None if v is None else _index_integer(v, "slice bound")
                      for v in (item.start, item.stop, item.step)]
            start, stop, step = slice(*bounds).indices(extent)
            length = len(range(start, stop, step))
            output_shape.append(length)
            # Unvisited strides need not fit int64 (e.g. x[::10**100]).
            steps.append(step if length > 1 else 1)
        else:
            try:
                start = _index_integer(item, "index")
            except ValueError as error:
                raise IndexError("only integers, slices, ellipsis and None are supported") from error
            if start < 0:
                start += extent
            if start < 0 or start >= extent:
                raise IndexError("tensor index is out of bounds")
            length = 1
            steps.append(1)
        starts.append(start)
        lengths.append(length)
    result = Tensor(_core._slice(x._impl, starts, steps, lengths))
    return reshape(result, tuple(output_shape))


def _tensor_sequence(tensors) -> tuple[Tensor, ...]:
    try:
        inputs = tuple(tensors)
    except TypeError as error:
        raise ValueError("expected an iterable of tensors") from error
    if not inputs or builtins.any(not isinstance(x, Tensor) for x in inputs):
        raise ValueError("expected a non-empty iterable of tensors")
    first = inputs[0]
    if builtins.any(x.dtype != first.dtype or x.device != first.device for x in inputs):
        raise ValueError("all tensors must have the same dtype and device")
    return inputs


def concat(tensors, axis=0) -> Tensor:
    """Join tensors along an existing axis into independent contiguous storage."""
    inputs = _tensor_sequence(tensors)
    axis = _axes_in_rank((_normalize_axis(axis),), len(inputs[0].shape))[0]
    return Tensor(_core._concat(inputs[0]._impl, [x._impl for x in inputs[1:]], axis))


def stack(tensors, axis=0) -> Tensor:
    """Join equal-shaped tensors along a new axis (including scalar inputs)."""
    inputs = _tensor_sequence(tensors)
    if builtins.any(x.shape != inputs[0].shape for x in inputs):
        raise ValueError("stack inputs must have equal shapes")
    axis = _axes_in_rank((_normalize_axis(axis),), len(inputs[0].shape) + 1)[0]
    return concat([expand_dims(x, axis) for x in inputs], axis)


def split(x: Tensor, indices_or_sections, axis=0) -> tuple[Tensor, ...]:
    """NumPy-style equal sections or cut indices; each result owns its storage."""
    if not isinstance(x, Tensor):
        raise ValueError("split requires a tensor")
    axis = _axes_in_rank((_normalize_axis(axis),), len(x.shape))[0]
    extent = x.shape[axis]
    if isinstance(indices_or_sections, (bool, np.bool_)):
        raise ValueError("split sections must be an integer or iterable of integer cuts")
    try:
        sections = operator.index(indices_or_sections)
    except TypeError:
        try:
            cuts = [_index_integer(v, "split index") for v in indices_or_sections]
        except TypeError as error:
            raise ValueError("split sections must be an integer or iterable of integer cuts") from error
    else:
        if sections <= 0 or extent % sections:
            raise ValueError("split sections must be positive and divide the axis equally")
        cuts = [i * (extent // sections) for i in range(1, sections)]
    boundaries = [0, *cuts, extent]
    outputs = []
    for start, stop in zip(boundaries, boundaries[1:]):
        key = [slice(None)] * len(x.shape)
        key[axis] = slice(start, stop)
        outputs.append(x[tuple(key)])
    return tuple(outputs)


def _predicate_scalar(value, reference: Tensor) -> Tensor:
    if reference.dtype == "bool":
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError("a bool tensor requires boolean scalar operands")
    else:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError("numeric tensor operands must be real numeric scalars")
        if reference.dtype == "int32":
            if not isinstance(value, (int, np.integer)) or not -(2**31) <= int(value) < 2**31:
                raise ValueError("int32 operands require integer scalars in the int32 range")
    return tensor(value, dtype=reference.dtype, device=reference.device)


def _predicate_pair(lhs, rhs, name: str) -> Tensor:
    if not isinstance(lhs, Tensor) and not isinstance(rhs, Tensor):
        raise TypeError(f"{name} requires at least one Tensor operand")
    reference = lhs if isinstance(lhs, Tensor) else rhs
    lhs = lhs if isinstance(lhs, Tensor) else _predicate_scalar(lhs, reference)
    rhs = rhs if isinstance(rhs, Tensor) else _predicate_scalar(rhs, reference)
    if lhs.device != rhs.device:
        raise ValueError("operands must be on the same device")
    return Tensor(getattr(_core, name)(lhs._impl, rhs._impl))


def equal(lhs, rhs) -> Tensor:
    return _predicate_pair(lhs, rhs, "equal")


def not_equal(lhs, rhs) -> Tensor:
    return _predicate_pair(lhs, rhs, "not_equal")


def less(lhs, rhs) -> Tensor:
    return _predicate_pair(lhs, rhs, "less")


def less_equal(lhs, rhs) -> Tensor:
    return _predicate_pair(lhs, rhs, "less_equal")


def greater(lhs, rhs) -> Tensor:
    return _predicate_pair(lhs, rhs, "greater")


def greater_equal(lhs, rhs) -> Tensor:
    return _predicate_pair(lhs, rhs, "greater_equal")


def logical_and(lhs, rhs) -> Tensor:
    return _predicate_pair(lhs, rhs, "logical_and")


def logical_or(lhs, rhs) -> Tensor:
    return _predicate_pair(lhs, rhs, "logical_or")


def logical_xor(lhs, rhs) -> Tensor:
    return _predicate_pair(lhs, rhs, "logical_xor")


def logical_not(input: Tensor) -> Tensor:
    if not isinstance(input, Tensor):
        raise TypeError("logical_not requires a Tensor")
    return Tensor(_core.logical_not(input._impl))


def where(condition: Tensor, x, y) -> Tensor:
    """Select broadcast-compatible values using a boolean condition."""
    if not isinstance(condition, Tensor) or condition.dtype != "bool":
        raise TypeError("where condition must be a bool Tensor")
    reference = x if isinstance(x, Tensor) else y if isinstance(y, Tensor) else None
    if reference is not None:
        if reference.device != condition.device:
            raise ValueError("where operands must be on the same device")
        x = x if isinstance(x, Tensor) else _predicate_scalar(x, reference)
        y = y if isinstance(y, Tensor) else _predicate_scalar(y, reference)
    else:
        x = tensor(x, device=condition.device)
        y = tensor(y, device=condition.device)
    if x.device != condition.device or y.device != condition.device:
        raise ValueError("where operands must be on the same device")
    return Tensor(_core.where(condition._impl, x._impl, y._impl))


def any(input: Tensor, axis=None, keepdims=False) -> Tensor:
    """Reduce bool tensors with false as the identity for an empty selection."""
    return _reduce(input, axis, keepdims, "any")


def all(input: Tensor, axis=None, keepdims=False) -> Tensor:
    """Reduce bool tensors with true as the identity for an empty selection."""
    return _reduce(input, axis, keepdims, "all")


def masked_select(input: Tensor, mask: Tensor) -> Tensor:
    """Select matching leading dimensions, preserving trailing dimensions."""
    if not isinstance(input, Tensor) or not isinstance(mask, Tensor):
        raise TypeError("masked_select requires Tensor arguments")
    if input.device != mask.device:
        raise ValueError("mask and input must be on the same device")
    return Tensor(_core.masked_select(input._impl, mask._impl))


def _math_unary(input: Tensor, name: str) -> Tensor:
    if not isinstance(input, Tensor):
        raise TypeError(f"{name} expects a Tensor argument")
    return Tensor(getattr(_core, name)(input._impl))


def log(input: Tensor) -> Tensor:
    """Natural logarithm of float32 values, with IEEE NaN/inf domain results."""
    return _math_unary(input, "log")


def sqrt(input: Tensor) -> Tensor:
    """Float32 square root; negative nonzero values produce NaN."""
    return _math_unary(input, "sqrt")


def abs(input: Tensor) -> Tensor:
    """Elementwise magnitude; int32 minimum wraps to itself."""
    return _math_unary(input, "abs")


def min(input: Tensor, axis=None, keepdims=False) -> Tensor:
    """Minimum over selected axes, propagating NaN; empty reductions fail."""
    return _reduce(input, axis, keepdims, "min")


def argmax(input: Tensor, axis=None, keepdims=False) -> Tensor:
    """First maximum/NaN index as int32; None searches the flattened tensor."""
    if not isinstance(input, Tensor):
        raise TypeError("argmax expects a Tensor argument")
    if not isinstance(keepdims, (bool, np.bool_)):
        raise TypeError("keepdims must be a boolean")
    rank = len(input.shape)
    if axis is None:
        axes = tuple(range(rank))
    else:
        value = _normalize_axis(axis)
        axes = () if rank == 0 and value in (0, -1) else _axes_in_rank((value,), rank)
    result = Tensor(_core._argmax_axes(input._impl, axes))
    if keepdims and rank:
        selected = set(axes)
        result = result.reshape(tuple(1 if i in selected else n for i, n in enumerate(input.shape)))
    return result


def clip(input: Tensor, lower=None, upper=None) -> Tensor:
    """Clip to broadcasted same-dtype bounds; None leaves that side unbounded."""
    if not isinstance(input, Tensor):
        raise TypeError("clip expects a Tensor argument")
    if lower is None and upper is None:
        raise ValueError("clip requires at least one bound")
    if input.dtype not in ("float32", "int32"):
        raise ValueError("clip requires float32 or int32 tensors")
    bounds = []
    for index, value in enumerate((lower, upper)):
        if value is None:
            value = (-math.inf if index == 0 else math.inf) if input.dtype == "float32" else (-(2**31) if index == 0 else 2**31 - 1)
        bound = value if isinstance(value, Tensor) else _predicate_scalar(value, input)
        if bound.device != input.device:
            raise ValueError("device mismatch for clip")
        if bound.dtype != input.dtype:
            raise ValueError("clip input dtypes must match")
        bounds.append(bound)
    return Tensor(_core._clip(input._impl, bounds[0]._impl, bounds[1]._impl))


def topk(input: Tensor, k, axis=-1, largest=True, sorted=True) -> tuple[Tensor, Tensor]:
    """Return (values, int32 indices), with first-index ties and NaNs greatest.

    sorted=False returns the selected entries in original axis order.
    """
    if not isinstance(input, Tensor):
        raise TypeError("topk expects a Tensor argument")
    if isinstance(k, (bool, np.bool_)):
        raise TypeError("topk k must be an integer")
    try:
        k = operator.index(k)
    except TypeError:
        raise TypeError("topk k must be an integer") from None
    if not isinstance(largest, (bool, np.bool_)) or not isinstance(sorted, (bool, np.bool_)):
        raise TypeError("largest and sorted must be booleans")
    values, indices = _core._topk(input._impl, k, _normalize_axis(axis), builtins.bool(largest), builtins.bool(sorted))
    return Tensor(values), Tensor(indices)
