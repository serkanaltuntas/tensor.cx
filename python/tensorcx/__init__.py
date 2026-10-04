"""Python entrypoint for tensor.cx."""

from __future__ import annotations

from . import _core, experimental, testing
from .dlpack import from_dlpack
from .device import Device, best_device, device, device_name, devices, is_available
from .tensor import (
    Tensor,
    log, sqrt, abs, min, argmax, clip, topk,
    linear, embedding, scaled_dot_product_attention, attention,
    masked_select,
    all,
    any,
    where,
    logical_not,
    logical_xor,
    logical_or,
    logical_and,
    greater_equal,
    greater,
    less_equal,
    less,
    not_equal,
    equal,
    astype,
    split,
    stack,
    concat,
    empty,
    expand_dims,
    exp,
    gelu,
    layernorm,
    matmul,
    matmul_backends,
    max,
    mean,
    ones,
    randn,
    reshape,
    rmsnorm,
    silu,
    softmax,
    squeeze,
    sum,
    tensor,
    transpose,
    zeros,
)

__version__ = _core.version()
float32 = _core.float32
int32 = _core.int32
bool = _core.bool


def version() -> str:
    """Return the tensor.cx package version."""
    return _core.version()


__all__ = [
    "Tensor", "from_dlpack",
    "log", "sqrt", "abs", "min", "argmax", "clip", "topk",
    "linear", "embedding", "scaled_dot_product_attention", "attention",
    "masked_select",
    "all",
    "any",
    "where",
    "logical_not",
    "logical_xor",
    "logical_or",
    "logical_and",
    "greater_equal",
    "greater",
    "less_equal",
    "less",
    "not_equal",
    "equal",
    "__version__",
    "Device",
    "astype",
    "split",
    "stack",
    "concat",
    "best_device",
    "device",
    "device_name",
    "devices",
    "empty",
    "expand_dims",
    "exp",
    "experimental",
    "float32",
    "gelu",
    "int32",
    "bool",
    "is_available",
    "layernorm",
    "matmul",
    "matmul_backends",
    "max",
    "mean",
    "ones",
    "randn",
    "reshape",
    "rmsnorm",
    "silu",
    "softmax",
    "squeeze",
    "sum",
    "tensor",
    "transpose",
    "testing",
    "version",
    "zeros",
]
