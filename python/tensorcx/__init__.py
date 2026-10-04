"""Python entrypoint for tensor.cx."""

from __future__ import annotations

from . import _core, experimental, testing
from .device import Device, best_device, device, device_name, devices, is_available
from .tensor import (
    Tensor,
    empty,
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
    sum,
    tensor,
    zeros,
)

__version__ = _core.version()
float32 = _core.float32
int32 = _core.int32


def version() -> str:
    """Return the tensor.cx package version."""
    return _core.version()


__all__ = [
    "Tensor",
    "__version__",
    "Device",
    "best_device",
    "device",
    "device_name",
    "devices",
    "empty",
    "exp",
    "experimental",
    "float32",
    "gelu",
    "int32",
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
    "sum",
    "tensor",
    "testing",
    "version",
    "zeros",
]
