"""Prototype scope, discovery, and unavailable-device behavior."""

import os
import subprocess
import sys

import pytest

import cortex_runtime as cx
from cortex_runtime import backend


def test_cuda_hidden_device_import_still_has_cpu():
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    subprocess.run(
        [sys.executable, "-c", """
import cortex_runtime as cx
assert not cx.is_available('cuda')
assert 'cuda' not in cx.devices()
assert 'cpu' in cx.devices()
assert (cx.ones((3,)) + cx.ones((3,))).numpy().tolist() == [2, 2, 2]
try:
    cx.ones((1,), device='cuda')
except ValueError as error:
    assert 'device is not available: cuda' in str(error)
else:
    raise AssertionError('unavailable CUDA accepted')
"""],
        env=environment, check=True, capture_output=True, text=True,
    )


@pytest.fixture
def cuda():
    if not cx.is_available("cuda"):
        pytest.skip("CUDA backend is not available")
    return "cuda"


def test_cuda_registry_discovery(cuda):
    assert cuda in backend.backend_names()
    assert cuda in cx._core.devices()
    assert cx.device_name(cuda)
    assert cx.device(cuda).name == cx._core.device_name(cuda)
    assert cx.ones((1,), device="cuda:0").device == cuda
    with pytest.raises(ValueError, match="index 0"):
        cx.ones((1,), device="cuda:1")


def test_cuda_int32_copy_but_no_arithmetic(cuda):
    a = cx.tensor([1, -2147483648, 2147483647], dtype=cx.int32, device=cuda)
    assert a.numpy().tolist() == [1, -2147483648, 2147483647]
    for operation in (lambda: a + a, lambda: a * a,
                      lambda: cx.zeros((1,), dtype=cx.int32, device=cuda),
                      lambda: cx.ones((0,), dtype=cx.int32, device=cuda)):
        with pytest.raises(ValueError, match="only support.*float32"):
            operation()


@pytest.mark.parametrize("operation", [
    lambda x: cx.exp(x), lambda x: cx.gelu(x), lambda x: cx.silu(x),
    lambda x: cx.sum(x, axis=0), lambda x: cx.max(x, axis=0),
    lambda x: cx.mean(x, axis=0), lambda x: cx.softmax(x, axis=0),
    lambda x: cx.rmsnorm(x, axis=0), lambda x: cx.layernorm(x, axis=0),
    lambda x: cx.matmul(x, x), lambda x: cx.matmul_backends(x.device),
])
def test_cuda_float32_operations_are_available(cuda, operation):
    x = cx.ones((2, 2), device=cuda)
    assert operation(x) is not None


@pytest.mark.parametrize("shape", [(-1,), (True,), (1.5,), (2**62,)])
def test_cuda_rejects_invalid_or_overflowing_fill_shape(cuda, shape):
    with pytest.raises(ValueError):
        cx.ones(shape, device=cuda)
