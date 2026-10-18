"""Real GPU acceptance for the optional, bounded Vulkan/SPIR-V runtime slice."""
from concurrent.futures import ThreadPoolExecutor
import gc
import operator

import numpy as np
import pytest

import tensorcx as cx


_DEVICES = [name for name in cx.devices() if name.partition(":")[0] == "vulkan"]


@pytest.fixture(params=_DEVICES or [pytest.param("vulkan", marks=pytest.mark.skip(reason="Vulkan GPU unavailable"))])
def vulkan_device(request):
    yield request.param
    assert cx._core._vulkan_validation_errors() == 0


@pytest.mark.parametrize("dtype", [np.float32, np.int32, np.bool_])
@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (1,), (3,), (65,), (2, 3, 5)])
def test_vulkan_copy_fill_and_lifetime(vulkan_device, dtype, shape):
    data = (np.arange(int(np.prod(shape))) % 3).astype(dtype).reshape(shape)
    host = cx.tensor(data)
    x = host.to(vulkan_device)
    del host
    gc.collect()
    for _ in range(2):
        np.testing.assert_array_equal(x.numpy(), data)
    for factory in (cx.zeros, cx.ones):
        expected = factory(shape, dtype=x.dtype)
        result = factory(shape, dtype=x.dtype, device=vulkan_device)
        assert result.device == vulkan_device
        assert result.nbytes == expected.nbytes
        np.testing.assert_array_equal(result.numpy(), expected.numpy())
    view = x.reshape((data.size,))
    assert cx._core._shares_storage(x._impl, view._impl)
    del x
    np.testing.assert_array_equal(view.numpy(), data.reshape(-1))


@pytest.mark.parametrize("operation", [operator.add, operator.sub, operator.mul, operator.truediv])
@pytest.mark.parametrize("shapes", [((2, 1, 5), (1, 3, 1)), ((), (65,)), ((2, 0, 1), (3,))])
def test_vulkan_float_arithmetic_broadcast_and_scalars(vulkan_device, operation, shapes):
    rng = np.random.default_rng(824)
    a = cx.tensor(rng.uniform(0.25, 5, shapes[0]).astype(np.float32))
    b = cx.tensor(rng.uniform(0.25, 5, shapes[1]).astype(np.float32))
    x, y = a.to(vulkan_device), b.to(vulkan_device)
    for actual, expected in ((operation(x, y), operation(a, b)),
                             (operation(x, 2.5), operation(a, 2.5)),
                             (operation(2.5, x), operation(2.5, a))):
        cx.testing.assert_allclose(actual.cpu(), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal((-x).numpy(), (-a).numpy())
    np.testing.assert_array_equal(x.numpy(), a.numpy())


@pytest.mark.parametrize("operation", [operator.add, operator.sub, operator.mul])
def test_vulkan_int32_wrap_and_scalars(vulkan_device, operation):
    a = cx.tensor([[np.iinfo(np.int32).min], [np.iinfo(np.int32).max], [-17]], dtype=cx.int32)
    b = cx.tensor([1, 2, -3, np.iinfo(np.int32).max], dtype=cx.int32)
    x, y = a.to(vulkan_device), b.to(vulkan_device)
    for actual, expected in ((operation(x, y), operation(a, b)),
                             (operation(x, -3), operation(a, -3)),
                             (operation(-3, x), operation(-3, a)), (-x, -a)):
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())


def test_vulkan_boolean_padding_and_shape_views(vulkan_device):
    for size in range(10):
        x = cx.ones((1, size, 1), dtype=cx.bool, device=vulkan_device)
        y = cx.expand_dims(cx.squeeze(x, axis=(0, 2)), axis=0)
        assert cx._core._shares_storage(x._impl, y._impl)
        np.testing.assert_array_equal(y.numpy(), np.ones((1, size), dtype=bool))


def test_vulkan_dispatch_uses_second_grid_dimension(vulkan_device):
    # Exceeds the backend's portable 65535-group X cap * local_size(64).
    size = 65_535 * 64 + 7
    a = cx.tensor(np.arange(size, dtype=np.float32) % 4096)
    x = a.to(vulkan_device)
    np.testing.assert_array_equal(((x + 1.0) * 0.5).numpy(), ((a + 1.0) * 0.5).numpy())


def test_vulkan_threaded_reuse_preserves_inputs(vulkan_device):
    data = np.arange(4099, dtype=np.int32) - 2000
    shared = cx.tensor(data, device=vulkan_device)
    def calculate(value):
        for _ in range(4):
            y = (shared + value) * (value + 1)
            np.testing.assert_array_equal(y.numpy(), (data + value) * (value + 1))
        return y.numpy()
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(calculate, range(12)))
    np.testing.assert_array_equal(shared.numpy(), data)


def test_vulkan_unsupported_operations_and_bad_metadata(vulkan_device):
    x = cx.ones((2, 3), device=vulkan_device)
    for action in (lambda: cx.matmul(x, x), lambda: cx.sum(x), lambda: cx.exp(x),
                   lambda: cx.softmax(x, axis=-1), lambda: cx.rmsnorm(x, axis=-1), lambda: cx.layernorm(x, axis=-1)):
        with pytest.raises(ValueError, match="not supported"):
            action()
    with pytest.raises(BufferError, match="CPU and CUDA"):
        x.__dlpack__()
    with pytest.raises(ValueError, match="shape mismatch"):
        _ = x + cx.ones((4,), device=vulkan_device)
    with pytest.raises(ValueError, match="dtype mismatch"):
        _ = x + cx.ones(x.shape, dtype=cx.int32, device=vulkan_device)
    with pytest.raises(ValueError, match="float32"):
        _ = cx.ones((3,), dtype=cx.int32, device=vulkan_device) / 2
    with pytest.raises(ValueError, match="limit"):
        cx.empty((2**32,), device=vulkan_device)
    with pytest.raises(ValueError):
        cx.empty((-1,), device=vulkan_device)
    with pytest.raises(ValueError, match="not available"):
        cx.ones((2,), device="vulkan:99999")


def test_vulkan_capabilities_and_diagnostics(vulkan_device):
    caps = cx.capabilities(vulkan_device)
    assert caps["add"] == ["float32", "int32"]
    assert caps["copy"] == ["float32", "int32", "bool"]
    for name in ("astype", "transpose", "indexing", "concat", "stack", "split"):
        assert caps[name] == ["float32", "int32", "bool"]
    assert not {"matmul", "sum", "from_dlpack", "attention"} & caps.keys()
    info = cx.diagnostics()["backends"][vulkan_device]
    assert info["available"] is True
    assert info["api_version"]
    assert info["memory"]["scope"] == "device_local_heaps"
    assert info["memory"]["available_bytes"] is None
    assert "UUID" not in str(info) and "GPU-" not in str(info)
    assert cx.device(vulkan_device).type == "vulkan"
    if vulkan_device == "vulkan":
        assert cx.device("vulkan:0") == cx.device("vulkan")
    if cx.is_available("cuda"):
        assert cx.best_device() == "cuda"
