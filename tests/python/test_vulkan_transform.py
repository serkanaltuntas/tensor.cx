"""Exact CPU parity for Vulkan contiguous transformations, including bool packing."""
from concurrent.futures import ThreadPoolExecutor
import gc

import numpy as np
import pytest

import tensorcx as cx

_DEVICES = [name for name in cx.devices() if name.partition(":")[0] == "vulkan"]


@pytest.fixture(params=_DEVICES or [pytest.param("vulkan", marks=pytest.mark.skip(reason="Vulkan GPU unavailable"))])
def vulkan_device(request):
    yield request.param
    assert cx._core._vulkan_validation_errors() == 0


def assert_exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    a, b = actual.numpy(), expected.numpy()
    if actual.dtype == cx.float32:
        a, b = a.view(np.uint32), b.view(np.uint32)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("source", [cx.float32, cx.int32, cx.bool])
@pytest.mark.parametrize("target", [cx.float32, cx.int32, cx.bool])
@pytest.mark.parametrize("shape", [(), (0,), (3, 0, 5), (1,), (3,), (5,), (65,), (3, 5, 7)])
def test_vulkan_all_cast_pairs(vulkan_device, source, target, shape):
    values = np.resize(np.array([-17, -1, 0, 1, 23], dtype=np.int32), shape).astype(source)
    if source == cx.float32:
        values += np.float32(0.75)
    host = cx.tensor(values)
    x = host.to(vulkan_device)
    result = x.astype(target)
    assert result.device == vulkan_device
    assert not cx._core._shares_storage(x._impl, result._impl)
    assert_exact(result, host.astype(target))
    assert_exact(x, host)
    del x
    gc.collect()
    assert_exact(result, host.astype(target))


def test_vulkan_float_bool_preserves_subnormal_truth(vulkan_device):
    bits = np.array([0, 0x80000000, 1, 0x80000001, 0x7fc12345, 0x7f800000,
                     0xff800000, 0x3f800000, 0xbf800000], dtype=np.uint32)
    host = cx.tensor(bits.view(np.float32))
    x = host.to(vulkan_device)
    assert_exact(x.astype(cx.bool), host.astype(cx.bool))
    assert_exact(x.astype(cx.float32), host)


def test_vulkan_noncanonical_bool_bytes_from_dlpack(vulkan_device):
    # DLPack producers can expose bool bytes other than 0/1. Numeric casts must
    # normalize their truth values, whereas same-dtype movement preserves bytes.
    values = np.resize(np.array([0, 2, 255, 1, 128], dtype=np.uint8), (3, 5)).view(np.bool_)
    host = cx.from_dlpack(values, copy=False)
    x = host.to(vulkan_device)
    for dtype in (cx.float32, cx.int32):
        assert_exact(x.astype(dtype), host.astype(dtype))
    for actual, expected in ((x.astype(cx.bool), host.astype(cx.bool)),
                             (x.T, host.T), (x[:, ::-1], host[:, ::-1]),
                             (cx.concat([x, x], axis=-1), cx.concat([host, host], axis=-1))):
        np.testing.assert_array_equal(actual.numpy().view(np.uint8), expected.numpy().view(np.uint8))


@pytest.mark.parametrize("shape,key", [
    ((3, 5, 7), (slice(None, None, -1), slice(None, None, -2), slice(1, 7, 3))),
    ((5, 7), (None, Ellipsis, -2)), ((3, 5), (slice(2, 1),)),
    ((2, 0, 3), (1, Ellipsis, None)), ((), (None, Ellipsis, None)),
    ((65,), (slice(None, None, -1),)), ((3,), (slice(None, None, 10**100),)),
])
def test_vulkan_bool_reorder_and_slice(vulkan_device, shape, key):
    values = (np.arange(np.prod(shape, dtype=int)).reshape(shape) % 3 == 1)
    host = cx.tensor(values)
    x = host.to(vulkan_device)
    for result, expected in ((x.T, host.T), (x[key], host[key])):
        assert_exact(result, expected)
        assert not cx._core._shares_storage(x._impl, result._impl)
    assert_exact(x, host)


@pytest.mark.parametrize("dtype", [cx.float32, cx.int32, cx.bool])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("sizes", [(1, 2, 3), (0, 1, 0, 2, 0), (3,), (0, 0), (1,) * 9])
def test_vulkan_concat_word_boundaries(vulkan_device, dtype, axis, sizes):
    inputs = []
    for i, size in enumerate(sizes):
        shape = [3, 5, 7]
        shape[axis] = size
        values = ((np.arange(np.prod(shape, dtype=int)) + i) % 7).reshape(shape).astype(dtype)
        inputs.append(cx.tensor(values))
    tensors = [x.to(vulkan_device) for x in inputs]
    actual = cx.concat(tensors, axis=axis)
    expected = cx.concat(inputs, axis=axis)
    assert actual.device == vulkan_device
    assert_exact(actual, expected)
    for x, host in zip(tensors, inputs):
        assert_exact(x, host)
        assert not cx._core._shares_storage(x._impl, actual._impl)
    del tensors
    gc.collect()
    assert_exact(actual, expected)


@pytest.mark.parametrize("size", range(10))
def test_vulkan_bool_stack_split_compose_on_device(vulkan_device, size, monkeypatch):
    values = np.arange(3 * size).reshape(3, size) % 3 == 1
    host = cx.tensor(values)
    x = host.to(vulkan_device)
    # Block public host exports while the workload is constructed. The bool
    # transpose/slice/concat path must execute on Vulkan, not through NumPy.
    with monkeypatch.context() as patch:
        patch.setattr(cx.Tensor, "numpy", lambda self: pytest.fail("unexpected host export"))
        patch.setattr(cx.Tensor, "cpu", lambda self: pytest.fail("unexpected CPU fallback"))
        parts = cx.split(x[:, ::-1].T, [1, 1], axis=-1)
        result = cx.stack([cx.concat(parts, axis=-1)] * 3, axis=1).astype(cx.int32)
    parts = cx.split(host[:, ::-1].T, [1, 1], axis=-1)
    expected = cx.stack([cx.concat(parts, axis=-1)] * 3, axis=1).astype(cx.int32)
    assert_exact(result, expected)


def test_vulkan_transform_two_dimensional_dispatch(vulkan_device):
    # Full-word writes for reordering and conversion also use the Y grid.
    size = 65_535 * 64 + 7
    values = np.arange(size, dtype=np.int32) % 5
    host = cx.tensor(values)
    x = host.to(vulkan_device)
    assert_exact(x[::-1], host[::-1])
    assert_exact(x.astype(cx.float32), host.astype(cx.float32))


def test_vulkan_transform_concurrent_error_flags_are_private(vulkan_device):
    good = cx.tensor(np.arange(257, dtype=np.float32) - 129, device=vulkan_device)
    bad = cx.tensor([1., np.nan, 2.], device=vulkan_device)
    def work(i):
        if i % 2:
            with pytest.raises(ValueError, match="out of range"):
                bad.astype(cx.int32)
        else:
            y = cx.concat([good[::-1].astype(cx.int32)] * 3)
            np.testing.assert_array_equal(y.numpy(), np.tile(good.numpy()[::-1].astype(np.int32), 3))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(work, range(24)))
