"""Vulkan numeric reductions: exact CPU ordering and device-resident inputs."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import tensorcx as cx

_DEVICES = [name for name in cx.devices() if name.partition(":")[0] == "vulkan"]
_NAMES = ("sum", "max", "mean", "min", "argmax")


@pytest.fixture(params=_DEVICES or [pytest.param("vulkan", marks=pytest.mark.skip(reason="Vulkan GPU unavailable"))])
def vulkan_device(request):
    yield request.param
    assert cx._core._vulkan_validation_errors() == 0


def assert_cpu_bits(actual, expected, *, preserve_nan=False):
    assert actual.device.partition(":")[0] == "vulkan"
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    # NumPy has its own rank limit; the runtime permits higher-rank metadata.
    a, b = actual.reshape((-1,)).numpy(), expected.reshape((-1,)).numpy()
    if actual.dtype == cx.int32:
        np.testing.assert_array_equal(a, b)
    else:
        nan = np.isnan(b)
        np.testing.assert_array_equal(np.isnan(a), nan)
        np.testing.assert_array_equal(a.view(np.uint32)[~nan], b.view(np.uint32)[~nan])
        if preserve_nan:
            np.testing.assert_array_equal(a.view(np.uint32)[nan], b.view(np.uint32)[nan])
        else:
            assert np.all(a.view(np.uint32)[nan] == np.uint32(0x7fc00000))


@pytest.mark.parametrize("name", ("sum", "max", "mean", "min"))
@pytest.mark.parametrize("dtype", (cx.float32, cx.int32))
@pytest.mark.parametrize("shape,axis", [
    ((2, 3, 5), None), ((2, 3, 5), ()), ((2, 3, 5), (2, 0)),
    ((2, 3, 5), (1,)), ((2, 3, 5), -1), ((1, 3, 1), (0, 2)),
    ((2, 0, 3), None), ((2, 0, 3), (0, 2)), ((2, 0, 3), (1,)),
    ((2, 0, 3), ()), ((0, 0), 0), ((0,), None),
    ((), None), ((), ()), ((), 0), ((), -1),
])
def test_vulkan_reduction_cpu_parity(vulkan_device, name, dtype, shape, axis):
    values = np.resize(np.array([-9, 5, 1, -3, 17], dtype=dtype), shape)
    host = cx.tensor(values)
    x = host.to(vulkan_device)
    for keepdims in (False, True):
        try:
            expected = getattr(host, name)(axis, keepdims=keepdims)
        except ValueError:
            with pytest.raises(ValueError):
                getattr(x, name)(axis, keepdims=keepdims)
            continue
        actual = getattr(cx, name)(x, axis, keepdims=keepdims)
        assert_cpu_bits(actual, expected, preserve_nan=name in ("max", "min") or axis == ())
        assert not cx._core._shares_storage(x._impl, actual._impl)
    np.testing.assert_array_equal(x.numpy(), values)


@pytest.mark.parametrize("dtype", (cx.float32, cx.int32))
@pytest.mark.parametrize("shape,axis", [((2, 3, 5), None), ((2, 3, 5), 0),
    ((2, 3, 5), 1), ((2, 3, 5), -1), ((0, 3), 1), ((2, 0), 1),
    ((0,), None), ((), None), ((), 0), ((), -1)])
def test_vulkan_argmax_first_index(vulkan_device, dtype, shape, axis):
    values = np.resize(np.array([9, 9, -8, 3, 9, 0, 1], dtype=dtype), shape)
    host, x = cx.tensor(values), cx.tensor(values, device=vulkan_device)
    for keepdims in (False, True):
        try:
            expected = host.argmax(axis, keepdims)
        except ValueError:
            with pytest.raises(ValueError):
                x.argmax(axis, keepdims)
        else:
            assert_cpu_bits(x.argmax(axis, keepdims), expected)


@pytest.mark.parametrize("name", _NAMES)
def test_vulkan_reduction_special_bits(vulkan_device, name):
    bits = np.array([
        [0x80000000, 0, 0x80000000, 0],
        [0, 0x80000000, 0, 0x80000000],
        [1, 2, 3, 0x007fffff],
        [0x80000001, 0x80000003, 0x807fffff, 0x80800000],
        [0x7fc12345, 0xffc23456, 0x3f800000, 0x7f800000],
        [0x3f800000, 0xff812345, 0x7f854321, 0x7fc12345],
        [0x7f800000, 0xff800000, 0x7f7fffff, 1],
        [0x4b800000, 0x3f800000, 0xcb800000, 0x3f800000],
    ], dtype=np.uint32)
    # Avoid Python scalar conversion quieting signaling NaNs before upload.
    host = cx.from_dlpack(bits.view(np.float32), copy=False)
    x = host.to(vulkan_device)
    for axis in (0, 1, None):
        assert_cpu_bits(getattr(x, name)(axis), getattr(host, name)(axis),
                        preserve_nan=name in ("max", "min"))
    if name != "argmax":
        for axis in ((0, 1), (1, 0), ()):
            assert_cpu_bits(getattr(x, name)(axis), getattr(host, name)(axis),
                            preserve_nan=name in ("max", "min") or axis == ())
    np.testing.assert_array_equal(x.numpy().view(np.uint32), bits)


@pytest.mark.parametrize("name", _NAMES)
def test_vulkan_reduction_random_bits(vulkan_device, name):
    rng = np.random.default_rng(1729)
    for dtype in (cx.float32, cx.int32):
        words = rng.integers(0, 2**32, size=(7, 11, 13), dtype=np.uint32)
        values = words.view(dtype)
        host = cx.from_dlpack(values, copy=False)
        x = host.to(vulkan_device)
        if dtype == cx.int32 and name == "mean":
            continue
        axes = (0, 1, 2, None) if name == "argmax" else (0, 1, 2, None, (2, 0), ())
        for axis in axes:
            assert_cpu_bits(getattr(x, name)(axis), getattr(host, name)(axis),
                            preserve_nan=name in ("max", "min") or axis == ())
        np.testing.assert_array_equal(x.numpy().view(np.uint32), words)


@pytest.mark.parametrize("name", _NAMES)
def test_vulkan_reduction_empty_and_axis_validation(vulkan_device, name):
    for shape in ((2, 3), (2, 0)):
        x = cx.empty(shape, device=vulkan_device)
        for axis in (2, -3, True, 0.5, (0, 0), (0, -2)):
            with pytest.raises((ValueError, TypeError)):
                getattr(x, name)(axis)
        for axes in ((0, 0), (0, -2), (2,), (-3,)):
            with pytest.raises(ValueError):
                getattr(cx._core, f"_{name}_axes")(x._impl, axes)
        with pytest.raises(TypeError, match="keepdims"):
            getattr(x, name)(0, keepdims=1)
    scalar = cx.tensor(-0., device=vulkan_device)
    for axis in (1, -2):
        with pytest.raises(ValueError):
            getattr(cx._core, name)(scalar._impl, axis)
    with pytest.raises(ValueError):
        getattr(cx, name)(cx.empty((0,), dtype=cx.bool, device=vulkan_device))
    if name == "mean":
        for axis in (None, (), 0):
            with pytest.raises(ValueError, match="float32"):
                cx.empty((0,), dtype=cx.int32, device=vulkan_device).mean(axis)


def test_vulkan_reduction_high_rank_and_unvisited_metadata(vulkan_device):
    host = cx.tensor(np.arange(6, dtype=np.float32)).reshape((2,) + (1,) * 298 + (3,))
    x = host.to(vulkan_device)
    for name in _NAMES:
        axis = None if name == "argmax" else (299, 0)
        expected = getattr(host, name)(axis, keepdims=True)
        assert_cpu_bits(getattr(x, name)(axis, keepdims=True), expected)
    for shape in ((0, 1, 2**62, 4, 0), (0, 2**32)):
        empty = cx.empty(shape, device=vulkan_device)
        for name in ("sum", "mean", "max", "min"):
            axes = tuple(range(1, len(shape) - (len(shape) > 2)))
            assert getattr(empty, name)(axes).nbytes == 0
        with pytest.raises(ValueError, match="int32"):
            cx.empty((0, 2**31), device=vulkan_device).argmax(1)


def test_vulkan_reduction_second_dispatch_dimension(vulkan_device):
    # More than 65,535 workgroups, so real output execution traverses grid Y.
    values = (np.arange(4_194_257, dtype=np.int32) % 257).astype(np.float32)[:, None]
    x = cx.tensor(values, device=vulkan_device)
    for name in _NAMES:
        result = getattr(x, name)(1)
        assert result.device == vulkan_device and result.shape == (len(values),)
        expected = np.zeros(len(values), np.int32) if name == "argmax" else values[:, 0]
        np.testing.assert_array_equal(result.numpy(), expected)


@pytest.mark.parametrize("dtype", (cx.float32, cx.int32))
def test_vulkan_reduction_long_axis(vulkan_device, dtype):
    values = np.ones(1_048_579, dtype=dtype)
    if dtype == cx.int32:
        values[::3] = np.iinfo(np.int32).max
    host, x = cx.tensor(values), cx.tensor(values, device=vulkan_device)
    for name in _NAMES:
        if name == "mean" and dtype == cx.int32:
            continue
        assert_cpu_bits(getattr(x, name)(), getattr(host, name)())


def test_vulkan_reduction_threads_and_no_host_export(vulkan_device, monkeypatch):
    values = np.arange(105, dtype=np.float32).reshape(3, 5, 7)
    host, x = cx.tensor(values), cx.tensor(values, device=vulkan_device)
    def calculate(index):
        name = _NAMES[index % len(_NAMES)]
        return name, getattr(x, name)(1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(calculate, range(15)))
    for name, actual in results:
        assert_cpu_bits(actual, getattr(host, name)(1))
    def forbidden(*args, **kwargs):
        raise AssertionError("reduction unexpectedly exported tensor data")
    with monkeypatch.context() as patch:
        patch.setattr(cx.Tensor, "cpu", forbidden)
        patch.setattr(cx.Tensor, "numpy", forbidden)
        centered = x - x.mean((0, 2), keepdims=True)
        minimum = centered.min((0, 2))
        maximum_index = centered.argmax(2)
        total = centered.sum((0, 2))
    reference = host - host.mean((0, 2), keepdims=True)
    assert_cpu_bits(minimum, reference.min((0, 2)))
    assert_cpu_bits(maximum_index, reference.argmax(2))
    assert_cpu_bits(total, reference.sum((0, 2)))
    np.testing.assert_array_equal(x.numpy(), values)
