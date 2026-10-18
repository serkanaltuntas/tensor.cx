"""Real Vulkan comparison/mask acceptance, including packed words and scan tails."""
from concurrent.futures import ThreadPoolExecutor
import operator

import numpy as np
import pytest

import tensorcx as cx

_DEVICES = [name for name in cx.devices() if name.partition(":")[0] == "vulkan"]
_COMPARE = [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge]


@pytest.fixture(params=_DEVICES or [pytest.param("vulkan", marks=pytest.mark.skip(reason="Vulkan GPU unavailable"))])
def vulkan_device(request):
    yield request.param
    assert cx._core._vulkan_validation_errors() == 0


def assert_bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    a, b = actual.numpy(), expected.numpy()
    if actual.dtype == cx.float32:
        a, b = a.view(np.uint32), b.view(np.uint32)
    elif actual.dtype == cx.bool:
        a, b = a.view(np.uint8), b.view(np.uint8)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("operation", _COMPARE)
def test_vulkan_random_float_comparisons_preserve_ieee_order(vulkan_device, operation):
    rng = np.random.default_rng(716)
    bits = rng.integers(0, 2**32, size=(2, 4099), dtype=np.uint32)
    a, b = bits.view(np.float32)
    x, y = [cx.tensor(value, device=vulkan_device) for value in (a, b)]
    with np.errstate(invalid="ignore"):
        expected = operation(a, b)
    np.testing.assert_array_equal(operation(x, y).numpy(), expected)


def test_vulkan_noncanonical_bool_predicates(vulkan_device):
    raw = np.array([0, 2, 255, 1, 128, 0, 255], dtype=np.uint8)
    host = cx.from_dlpack(raw.view(np.bool_), copy=False)
    x = host.to(vulkan_device)
    right = cx.tensor([True, False, True, True, False, True, True])
    y = right.to(vulkan_device)
    for operation in (*_COMPARE, operator.and_, operator.or_, operator.xor):
        assert_bits(operation(x, y), operation(host, right))
    assert_bits(~x, ~host)
    for name in ("any", "all"):
        for axes in (None, (), -1):
            assert_bits(getattr(x, name)(axes), getattr(host, name)(axes))
    for result, expected in ((cx.where(y, x, x[::-1]), cx.where(right, host, host[::-1])),
                             (x[y], host[right]), (x[x], host[host])):
        assert_bits(result, expected)


@pytest.mark.parametrize("dtype", [cx.float32, cx.int32, cx.bool])
@pytest.mark.parametrize("count", [0, 1, 3, 4, 5, 63, 64, 65, 127, 257, 4099, 65537])
def test_vulkan_mask_scan_prefix_boundaries(vulkan_device, dtype, count):
    rng = np.random.default_rng(count)
    data = rng.integers(-5, 6, size=(count, 3), dtype=np.int32).astype(dtype)
    if dtype == cx.float32 and count:
        data.view(np.uint32).reshape(-1)[:3] = [0x80000000, 0x7fc12345, 1]
    host = cx.tensor(data)
    x = host.to(vulkan_device)
    for flags in (np.zeros(count, dtype=bool), np.ones(count, dtype=bool), rng.random(count) > 0.7):
        mask = cx.tensor(flags)
        m = mask.to(vulkan_device)
        assert_bits(x[m], host[mask])
        np.testing.assert_array_equal(m.numpy(), flags)
    assert_bits(x, host)


def test_vulkan_mask_scan_large_sparse_input(vulkan_device):
    size = 1_048_577
    data = np.arange(size, dtype=np.int32)
    flags = np.zeros(size, dtype=bool)
    flags[[0, 3, 64, 65535, size - 2, size - 1]] = True
    actual = cx.tensor(data, device=vulkan_device)[cx.tensor(flags, device=vulkan_device)]
    np.testing.assert_array_equal(actual.numpy(), data[flags])


def test_vulkan_packed_predicates_use_second_dispatch_dimension(vulkan_device):
    size = (65_535 * 64 + 1) * 4 + 3
    data = np.arange(size, dtype=np.int32) % 3
    x = cx.tensor(data, device=vulkan_device)
    m = x > 1
    expected = data > 1
    np.testing.assert_array_equal(m.numpy(), expected)
    np.testing.assert_array_equal((~m).numpy(), ~expected)
    np.testing.assert_array_equal(cx.where(m, False, True).numpy(), ~expected)
    # A scalar leading mask selects the entire input block and exercises the
    # gather's multi-dimensional dispatch without a huge mask-prefix buffer.
    selected = x[cx.tensor(True, device=vulkan_device)]
    np.testing.assert_array_equal(selected.numpy(), data[None, :])


def test_vulkan_mask_pipeline_stays_on_device(vulkan_device, monkeypatch):
    data = np.arange(65 * 3, dtype=np.float32).reshape(65, 3) - 100
    host = cx.tensor(data)
    x = host.to(vulkan_device)
    def forbidden(*args, **kwargs):
        pytest.fail("unexpected tensor export or CPU fallback")
    with monkeypatch.context() as patch:
        patch.setattr(cx.Tensor, "numpy", forbidden)
        patch.setattr(cx.Tensor, "cpu", forbidden)
        rows = ((x > 0) & (x < 50)).any(axis=1)
        y = cx.where(x > 0, x, -x)[rows]
        result = cx.concat(cx.split(y, [1], axis=0), axis=0).T
    rows = ((host > 0) & (host < 50)).any(axis=1)
    expected = cx.concat(cx.split(cx.where(host > 0, host, -host)[rows], [1], axis=0), axis=0).T
    assert_bits(result, expected)


def test_vulkan_predicate_threaded_reuse(vulkan_device):
    data = np.arange(257, dtype=np.int32)
    x = cx.tensor(data, device=vulkan_device)
    def work(boundary):
        mask = (x >= boundary) & (x < boundary + 65)
        np.testing.assert_array_equal(x[mask].numpy(), data[(data >= boundary) & (data < boundary + 65)])
        assert bool(mask.any()) and not bool(mask.all())
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(work, range(0, 128, 8)))
    np.testing.assert_array_equal(x.numpy(), data)
