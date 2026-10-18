"""Vulkan math numerical oracles, stored-bit selection and atomic results."""
from concurrent.futures import ThreadPoolExecutor
import math

import numpy as np
import pytest

import tensorcx as cx

_DEVICES = [name for name in cx.devices() if name.partition(":")[0] == "vulkan"]


@pytest.fixture(params=_DEVICES or [pytest.param("vulkan", marks=pytest.mark.skip(reason="Vulkan GPU unavailable"))])
def vulkan_device(request):
    yield request.param
    assert cx._core._vulkan_validation_errors() == 0


def from_bits(bits, device):
    return cx.from_dlpack(np.asarray(bits, np.uint32).view(np.float32), copy=False).to(device)


def sqrt_bits_oracle(bits):
    """Exact Python-integer midpoint decision, independent of GPU/root digits."""
    bits = int(bits)
    if not bits & 0x7fffffff:
        return bits
    if bits & 0x7fffffff > 0x7f800000:
        return bits | 0x00400000
    if bits & 0x80000000:
        return 0x7fc00000
    if bits == 0x7f800000:
        return bits
    field, significand = bits >> 23, bits & 0x7fffff
    exponent = field - 127
    if field:
        significand |= 0x800000
    else:
        shift = 24 - significand.bit_length()
        significand <<= shift
        exponent = -126 - shift
    radicand = significand << (23 + (exponent & 1))
    root = math.isqrt(radicand)
    midpoint_comparison = 4 * radicand - (2 * root + 1)**2
    root += midpoint_comparison > 0 or (midpoint_comparison == 0 and root % 2)
    exponent //= 2
    if root == 1 << 24:
        root >>= 1
        exponent += 1
    return ((exponent + 127) << 23) | (root & 0x7fffff)


def test_vulkan_sqrt_exact_midpoints_and_exponents(vulkan_device):
    mantissas = [0, 1, 2, 3, 0x00123456, 0x003504f2, 0x003504f3, 0x003504f4,
                 0x007ffffd, 0x007ffffe, 0x007fffff]
    words = [field << 23 | m for field in range(255) for m in mantissas]
    words += [0x80000000, 0x80000001, 0xbf800000, 0x7f800000,
              0xff800000, 0x7f812345, 0xffc12345]
    bits = np.array(words, np.uint32)
    expected = np.array([sqrt_bits_oracle(b) for b in bits], np.uint32)
    x = from_bits(bits, vulkan_device)
    actual = cx.sqrt(x)
    np.testing.assert_array_equal(actual.numpy().view(np.uint32), expected)
    np.testing.assert_array_equal(x.numpy().view(np.uint32), bits)
    assert not cx._core._shares_storage(x._impl, actual._impl)


@pytest.mark.parametrize("name", ["sqrt", "log"])
def test_vulkan_math_random_full_exponent_range(vulkan_device, name):
    rng = np.random.default_rng(773)
    words = rng.integers(1, 0x7f800000, 131071, dtype=np.uint32)
    values = words.view(np.float32)
    expected = getattr(np, name)(values.astype(np.float64)).astype(np.float32)
    actual = getattr(cx, name)(from_bits(words, vulkan_device)).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)
    if name == "sqrt":
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_vulkan_log_near_one_and_subnormal_extents(vulkan_device):
    near_one = np.arange(0x3f7f0000, 0x3f810001, dtype=np.uint32)
    subnormal = np.unique(np.concatenate([
        np.arange(1, 1025, dtype=np.uint32),
        np.arange(0x007ffc00, 0x00800400, dtype=np.uint32),
        np.array([1 << n for n in range(23)], np.uint32),
    ]))
    words = np.concatenate([near_one, subnormal])
    values = words.view(np.float32)
    expected = np.log(values.astype(np.float64)).astype(np.float32)
    actual = cx.log(from_bits(words, vulkan_device)).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)
    assert actual[:near_one.size][near_one == 0x3f800000].view(np.uint32).item() == 0


def clip_bits_oracle(a, lo, hi):
    def nan(x):
        return x & 0x7fffffff > 0x7f800000

    def less(a, b):
        if not a & 0x7fffffff and not b & 0x7fffffff:
            return False
        key = lambda x: (~x & 0xffffffff) if x & 0x80000000 else x ^ 0x80000000
        return key(a) < key(b)

    a, lo, hi = np.broadcast_arrays(a, lo, hi)
    result = np.empty(a.shape, np.uint32)
    for index in np.ndindex(a.shape):
        value, lower, upper = (int(x[index]) for x in (a, lo, hi))
        if not nan(value) and (nan(lower) or not less(lower, value)):
            value = lower
        if not nan(value) and (nan(upper) or not less(value, upper)):
            value = upper
        result[index] = value
    return result


def test_vulkan_clip_three_way_broadcast_and_stored_bits(vulkan_device):
    a = np.array([0x80000000, 0, 1, 0x80000001, 0x7f812345,
                  0xffc12345, 0x7f800000, 0xff800000], np.uint32).reshape(2, 1, 4)
    lo = np.array([0, 0x80000000, 0x7f854321, 0x807fffff, 0x3f800000], np.uint32).reshape(1, 5, 1)
    hi = np.array([0x80000000, 0x7fc54321, 0x007fffff, 0xbf800000], np.uint32)
    tensors = [from_bits(b, vulkan_device) for b in (a, lo, hi)]
    result = cx.clip(*tensors)
    np.testing.assert_array_equal(result.numpy().view(np.uint32), clip_bits_oracle(a, lo, hi))
    for source, expected in zip(tensors, (a, lo, hi)):
        np.testing.assert_array_equal(source.numpy().view(np.uint32), expected)
        assert not cx._core._shares_storage(source._impl, result._impl)


@pytest.mark.parametrize("dtype", [cx.float32, cx.int32])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("sorted", [False, True])
def test_vulkan_topk_stable_bits_and_independent_outputs(vulkan_device, dtype, largest, sorted):
    if dtype == cx.float32:
        words = np.array([0x7f812345, 0xff800000, 0x7f800000, 0x80000000,
                          0, 1, 0x80000001, 0xffc54321], np.uint32)
        expected_order = [0, 7, 2, 5, 3, 4, 6, 1] if largest else [1, 6, 3, 4, 5, 2, 0, 7]
        x = from_bits(words, vulkan_device)
    else:
        words = np.array([-2**31, 2**31-1, 16777217, 16777216, 0, 0, -1, -1], np.int32).view(np.uint32)
        expected_order = [1, 2, 3, 4, 5, 6, 7, 0] if largest else [0, 6, 7, 4, 5, 3, 2, 1]
        x = cx.tensor(words.view(np.int32), device=vulkan_device)
    for k in (0, 1, 3, 7, 8):
        values, indices = cx.topk(x, k, largest=largest, sorted=sorted)
        selected = np.array(expected_order[:k], np.int32)
        if not sorted:
            selected.sort()
        assert values.dtype == dtype and indices.dtype == cx.int32
        np.testing.assert_array_equal(indices.numpy(), selected)
        np.testing.assert_array_equal(values.numpy().view(np.uint32), words[selected])
        for first, second in ((x, values), (x, indices), (values, indices)):
            assert not cx._core._shares_storage(first._impl, second._impl)
        exported_values, exported_indices = values.numpy(), indices.numpy()
        exported_values[...] = 123
        exported_indices[...] = 456
        np.testing.assert_array_equal(values.numpy().view(np.uint32), words[selected])
        np.testing.assert_array_equal(indices.numpy(), selected)
    np.testing.assert_array_equal(x.numpy().view(np.uint32), words)


def test_vulkan_math_high_rank_empty_and_blocked_export(vulkan_device, monkeypatch):
    x = cx.ones((1,) * 300 + (2, 3), device=vulkan_device)
    lo, hi = cx.tensor(0., device=vulkan_device), cx.tensor(2., device=vulkan_device)

    def forbidden(*args, **kwargs):
        raise AssertionError("Vulkan math attempted a host export")

    with monkeypatch.context() as guard:
        guard.setattr(cx.Tensor, "cpu", forbidden)
        guard.setattr(cx.Tensor, "numpy", forbidden)
        guard.setattr(cx._core, "vulkan_to_cpu", forbidden)
        outputs = [cx.log(x), cx.sqrt(x), cx.abs(x), cx.clip(x, lo, hi), *cx.topk(x, 2)]
    for value in outputs:
        assert value.device == vulkan_device and len(value.shape) == 302
        assert np.isfinite(value.reshape((-1,)).numpy()).all()
    empty = cx.empty((2**40, 0), device=vulkan_device)
    for name in ("log", "sqrt", "abs"):
        assert getattr(cx, name)(empty).shape == empty.shape
    assert cx.clip(empty, lo, hi).shape == empty.shape
    for output in cx.topk(empty, 0):
        assert output.shape == empty.shape


def test_vulkan_math_second_dispatch_dimension(vulkan_device):
    count = 65_535 * 64 + 7
    data = (np.arange(count, dtype=np.int32) % 17).reshape(count, 1)
    x = cx.tensor(data, device=vulkan_device)
    np.testing.assert_array_equal(cx.clip(x, 3, 11).numpy(), np.clip(data, 3, 11))
    values, indices = cx.topk(x, 1)
    np.testing.assert_array_equal(values.numpy(), data)
    np.testing.assert_array_equal(indices.numpy(), np.zeros(data.shape, np.int32))


def test_vulkan_math_threaded_composition(vulkan_device):
    original = np.arange(1, 100, dtype=np.float32).reshape(3, 3, 11)
    x = cx.tensor(original, device=vulkan_device)

    def calculate(index):
        clipped = cx.clip(cx.log(cx.sqrt(x)), 0., index + .5)
        return clipped.topk(3, axis=1, largest=index % 2 == 0, sorted=index % 3 == 0)

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(calculate, range(12)))
    host = cx.tensor(original)
    for index, (values, indices) in enumerate(results):
        expected, expected_indices = cx.clip(cx.log(cx.sqrt(host)), 0., index + .5).topk(
            3, axis=1, largest=index % 2 == 0, sorted=index % 3 == 0)
        assert values.device == indices.device == vulkan_device
        assert values.shape == indices.shape == (3, 3, 11)
        np.testing.assert_allclose(values.numpy(), expected.numpy(), rtol=1e-6, atol=0)
        np.testing.assert_array_equal(indices.numpy(), expected_indices.numpy())
    np.testing.assert_array_equal(x.numpy(), original)
