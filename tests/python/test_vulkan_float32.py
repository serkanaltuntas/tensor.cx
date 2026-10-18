"""Binary32 RNE arithmetic on Vulkan, independent of device denormal controls."""
from concurrent.futures import ThreadPoolExecutor
import operator

import numpy as np
import pytest

import tensorcx as cx

_DEVICES = [name for name in cx.devices() if name.partition(":")[0] == "vulkan"]
_OPS = [operator.add, operator.sub, operator.mul, operator.truediv]


@pytest.fixture(params=_DEVICES or [pytest.param("vulkan", marks=pytest.mark.skip(reason="Vulkan GPU unavailable"))])
def vulkan_device(request):
    yield request.param
    assert cx._core._vulkan_validation_errors() == 0


def assert_float_bits(actual, expected):
    a = actual.numpy() if isinstance(actual, cx.Tensor) else np.asarray(actual)
    b = expected.numpy() if isinstance(expected, cx.Tensor) else np.asarray(expected)
    assert a.shape == b.shape and a.dtype == b.dtype == np.float32
    nan = np.isnan(b)
    np.testing.assert_array_equal(np.isnan(a), nan)
    np.testing.assert_array_equal(a.view(np.uint32)[~nan], b.view(np.uint32)[~nan])
    # Vulkan arithmetic canonicalizes NaNs instead of preserving source bits.
    assert np.all(a.view(np.uint32)[nan] == np.uint32(0x7fc00000))


@pytest.mark.parametrize("operation", _OPS)
def test_vulkan_float32_all_exponent_boundaries(vulkan_device, operation):
    exponents = np.array([0, 1, 2, 22, 23, 24, 25, 126, 127, 128, 230, 253, 254, 255], np.uint32)
    mantissas = np.array([0, 1, 2, 3, 0x3fffff, 0x400000, 0x7ffffd, 0x7ffffe, 0x7fffff], np.uint32)
    magnitudes = ((exponents[:, None] << 23) | mantissas).reshape(-1)
    bits = np.concatenate([magnitudes, magnitudes | np.uint32(0x80000000)])
    a, b = bits.view(np.float32)[:, None], bits.view(np.float32)[None, :]
    host_a, host_b = cx.tensor(a), cx.tensor(b)
    x, y = host_a.to(vulkan_device), host_b.to(vulkan_device)
    with np.errstate(all="ignore"):
        expected = operation(a, b)
    assert_float_bits(operation(x, y), expected)
    assert_float_bits(operation(x, y), operation(host_a, host_b))
    np.testing.assert_array_equal(x.numpy().view(np.uint32), host_a.numpy().view(np.uint32))
    np.testing.assert_array_equal(y.numpy().view(np.uint32), host_b.numpy().view(np.uint32))


@pytest.mark.parametrize("operation", _OPS)
def test_vulkan_float32_random_bits(vulkan_device, operation):
    rng = np.random.default_rng(991)
    a, b = rng.integers(0, 2**32, size=(2, 1_048_579), dtype=np.uint32).view(np.float32)
    with np.errstate(all="ignore"):
        expected = operation(a, b)
    x, y = cx.tensor(a, device=vulkan_device), cx.tensor(b, device=vulkan_device)
    assert_float_bits(operation(x, y), expected)


@pytest.mark.parametrize("operation", _OPS)
@pytest.mark.parametrize("scalar_bits", [0, 0x80000000, 1, 0x007fffff, 0x00800000,
                                        0x7f7fffff, 0x7f800000, 0xff800000,
                                        0x7fc12345, 0x3f800001, 0x3f000000, 0x40000000])
@pytest.mark.parametrize("shape", [(), (0,), (2, 1, 9)])
def test_vulkan_float32_scalars_and_empty_shapes(vulkan_device, operation, scalar_bits, shape):
    bits = np.resize(np.array([0, 0x80000000, 1, 3, 0x007fffff, 0x00800001,
                               0x7f7fffff, 0x3f800001, 0xbf800000], np.uint32), shape)
    host = cx.tensor(bits.view(np.float32))
    x = host.to(vulkan_device)
    scalar = np.array(scalar_bits, np.uint32).view(np.float32)[()]
    for result, expected in ((operation(x, scalar), operation(host, scalar)),
                             (operation(scalar, x), operation(scalar, host))):
        assert result.device == vulkan_device
        assert_float_bits(result, expected)


@pytest.mark.parametrize("operation", _OPS)
def test_vulkan_float32_subnormal_mask_composition(vulkan_device, operation):
    bits = np.array([1, 2, 3, 0x007fffff, 0x00800000, 0x00800001,
                     0x80000001, 0x807fffff, 0x80800000], np.uint32)
    host = cx.tensor(bits.view(np.float32))
    x = host.to(vulkan_device)
    # Subnormal inputs can produce ordinary finite values (e.g. x/x). Truth and
    # selection must therefore agree with CPU after the arithmetic, not only
    # under an absolute error tolerance that hides tiny numerical differences.
    result, expected = operation(x, x), operation(host, host)
    assert_float_bits(result, expected)
    np.testing.assert_array_equal(result.astype(cx.bool).numpy(), expected.astype(cx.bool).numpy())
    assert_float_bits(result[result > 0], expected[expected > 0])


def test_vulkan_float32_round_to_even_and_signed_cancellation(vulkan_device):
    bits = np.array([0x3f800000, 0x3f800001, 0x3f800002, 1, 3, 5,
                     0x80000000, 0, 0x80000001, 0x00800000], np.uint32)
    host = cx.tensor(bits.view(np.float32))
    x = host.to(vulkan_device)
    for result, expected in ((x + np.float32(2**-24), host + np.float32(2**-24)),
                             (x * 0.5, host * 0.5), (x / 2, host / 2),
                             (x - x, host - host), (-x - x, -host - host)):
        assert_float_bits(result, expected)


def test_vulkan_float32_division_uses_second_dispatch_dimension(vulkan_device):
    size = 65_535 * 64 + 7
    bits = np.resize(np.array([1, 3, 0x007fffff, 0x00800000, 0x3f800001, 0x7f7fffff], np.uint32), size)
    host = cx.tensor(bits.view(np.float32))
    assert_float_bits(host.to(vulkan_device) / 2, host / 2)


def test_vulkan_float32_threaded_finite_and_nan_inputs(vulkan_device):
    bits = np.array([0, 1, 3, 0x00800000, 0x7f7fffff, 0x7f800000, 0x7fc00001], np.uint32)
    host = cx.tensor(bits.view(np.float32))
    x = host.to(vulkan_device)
    def work(i):
        operation = _OPS[i % len(_OPS)]
        assert_float_bits(operation(x, x), operation(host, host))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(work, range(24)))
