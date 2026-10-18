"""Real Vulkan matmul: binary32 ordering, empty metadata and private outputs."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import tensorcx as cx

_DEVICES = [name for name in cx.devices() if name.partition(":")[0] == "vulkan"]


@pytest.fixture(params=_DEVICES or [pytest.param("vulkan", marks=pytest.mark.skip(reason="Vulkan GPU unavailable"))])
def vulkan_device(request):
    yield request.param
    assert cx._core._vulkan_validation_errors() == 0


def sequential_matmul(a, b):
    """Independent NumPy elementwise oracle: no BLAS/FMA or reassociation."""
    shape = np.broadcast_shapes(a.shape[:-2], b.shape[:-2]) + (a.shape[-2], b.shape[-1])
    result = np.zeros(shape, np.float32)
    with np.errstate(all="ignore"):
        for inner in range(a.shape[-1]):
            product = np.multiply(a[..., :, inner, None], b[..., None, inner, :], dtype=np.float32)
            result = np.add(result, product, dtype=np.float32)
    return result


def assert_bits(actual, expected):
    values = actual.numpy()
    assert actual.shape == expected.shape and actual.dtype == cx.float32
    nan = np.isnan(expected)
    np.testing.assert_array_equal(np.isnan(values), nan)
    np.testing.assert_array_equal(values.view(np.uint32)[~nan], expected.view(np.uint32)[~nan])
    assert np.all(values.view(np.uint32)[nan] == np.uint32(0x7fc00000))


@pytest.mark.parametrize("inner", [1, 2, 3, 17, 65, 257])
@pytest.mark.parametrize("preference", ["auto", "custom"])
def test_vulkan_matmul_separate_rounding_and_underflow(vulkan_device, inner, preference):
    rng = np.random.default_rng(189 + inner)
    a_bits = rng.integers(0, 2**32, (3, 1, 4, inner), dtype=np.uint32)
    b_bits = rng.integers(0, 2**32, (1, 2, inner, 5), dtype=np.uint32)
    boundaries = np.array([1, 3, 0x007fffff, 0x00800000, 0x80000001,
                           0x3f000000, 0x3f800001, 0x7f7fffff, 0x80000000], np.uint32)
    a_bits.reshape(-1)[:len(boundaries)] = boundaries[:a_bits.size]
    b_bits.reshape(-1)[:len(boundaries)] = boundaries[::-1][:b_bits.size]
    a, b = a_bits.view(np.float32), b_bits.view(np.float32)
    # DLPack avoids Python scalar conversions quieting input signaling NaNs.
    x, y = [cx.from_dlpack(value, copy=False).to(vulkan_device) for value in (a, b)]
    actual = cx.matmul(x, y, backend=preference)
    assert actual.device == vulkan_device
    assert_bits(actual, sequential_matmul(a, b))
    for source, original in ((x, a_bits), (y, b_bits)):
        np.testing.assert_array_equal(source.numpy().view(np.uint32), original)
        assert not cx._core._shares_storage(actual._impl, source._impl)


@pytest.mark.parametrize("preference", ["auto", "custom"])
def test_vulkan_matmul_special_values_and_no_fma(vulkan_device, preference):
    bits = np.array([
        [1, 3, 0x007fffff, 0x00800000],
        [0x80000001, 0x80000003, 0x807fffff, 0x80800000],
        [0x80000000, 0, 0x80000000, 0],
        [0x7f812345, 0x3f800000, 0x7fc12345, 0],
        [0x7f800000, 0xff800000, 0x7f7fffff, 0x3f800000],
        # Fusing the second multiply/add would produce a different result.
        [0xbf800000, 0x3f800001, 0, 0],
    ], np.uint32)
    right = np.array([[1., 0., -1.], [1. - 2**-23, 2., 0.5],
                      [0.5, 1., 0.], [1., -1., 2.]], np.float32)
    host = cx.from_dlpack(bits.view(np.float32), copy=False)
    x, y = host.to(vulkan_device), cx.tensor(right, device=vulkan_device)
    assert_bits(cx.matmul(x, y, backend=preference), sequential_matmul(bits.view(np.float32), right))
    assert cx.matmul(x, y, backend=preference).numpy()[5, 0] == np.float32(0)
    np.testing.assert_array_equal(x.numpy().view(np.uint32), bits)


@pytest.mark.parametrize("shapes", [
    ((2**40, 0), (0, 0)), ((0, 2**40), (2**40, 0)),
    ((0, 2**40, 4, 0, 2), (1, 1, 1, 2, 0)),
    ((1,) * 300 + (2, 0), (0, 3)), ((0,), (0,)),
])
def test_vulkan_matmul_zero_and_unused_huge_dimensions(vulkan_device, shapes):
    a, b = [cx.empty(shape, device=vulkan_device) for shape in shapes]
    host_a, host_b = [cx.empty(shape) for shape in shapes]
    for preference in ("auto", "custom"):
        result = cx.matmul(a, b, backend=preference)
        expected = cx.matmul(host_a, host_b)
        assert result.shape == expected.shape and result.dtype == cx.float32
        flat = result.reshape((-1,)).numpy()
        np.testing.assert_array_equal(flat.view(np.uint32), np.zeros(flat.size, np.uint32))


def test_vulkan_matmul_large_output_uses_second_grid_dimension(vulkan_device):
    count = 65_535 * 64 + 7
    values = (np.arange(count, dtype=np.float32) % 17).reshape(count, 1)
    x = cx.tensor(values, device=vulkan_device)
    result = cx.matmul(x, cx.tensor([2.], device=vulkan_device), backend="custom")
    np.testing.assert_array_equal(result.numpy(), values[:, 0] * 2)
    np.testing.assert_array_equal(x.numpy(), values)


@pytest.mark.parametrize("preference", ["optimized", "reference", "cpu", "unknown", ""])
def test_vulkan_matmul_rejects_unavailable_preferences(vulkan_device, preference):
    assert cx.matmul_backends(vulkan_device) == ["auto", "custom"]
    for shape in ((3,), (0,)):
        x = cx.ones(shape, device=vulkan_device)
        with pytest.raises(ValueError, match="auto or custom"):
            cx.matmul(x, x, backend=preference)
        np.testing.assert_array_equal(x.numpy(), np.ones(shape, np.float32))
    assert cx.matmul(cx.ones((3,), device=vulkan_device), cx.ones((3,), device=vulkan_device)).numpy() == 3


def test_vulkan_matmul_threaded_composition_and_private_output(vulkan_device, monkeypatch):
    a = cx.tensor(np.arange(30, dtype=np.float32).reshape(2, 3, 5) / 8, device=vulkan_device)
    b = cx.tensor(np.arange(20, dtype=np.float32).reshape(1, 5, 4) / 8, device=vulkan_device)
    expected = a.numpy() @ b.numpy()

    def run(index):
        return cx.matmul(a, b, backend="auto" if index % 2 else "custom") + index

    def forbidden(*args, **kwargs):
        raise AssertionError("matmul attempted host export")

    with monkeypatch.context() as guard:
        guard.setattr(cx.Tensor, "cpu", forbidden)
        guard.setattr(cx.Tensor, "numpy", forbidden)
        guard.setattr(cx._core, "vulkan_to_cpu", forbidden)
        with ThreadPoolExecutor(max_workers=6) as pool:
            outputs = list(pool.map(run, range(12)))
    for index, output in enumerate(outputs):
        np.testing.assert_array_equal(output.numpy(), expected + index)
        assert not cx._core._shares_storage(a._impl, output._impl)
        assert not cx._core._shares_storage(b._impl, output._impl)
    np.testing.assert_array_equal(a.numpy(), np.arange(30, dtype=np.float32).reshape(2, 3, 5) / 8)
