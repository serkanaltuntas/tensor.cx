"""Vulkan neural-network numerical boundaries, private outputs and composition."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import tensorcx as cx
from test_vulkan import vulkan_device


def same_non_nan_bits(actual, expected):
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    valid = ~np.isnan(expected)
    np.testing.assert_array_equal(actual.view(np.uint32)[valid], expected.view(np.uint32)[valid])


def test_vulkan_exp_full_range(vulkan_device):
    rng = np.random.default_rng(801)
    random = rng.integers(0, 2**32, 131_071, dtype=np.uint32).view(np.float32)
    edges = np.array([-103.97208, -103.27893, -87.33655, 88.72284], np.float32).view(np.uint32)
    neighbors = np.concatenate([np.arange(int(x) - 64, int(x) + 65, dtype=np.uint32) for x in edges])
    special = np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
                        0x7f800000, 0xff800000, 0x7fc12345, 0xff812345], np.uint32)
    data = np.concatenate([np.linspace(-104, 89, 131_073, dtype=np.float32), random,
                           neighbors.view(np.float32), special.view(np.float32)])
    # Bypass Python float conversion so signaling NaN payloads reach storage.
    source = cx.from_dlpack(data, copy=False).to(vulkan_device)
    np.testing.assert_array_equal(source.numpy().view(np.uint32), data.view(np.uint32))
    with np.errstate(all="ignore"):
        expected = np.exp(data.astype(np.float64)).astype(np.float32)
    actual = source.exp().numpy()
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(actual == 0, expected == 0)
    valid = np.isfinite(expected)
    difference = np.abs(actual[valid].astype(np.float64) - expected[valid].astype(np.float64))
    bound = 1e-6 * np.abs(expected[valid].astype(np.float64)) + np.float64(2**-149)
    assert np.all(difference <= bound)
    assert not np.signbit(actual[~np.isnan(actual)]).any()
    np.testing.assert_array_equal(source.numpy().view(np.uint32), data.view(np.uint32))


@pytest.mark.parametrize("name", ["gelu", "silu"])
def test_vulkan_activations_full_range(vulkan_device, name):
    rng = np.random.default_rng(802)
    words = rng.integers(0, 2**32, 131_071, dtype=np.uint32)
    data = np.concatenate([words.view(np.float32), np.linspace(-12, 12, 65_537, dtype=np.float32),
        np.array([0, 0x80000000, 1, 3, 0x80000001, 0x80000003, 0x007fffff, 0x807fffff,
                  0x7f800000, 0xff800000, 0x7fc12345, 0xff812345], np.uint32).view(np.float32)])
    expected = getattr(cx.tensor(data), name)().numpy()
    actual = getattr(cx.tensor(data, device=vulkan_device), name)().numpy()
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True)
    tiny = np.array([0, 0x80000000, 1, 2, 3, 4, 0x80000001, 0x80000003,
                     0x007fffff, 0x807fffff], np.uint32).view(np.float32)
    expected = getattr(cx.tensor(tiny), name)().numpy()
    actual = getattr(cx.tensor(tiny, device=vulkan_device), name)().numpy()
    same_non_nan_bits(actual, expected)


@pytest.mark.parametrize("name", ["softmax", "rmsnorm", "layernorm"])
@pytest.mark.parametrize("shape", [(), (1,), (3, 5, 7), (2, 1025), (2, 0, 3), (0, 2, 3)])
def test_vulkan_nn_axis_contract(vulkan_device, name, shape):
    data = np.random.default_rng(803).uniform(-2, 2, shape).astype(np.float32)
    source, host = cx.tensor(data, device=vulkan_device), cx.tensor(data)
    for axis in range(-max(1, len(shape)), max(1, len(shape))):
        expected = getattr(host, name)(axis).numpy()
        result = getattr(source, name)(axis)
        assert result.device == vulkan_device and result.shape == shape
        assert not cx._core._shares_storage(result._impl, source._impl)
        if name == "softmax":
            np.testing.assert_allclose(result.numpy(), expected, rtol=2e-6, atol=2**-149, equal_nan=True)
        else:
            same_non_nan_bits(result.numpy(), expected)
    np.testing.assert_array_equal(source.numpy().view(np.uint32), data.view(np.uint32))


@pytest.mark.parametrize("name", ["softmax", "rmsnorm", "layernorm"])
@pytest.mark.parametrize("row", [[0., -0., 2**-149, -2**-149], [1., np.nan, 2., 3.],
    [np.inf, 1., 2., -np.inf], [-np.inf] * 4, [1e20] * 4, [.3] * 4,
    [1e20, -1e20, 1., 0.], [1e-20, -1e-20, 1e-30, 0.]])
def test_vulkan_nn_special_axis(vulkan_device, name, row):
    data = np.array([row, row], np.float32).T.copy()
    for eps in ([None] if name == "softmax" else [0., -0., 2**-149, 1e-5, 1e30]):
        kwargs = {} if eps is None else {"eps": eps}
        expected = getattr(cx.tensor(data), name)(0, **kwargs).numpy()
        actual = getattr(cx.tensor(data, device=vulkan_device), name)(0, **kwargs).numpy()
        if name == "softmax":
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2**-149, equal_nan=True)
        else:
            same_non_nan_bits(actual, expected)


@pytest.mark.parametrize("name", ["rmsnorm", "layernorm"])
@pytest.mark.parametrize("eps", [-1, np.nan, np.inf, 1e300, 1e-300])
def test_vulkan_nn_rejects_epsilon_even_empty(vulkan_device, name, eps):
    for shape in ((2, 3), (0, 3), (2, 0), (2**40, 0)):
        with pytest.raises(ValueError, match="epsilon"):
            getattr(cx.empty(shape, device=vulkan_device), name)(-1, eps)


@pytest.mark.parametrize("name", ["rmsnorm", "layernorm"])
@pytest.mark.parametrize("shape", [(2, 3), (0, 3), (2, 0)])
@pytest.mark.parametrize("affine", [False, True])
def test_vulkan_nn_epsilon_float32_max_boundary(vulkan_device, name, shape, affine):
    maximum = float(np.finfo(np.float32).max)
    # This out-of-range double still rounds to finite float32 max. The CPU
    # checks the original double domain, before narrowing it to the device ABI.
    above = maximum * (1 + 2**-26)
    assert np.float32(above) == np.float32(maximum)
    for device in ("cpu", vulkan_device):
        x = cx.ones(shape, device=device)
        kwargs = {"weight": cx.ones((shape[-1],), device=device)} if affine else {}
        with pytest.raises(ValueError, match="epsilon"):
            getattr(x, name)(-1, above, **kwargs)
        result = getattr(x, name)(-1, maximum, **kwargs)
        assert result.device == device and result.shape == shape


def test_vulkan_nn_empty_huge_and_high_rank(vulkan_device):
    for shape in ((2**40, 0), (0, 1, 2**62, 4, 0)):
        x = cx.empty(shape, device=vulkan_device)
        for name in ("exp", "gelu", "silu"):
            assert getattr(x, name)().shape == shape
        for name in ("softmax", "rmsnorm", "layernorm"):
            for axis in range(-len(shape), len(shape)):
                assert getattr(x, name)(axis).shape == shape
    indices = cx.tensor([0, 2**31 - 1], dtype=cx.int32, device=vulkan_device)
    result = cx.embedding(indices, cx.empty((2**40, 0), device=vulkan_device))
    assert result.shape == (2, 0)
    x = cx.ones((1,) * 300 + (2, 3), device=vulkan_device)
    for name in ("exp", "gelu", "silu", "softmax", "rmsnorm", "layernorm"):
        result = getattr(x, name)(*([-1] if name in ("softmax", "rmsnorm", "layernorm") else []))
        assert result.shape == x.shape
        assert np.isfinite(result.reshape((-1,)).numpy()).all()


def test_vulkan_nn_pipeline_without_host_export(vulkan_device, monkeypatch):
    values = np.arange(6, dtype=np.float32).reshape(2, 3) / 3
    host, x = cx.tensor(values), cx.tensor(values, device=vulkan_device)
    weight = cx.ones((4, 3), device=vulkan_device)
    affine = cx.ones((3,), device=vulkan_device)
    ids = cx.tensor([1, 0, 1], dtype=cx.int32, device=vulkan_device)
    mask = cx.tensor([[True, False], [False, False]], device=vulkan_device)
    expected_attention = cx.attention(host, host, host, cx.tensor([[True, False], [False, False]]))

    def forbidden(*args, **kwargs):
        raise AssertionError("Vulkan inference attempted a host tensor export")

    with monkeypatch.context() as guard:
        for name in ("cpu", "numpy"):
            guard.setattr(cx.Tensor, name, forbidden)
        guard.setattr(cx._core, "vulkan_to_cpu", forbidden)
        outputs = [x.exp(), x.gelu(), x.silu(), x.softmax(-1), x.rmsnorm(-1, weight=affine),
                   x.layernorm(-1, weight=affine, bias=affine), cx.linear(x, weight),
                   cx.embedding(ids, x), cx.attention(x, x, x, mask)]
    for output in outputs:
        assert output.device == vulkan_device and not cx._core._shares_storage(output._impl, x._impl)
        exported = output.numpy()
        exported[...] = 123
        assert not np.all(output.numpy() == 123)
    np.testing.assert_allclose(outputs[-1].numpy(), expected_attention.numpy(), rtol=2e-6, atol=1e-6)
    np.testing.assert_array_equal(x.numpy(), values)


def test_vulkan_nn_second_dispatch_dimension(vulkan_device):
    count = 65_535 * 64 + 7
    x = cx.zeros((count, 1), device=vulkan_device)
    np.testing.assert_array_equal(x.exp().numpy(), np.ones((count, 1), np.float32))
    np.testing.assert_array_equal(x.softmax(-1).numpy(), np.ones((count, 1), np.float32))


def test_vulkan_nn_concurrent_inference(vulkan_device):
    rng = np.random.default_rng(804)
    data = rng.normal(size=(7, 17)).astype(np.float32)
    weights = rng.normal(size=(5, 17)).astype(np.float32)
    x, w = cx.tensor(data, device=vulkan_device), cx.tensor(weights, device=vulkan_device)
    def forward(a, b):
        return cx.linear(a.layernorm(-1).gelu(), b).softmax(-1)
    expected = forward(cx.tensor(data), cx.tensor(weights)).numpy()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: forward(x, w), range(8)))
    for result in results:
        np.testing.assert_allclose(result.numpy(), expected, rtol=2e-5, atol=1e-6)
    np.testing.assert_array_equal(x.numpy(), data)
    np.testing.assert_array_equal(w.numpy(), weights)
