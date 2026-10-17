"""Exact native int32 contracts on every available backend, including CUDA."""
import gc
import operator

import numpy as np
import pytest

import tensorcx as cx
from tensorcx import _core


@pytest.fixture(params=cx.devices())
def device_name(request):
    return request.param


EDGES = np.array([-2**31, 2**31 - 1, -16777217, 16777217, -1, 0, 1, 65537], dtype=np.int32)
SHAPES = [(), (0,), (2, 0, 3), (1,), (255,), (256,), (257,), (3, 257), (2, 3, 5)]
OPS = [operator.add, operator.sub, operator.mul]


def verify(result, expected, device, *inputs):
    assert result.dtype == cx.int32 and result.device == device
    assert result.shape == expected.shape
    np.testing.assert_array_equal(result.numpy(), expected)
    for source in inputs:
        assert not _core._shares_storage(source._impl, result._impl)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("factory,value", [(cx.zeros, 0), (cx.ones, 1)])
def test_int32_factories(device_name, shape, factory, value):
    verify(factory(shape, dtype=cx.int32, device=device_name), np.full(shape, value, np.int32), device_name)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("op", OPS)
def test_int32_binary_exact_wrap(device_name, shape, op):
    a = np.resize(EDGES, shape)
    b = np.resize(EDGES[::-1], shape)
    x, y = cx.tensor(a, device=device_name), cx.tensor(b, device=device_name)
    with np.errstate(over="ignore"):
        expected = op(a, b)
    verify(op(x, y), expected, device_name, x, y)
    np.testing.assert_array_equal(op(cx.tensor(a), cx.tensor(b)).numpy(), expected)
    np.testing.assert_array_equal(x.numpy(), a)
    np.testing.assert_array_equal(y.numpy(), b)


@pytest.mark.parametrize("shape", [(), (0,), (257,)])
@pytest.mark.parametrize("scalar", [-2**31, 2**31 - 1, -16777217, 16777217, -1, 0])
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("reverse", [False, True])
def test_int32_scalar_precision_and_order(device_name, shape, scalar, op, reverse):
    a = np.resize(EDGES, shape)
    x = cx.tensor(a, device=device_name)
    with np.errstate(over="ignore"):
        expected = op(np.int32(scalar), a) if reverse else op(a, np.int32(scalar))
    result = op(np.int64(scalar), x) if reverse else op(x, np.int64(scalar))
    verify(result, expected, device_name, x)
    np.testing.assert_array_equal(x.numpy(), a)


@pytest.mark.parametrize("shape", SHAPES)
def test_int32_negate_wrap(device_name, shape):
    a = np.resize(EDGES, shape)
    x = cx.tensor(a, device=device_name)
    with np.errstate(over="ignore"):
        expected = -a
    verify(-x, expected, device_name, x)
    np.testing.assert_array_equal(x.numpy(), a)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("shapes", [((2, 1, 257), (3, 1)), ((257,), ()),
    ((), (257,)), ((2, 0, 3), (1, 3)), ((0, 3), (1, 3))])
def test_int32_broadcast_exact_wrap(device_name, op, shapes):
    a, b = (np.resize(EDGES, shape) for shape in shapes)
    x, y = cx.tensor(a, device=device_name), cx.tensor(b, device=device_name)
    with np.errstate(over="ignore"):
        expected = op(a, b)
    verify(op(x, y), expected, device_name, x, y)
    np.testing.assert_array_equal(x.numpy(), a)
    np.testing.assert_array_equal(y.numpy(), b)


REDUCTIONS = [((2, 3, 257), 0), ((2, 3, 257), 1), ((3, 257), -1),
    ((2, 3, 257), (0, 2)), ((2, 3, 257), (2, 0)), ((3, 257), None),
    ((2, 3), ()), ((2, 0, 3), 1), ((0, 2, 3), (1, 2)),
    ((0, 2, 3), (0, 2)), ((2, 0, 3), None), ((), 0), ((), -1), ((), None)]


@pytest.mark.parametrize("shape,axes", REDUCTIONS)
@pytest.mark.parametrize("name", ["sum", "max"])
@pytest.mark.parametrize("keepdims", [False, True])
def test_int32_reductions_exact_wrap(device_name, shape, axes, name, keepdims):
    a = np.resize(EDGES, shape)
    x = cx.tensor(a, device=device_name)
    kwargs = dict(axis=None if not shape and axes in (0, -1) else axes, keepdims=keepdims)
    if name == "sum":
        kwargs["dtype"] = np.int32
    try:
        expected = getattr(np, name)(a, **kwargs)
    except ValueError:
        with pytest.raises(ValueError, match="non-empty"):
            getattr(x, name)(axes, keepdims=keepdims)
    else:
        verify(getattr(x, name)(axes, keepdims=keepdims), expected, device_name, x)
        np.testing.assert_array_equal(getattr(cx.tensor(a), name)(axes, keepdims=keepdims).numpy(), expected)
    np.testing.assert_array_equal(x.numpy(), a)


@pytest.mark.parametrize("op", OPS)
def test_int32_random_bits_and_signed_max(device_name, op):
    rng = np.random.default_rng(4401)
    a, b = (rng.integers(0, 2**32, size=(7, 1027), dtype=np.uint32).view(np.int32) for _ in range(2))
    x, y = cx.tensor(a, device=device_name), cx.tensor(b, device=device_name)
    with np.errstate(over="ignore"):
        expected = op(a, b)
    verify(op(x, y), expected, device_name, x, y)
    verify(x.sum((0, 1)), a.sum(dtype=np.int32), device_name, x)
    verify(x.max(1), a.max(axis=1), device_name, x)
    negatives = np.array([[-2**31, -7, -16777217]], dtype=np.int32)
    assert cx.tensor(negatives, device=device_name).max().numpy() == -7


@pytest.mark.parametrize("shape", [(), (0,), (2, 0), (3,)])
def test_int32_float_only_rejections_and_invalid_scalars(device_name, shape):
    x = cx.tensor(np.resize(EDGES, shape), device=device_name)
    for action in (lambda: x / x, lambda: x / 2, lambda: 2 / x,
                   lambda: x.mean(), lambda: x.mean(()), lambda: cx.exp(x),
                   lambda: cx.gelu(x), lambda: cx.silu(x), lambda: cx.softmax(x, axis=-1)):
        with pytest.raises(ValueError, match="float32"):
            action()
    for scalar in (1.0, np.float32(1), np.inf, 2**31, -2**31 - 1):
        for op in OPS:
            with pytest.raises(ValueError, match="scalar"):
                op(x, scalar)
            with pytest.raises(ValueError, match="scalar"):
                op(scalar, x)


def test_int32_high_rank_alias_lifetime_and_no_numpy_fallback(device_name, monkeypatch):
    x = cx.tensor(EDGES, device=device_name)
    alias = x.reshape((1,) * 70 + (8,))
    y = cx.tensor([16777217, -16777217], dtype=cx.int32, device=device_name).reshape((2,) + (1,) * 71)
    with monkeypatch.context() as patch:
        patch.setattr(cx.Tensor, "numpy", lambda *a, **k: pytest.fail("host export during native computation"))
        result = ((alias + y) * 3 - 16777217).sum((0, 71))
        maximum = alias.max((70,))
    with np.errstate(over="ignore"):
        expected = ((EDGES[None, :] + np.array([16777217, -16777217], np.int32)[:, None]) * np.int32(3)
                    - np.int32(16777217)).sum(dtype=np.int32)
    assert result.shape == (1,) * 70
    assert result.reshape(()).numpy() == expected
    assert maximum.reshape(()).numpy() == EDGES.max()
    assert not _core._shares_storage(result._impl, x._impl)
    np.testing.assert_array_equal(x.numpy(), EDGES)
    del x, alias, y
    gc.collect()
    assert result.reshape(()).numpy() == expected


def test_int32_grid_stride_tail(device_name):
    # Above the 65535-block cap: the final lanes must take a second iteration.
    count = 65535 * 256 + 17
    x = cx.ones((count,), dtype=cx.int32, device=device_name)
    y = -(x * 16777217 + x)
    actual = y.numpy()
    assert actual.dtype == np.int32 and actual.shape == (count,)
    assert np.all(actual == -16777218)
