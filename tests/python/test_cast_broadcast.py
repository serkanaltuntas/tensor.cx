"""Explicit conversion and native broadcasting, shared by all backend gates."""
import gc
import operator

import numpy as np
import pytest

import tensorcx as cx
from tensorcx import _core


@pytest.fixture(params=cx.devices())
def device_name(request):
    return request.param


@pytest.mark.parametrize("source,target", [(cx.float32, cx.int32), (cx.int32, cx.float32),
                                         (cx.float32, cx.float32), (cx.int32, cx.int32)])
@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (1,), (257,), (2, 3, 5)])
def test_astype_values_shape_device_and_storage(device_name, source, target, shape):
    values = np.resize(np.array([-17, -1, 0, 1, 23], dtype=source), shape)
    if source == cx.float32:
        values = values + np.float32(0.75)
    x = cx.tensor(values, device=device_name)
    for convert in (lambda: x.astype(target), lambda: cx.astype(x, target),
                    lambda: x.astype(target, copy=False)):
        y = convert()
        assert y.shape == shape and y.dtype == target and y.device == device_name
        np.testing.assert_array_equal(y.numpy(), values.astype(target))
        if y is not x:
            assert not _core._shares_storage(x._impl, y._impl)
    assert x.astype(source, copy=False) is x
    assert x.astype(source, copy=np.bool_(False)) is x
    np.testing.assert_array_equal(x.numpy(), values)
    y = x.astype(target)
    del x
    gc.collect()
    np.testing.assert_array_equal(y.numpy(), values.astype(target))


def test_astype_boundary_rounding_and_truncation(device_name):
    ints = np.array([-2**31, -16777219, -16777217, -1, 0, 16777217, 16777219,
                     2**31 - 1], dtype=np.int32)
    actual = cx.tensor(ints, device=device_name).astype(cx.float32)
    np.testing.assert_array_equal(actual.numpy(), ints.astype(np.float32))
    limit = np.float32(2**31)
    floats = np.array([-limit, np.nextafter(-limit, np.float32(0)), -2.9, -0.9,
                       -0.0, 0.0, 0.9, 2.9, np.nextafter(limit, np.float32(0))], dtype=np.float32)
    result = cx.tensor(floats, device=device_name).astype(cx.int32)
    np.testing.assert_array_equal(result.numpy(), floats.astype(np.int32))
    with pytest.raises(ValueError):
        actual.astype(cx.int32)  # INT_MAX rounds up to 2**31 in float32.


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, np.float32(2**31),
    np.nextafter(np.float32(-2**31), np.float32(-np.inf)), np.finfo(np.float32).max])
def test_astype_rejects_invalid_float_without_mutation(device_name, invalid):
    values = np.array([1.75, -2.25, invalid, 4], dtype=np.float32)
    x = cx.tensor(values, device=device_name)
    view = x.reshape((2, 2))
    for copy in (True, False):
        with pytest.raises(ValueError):
            view.astype(cx.int32, copy=copy)
        np.testing.assert_array_equal(x.numpy(), values)
    # A failing operation must not poison the backend for subsequent work.
    np.testing.assert_array_equal(cx.tensor([1.75], device=device_name).astype(cx.int32).numpy(), [1])


def test_astype_same_dtype_preserves_special_float_bits(device_name):
    bits = np.array([0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 1], dtype=np.uint32)
    x = cx.tensor(bits.view(np.float32), device=device_name)
    y = x.astype(cx.float32)
    np.testing.assert_array_equal(y.numpy().view(np.uint32), x.numpy().view(np.uint32))
    assert not _core._shares_storage(x._impl, y._impl)


def test_astype_argument_errors(device_name):
    x = cx.tensor([1.0], device=device_name)
    for dtype in (None, "float64", "int64", "bool", np.float32, np.dtype("float32"), True, 1):
        with pytest.raises(ValueError, match="dtype"):
            x.astype(dtype)
    for copy in (None, 0, 1, "false", []):
        with pytest.raises(TypeError, match="copy"):
            x.astype(cx.float32, copy=copy)
    with pytest.raises(TypeError, match="Tensor"):
        cx.astype([1.0], cx.float32)
    with pytest.raises((ValueError, TypeError)):
        _core.astype(x._impl, "float64")


BROADCAST_SHAPES = [((2, 3), (3,)), ((3, 1), (1, 5)), ((2, 1, 3), (4, 1)),
    ((), (2, 3)), ((2, 3), ()), ((), ()), ((1,), ()),
    ((2, 0, 3), (1, 3)), ((0,), (1,)), ((1, 0, 1), (2, 1, 4)),
    ((2, 1, 3, 1, 2), (1, 4, 1, 5, 1)), ((257, 1), (1, 3))]


@pytest.mark.parametrize("lhs_shape,rhs_shape", BROADCAST_SHAPES)
@pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul, operator.truediv])
def test_broadcast_float32_numpy_and_cpu_parity(device_name, lhs_shape, rhs_shape, op):
    rng = np.random.default_rng(83)
    a = rng.uniform(-3, 3, lhs_shape).astype(np.float32)
    b = rng.uniform(0.2, 3, rhs_shape).astype(np.float32)
    x, y = cx.tensor(a, device=device_name), cx.tensor(b, device=device_name)
    result = op(x, y)
    expected = op(a, b)
    assert result.shape == expected.shape and result.dtype == cx.float32
    assert result.device == device_name
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6, atol=1e-6)
    cx.testing.assert_allclose(result, op(cx.tensor(a), cx.tensor(b)))
    np.testing.assert_array_equal(x.numpy(), a)
    np.testing.assert_array_equal(y.numpy(), b)


@pytest.mark.parametrize("lhs_shape,rhs_shape", [((3, 1), (1, 4)), ((), (3,)), ((0, 2), (2,))])
def test_broadcast_int32_contract(device_name, lhs_shape, rhs_shape):
    a = np.resize(np.array([-2**31, 2**31-1, -1], dtype=np.int32), lhs_shape)
    b = np.resize(np.array([2, -1, 0, 3], dtype=np.int32), rhs_shape)
    x, y = cx.tensor(a, device=device_name), cx.tensor(b, device=device_name)
    for op in (operator.add, operator.sub, operator.mul):
        if device_name == "cuda":
            with pytest.raises(ValueError, match="float32"):
                op(x, y)
        else:
            with np.errstate(over="ignore"):
                expected = op(a, b)
            np.testing.assert_array_equal(op(x, y).numpy(), expected)
    with pytest.raises(ValueError, match="float32"):
        x / y


def test_broadcast_no_arbitrary_rank_cap(device_name):
    # NumPy's rank limit is not a runtime limit. Use known values as the oracle.
    x = cx.tensor([1., 2., 3.], device=device_name).reshape((1,) * 70 + (3,))
    y = cx.tensor([10., 20.], device=device_name).reshape((2,) + (1,) * 71)
    result = x + y
    assert result.shape == (2,) + (1,) * 70 + (3,)
    np.testing.assert_array_equal(result.reshape((2, 3)).numpy(), [[11, 12, 13], [21, 22, 23]])


def test_broadcast_ieee_values_and_reverse_order(device_name):
    a = np.array([[0.0], [-0.0], [np.inf], [-np.inf], [np.nan]], dtype=np.float32)
    b = np.array([0.0, -0.0, 1.0, -2.0, np.inf], dtype=np.float32)
    x, y = cx.tensor(a, device=device_name), cx.tensor(b, device=device_name)
    for op in (operator.add, operator.sub, operator.mul, operator.truediv):
        for lhs, rhs, left, right in ((x, y, a, b), (y, x, b, a)):
            with np.errstate(all="ignore"):
                expected = op(left, right)
            actual = op(lhs, rhs).numpy()
            np.testing.assert_allclose(actual, expected, equal_nan=True)
            zeros = expected == 0
            np.testing.assert_array_equal(np.signbit(actual[zeros]), np.signbit(expected[zeros]))


def test_broadcast_errors_and_empty_validation(device_name):
    for left, right in [((2, 3), (2,)), ((0, 3), (2, 3)), ((1, 0), (2,))]:
        x, y = cx.empty(left, device=device_name), cx.empty(right, device=device_name)
        for op in (operator.add, operator.sub, operator.mul, operator.truediv):
            with pytest.raises(ValueError, match="shape mismatch"):
                op(x, y)
    with pytest.raises(ValueError, match="dtype mismatch"):
        cx.tensor([[1.0]], device=device_name) + cx.tensor([1, 2], device=device_name)
    # Both inputs are representable empty tensors; the broadcast result's
    # non-zero prefix would overflow before reaching its trailing empty axis.
    x = cx.empty((2**32, 1, 0), device=device_name)
    y = cx.empty((1, 2**32, 0), device=device_name)
    with pytest.raises(ValueError, match="overflow"):
        x + y


def test_cast_broadcast_composition_stays_native(device_name, monkeypatch):
    x = cx.tensor([[1, 2, 3], [4, 5, 6]], device=device_name)
    bias = cx.tensor([0.25, 0.5, 0.75], device=device_name)
    def forbidden(*args, **kwargs):
        raise AssertionError("operation unexpectedly exported tensor data")
    with monkeypatch.context() as patch:
        patch.setattr(cx.Tensor, "numpy", forbidden)
        patch.setattr(cx.Tensor, "cpu", forbidden)
        floats = x.astype(cx.float32)
        result = floats + bias
        centered = result - result.mean(-1, keepdims=True)
    np.testing.assert_allclose(centered.numpy(), [[-1.25, 0, 1.25], [-1.25, 0, 1.25]])
    assert centered.device == device_name
