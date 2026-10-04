"""Public arithmetic/view/reduction contracts on every available backend.

The CUDA acceptance gate requires CUDA before collection, then runs this file
without skips. CPU-only and Metal builds exercise the same public contract.
"""
import gc
import operator

import numpy as np
import pytest

import tensorcx as cx
from tensorcx import _core


@pytest.fixture(params=cx.devices())
def device_name(request):
    return request.param


@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (1,), (257,), (17, 19), (2, 3, 5)])
def test_arithmetic_float32_contract(device_name, shape):
    rng = np.random.default_rng(17)
    a = rng.uniform(-3, 3, shape).astype(np.float32)
    b = rng.uniform(0.25, 3, shape).astype(np.float32)
    x, y = cx.tensor(a, device=device_name), cx.tensor(b, device=device_name)
    for op in (operator.add, operator.sub, operator.mul, operator.truediv):
        for rhs in (y, 2, -0.75, np.int64(3), np.float32(1.25)):
            rhs_ref = b if rhs is y else np.float32(rhs)
            actual = op(x, rhs)
            assert actual.shape == shape and actual.dtype == cx.float32
            assert actual.device == device_name
            np.testing.assert_allclose(actual.numpy(), op(a, rhs_ref), rtol=1e-6, atol=1e-6)
            if rhs is not y:
                np.testing.assert_allclose(op(rhs, x).numpy(), op(rhs_ref, a), rtol=1e-6, atol=1e-6)
        cpu = op(cx.tensor(a), cx.tensor(b))
        cx.testing.assert_allclose(op(x, y), cpu)
    np.testing.assert_array_equal((-x).numpy(), -a)
    np.testing.assert_array_equal(x.numpy(), a)
    np.testing.assert_array_equal(y.numpy(), b)


@pytest.mark.parametrize("shape", [(), (0,), (2, 0), (4,)])
def test_arithmetic_int32_contract(device_name, shape):
    a = np.resize(np.array([-2**31, 2**31 - 1, -1, 0], dtype=np.int32), shape)
    b = np.full(shape, 2, dtype=np.int32)
    x, y = cx.tensor(a, device=device_name), cx.tensor(b, device=device_name)
    if device_name == "cuda":
        for fn in (lambda: x - y, lambda: -x, lambda: x + 2,
                   lambda: 2 - x, lambda: x * 2, lambda: 2 / x):
            with pytest.raises(ValueError, match="float32"):
                fn()
        return
    for op in (operator.add, operator.sub, operator.mul):
        for rhs in (y, 2, -(2**31), np.int64(2**31 - 1)):
            ref = b if rhs is y else np.int32(rhs)
            with np.errstate(over="ignore"):
                expected = op(a, ref)
                reverse = op(ref, a)
            actual = op(x, rhs)
            assert actual.dtype == cx.int32 and actual.device == device_name
            np.testing.assert_array_equal(actual.numpy(), expected)
            if rhs is not y:
                np.testing.assert_array_equal(op(rhs, x).numpy(), reverse)
    with np.errstate(over="ignore"):
        np.testing.assert_array_equal((-x).numpy(), -a)
    for fn in (lambda: x / y, lambda: x / 2, lambda: 2 / x):
        with pytest.raises(ValueError, match="float32"):
            fn()


def test_arithmetic_special_float_values(device_name):
    a = np.array([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan], dtype=np.float32)
    b = np.array([-0.0, 0.0, 0.0, -0.0, np.inf, 2.0, 1.0], dtype=np.float32)
    x, y = cx.tensor(a, device=device_name), cx.tensor(b, device=device_name)
    for op in (operator.add, operator.sub, operator.mul, operator.truediv):
        for rhs, ref in ((y, b), (0.0, np.float32(0)), (-0.0, np.float32(-0.0)),
                         (np.inf, np.float32(np.inf)), (np.nan, np.float32(np.nan))):
            with np.errstate(all="ignore"):
                expected = op(a, ref)
                reverse = op(ref, a)
            actual = op(x, rhs).numpy()
            np.testing.assert_allclose(actual, expected, equal_nan=True)
            np.testing.assert_array_equal(np.signbit(actual[expected == 0]), np.signbit(expected[expected == 0]))
            if rhs is not y:
                np.testing.assert_allclose(op(rhs, x).numpy(), reverse, equal_nan=True)
    result = (-x).numpy()
    np.testing.assert_allclose(result, -a, equal_nan=True)
    np.testing.assert_array_equal(np.signbit(result[:2]), np.signbit((-a)[:2]))


@pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul, operator.truediv])
def test_arithmetic_rejects_invalid_inputs(device_name, op):
    x = cx.ones((2,), device=device_name)
    with pytest.raises(ValueError, match="shape mismatch"):
        op(x, cx.ones((3,), device=device_name))
    with pytest.raises(ValueError, match="shape mismatch"):
        op(x, cx.ones((2, 3), device=device_name))
    with pytest.raises(ValueError, match="dtype mismatch"):
        op(x, cx.tensor([1, 2], device=device_name))
    if device_name != "cpu":
        with pytest.raises(ValueError, match="device mismatch"):
            op(x, cx.ones((2,)))
    for scalar in (True, np.bool_(False), "2", 2j, [2], object()):
        with pytest.raises(TypeError):
            op(x, scalar)
    for array in (np.array(2.0), np.array([2.0]), np.array([2.0, 3.0])):
        with pytest.raises(TypeError):
            op(x, array)
        with pytest.raises(TypeError):
            op(array, x)
    integer = cx.tensor([1, 2], device=device_name)
    for scalar in (1.0, np.float32(1), 1.25, np.inf, 2**31, -(2**31) - 1):
        with pytest.raises(ValueError, match="scalar"):
            op(integer, scalar)


@pytest.mark.parametrize("dtype", [cx.float32, cx.int32])
@pytest.mark.parametrize("shape,target", [((2, 3), (3, 2)), ((2, 3), (3, -1)),
    ((2, 3), -1), ((1,), ()), ((), (1, 1)), ((0,), (2, 0, 3)),
    ((0,), (-1, 3)), ((2, 3), (np.int64(6),))])
def test_reshape_metadata_values_and_lifetime(device_name, dtype, shape, target):
    values = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    x = cx.tensor(values, device=device_name)
    y = cx.reshape(x, target)
    assert _core._shares_storage(x._impl, y._impl)
    assert not _core._shares_storage(x._impl, cx.tensor(values, device=device_name)._impl)
    assert y.dtype == dtype and y.device == device_name and y.nbytes == x.nbytes
    assert x.shape == shape
    np.testing.assert_array_equal(y.numpy(), values.reshape(target))
    z = y.reshape(shape)
    del x, y
    gc.collect()
    np.testing.assert_array_equal(z.numpy(), values)


@pytest.mark.parametrize("shape", [(5,), (-1, -1), (-2, 3), (0, -1), (True, 6),
    (np.bool_(True), 6), (1.5, 4), (2**80,), None])
def test_reshape_rejects_invalid_shapes(device_name, shape):
    x = cx.tensor(np.arange(6, dtype=np.float32), device=device_name)
    with pytest.raises(ValueError):
        x.reshape(shape)


def test_empty_reshape_inference_and_overflow(device_name):
    x = cx.empty((0,), device=device_name)
    for shape in ((0, -1), (-1, 0), (0, 2**62, 4), (2**62, 4, 0)):
        with pytest.raises(ValueError):
            x.reshape(shape)
    y = x.reshape((0, 1, 2**62, 4, 0))
    assert y.shape == (0, 1, 2**62, 4, 0) and y.nbytes == 0


@pytest.mark.parametrize("name", ["sum", "max", "mean"])
@pytest.mark.parametrize("shape,axis", [((2, 3, 4), 0), ((2, 3, 4), 1),
    ((2, 3, 4), -1), ((3,), 0), ((), 0), ((), -1), ((0, 3), 1)])
def test_keepdims_matches_numpy_and_methods(device_name, name, shape, axis):
    values = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    x = cx.tensor(values, device=device_name)
    # tensor.cx permits scalar axis 0/-1 for all reductions; NumPy mean
    # requires axis=None for its rank-0 reference.
    reference_axis = axis if shape else None
    expected = getattr(np, name)(values, axis=reference_axis, keepdims=True)
    for result in (getattr(cx, name)(x, axis=axis, keepdims=True),
                   getattr(x, name)(axis, keepdims=np.bool_(True))):
        assert result.device == device_name and result.dtype == cx.float32
        np.testing.assert_allclose(result.numpy(), expected)
    original = getattr(x, name)(axis)
    assert original.shape == getattr(np, name)(values, axis=reference_axis).shape


def test_keepdims_empty_and_int32_contract(device_name):
    empty = cx.empty((2, 0, 3), device=device_name)
    np.testing.assert_array_equal(empty.sum(1, keepdims=True).numpy(), np.zeros((2, 1, 3)))
    assert np.isnan(empty.mean(1, keepdims=True).numpy()).all()
    with pytest.raises(ValueError):
        empty.max(1, keepdims=True)
    if device_name != "cuda":
        x = cx.tensor([[1, 2], [3, 4]], device=device_name)
        np.testing.assert_array_equal(x.sum(-1, keepdims=True).numpy(), [[3], [7]])
        np.testing.assert_array_equal(x.max(0, keepdims=True).numpy(), [[3, 4]])
        assert x.sum(0, keepdims=True).dtype == cx.int32


def test_keepdims_validates_before_execution(device_name):
    x = cx.ones((2,), device=device_name)
    for name in ("sum", "max", "mean"):
        for flag in (1, "yes", None, []):
            with pytest.raises(TypeError, match="keepdims"):
                getattr(x, name)(0, keepdims=flag)
        with pytest.raises(ValueError, match="axis"):
            getattr(x, name)(2, keepdims=True)
    with pytest.raises(TypeError, match="Tensor"):
        cx.reshape([1, 2], (2,))


def test_native_reshape_and_scalar_reject_bypassed_python_validation():
    x = cx.tensor([1, 2])._impl
    with pytest.raises(ValueError):
        _core.reshape(x, (1,))
    with pytest.raises(ValueError):
        _core.reshape(x, (np.bool_(True), 2))
    for scalar in (True, np.bool_(False), 1.25, 2**31):
        with pytest.raises(ValueError):
            _core.add_scalar(x, scalar)
