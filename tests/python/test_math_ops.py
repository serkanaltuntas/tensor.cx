"""Math extension parity, dtype/domain contracts and native device execution."""
import builtins
import numpy as np
import pytest
import tensorcx as cx
from tensorcx import _core

@pytest.fixture(params=cx.devices())
def device(request):
    return request.param


def check(result, expected, device, dtype=None):
    expected = np.asarray(expected)
    assert result.device == device and result.shape == expected.shape
    assert result.dtype == (dtype or str(expected.dtype))
    if result.dtype == 'float32':
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6, atol=0, equal_nan=True)
    else:
        np.testing.assert_array_equal(result.numpy(), expected)


@pytest.mark.parametrize('name', ['log', 'sqrt', 'abs'])
@pytest.mark.parametrize('shape', [(), (0,), (2, 0, 3), (1,), (2, 3, 5), (513,)])
def test_unary_math_parity(device, name, shape):
    a = np.random.default_rng(912).uniform(.01, 100, shape).astype(np.float32)
    x = cx.tensor(a, device=device)
    expected = getattr(np, name)(a)
    y = getattr(cx, name)(x)
    check(y, expected, device)
    check(getattr(x, name)(), expected, device)
    check(getattr(cx, name)(cx.tensor(a)), expected, 'cpu')
    assert not _core._shares_storage(x._impl, y._impl)
    np.testing.assert_array_equal(x.numpy(), a)


@pytest.mark.parametrize('name', ['log', 'sqrt', 'abs'])
def test_unary_ieee_and_subnormal(device, name):
    bits = np.array([0, 0x80000000, 1, 2, 0x007fffff, 0x00800000,
                     0x80000001, 0x807fffff, 0x80800000, 0x3f800000,
                     0xbf800000, 0x7f7fffff, 0x7f800000, 0xff800000, 0x7fc01234], np.uint32)
    a = bits.view(np.float32)
    with np.errstate(all='ignore'):
        expected = getattr(np, name)(a)
    y = getattr(cx, name)(cx.tensor(a, device=device))
    check(y, expected, device)
    if name == 'abs':
        np.testing.assert_array_equal(y.numpy().view(np.uint32), bits & 0x7fffffff)
    if name == 'sqrt':
        assert y.numpy().view(np.uint32)[1] == 0x80000000


def test_abs_integer_wrap(device):
    a = np.array([-2**31, -2**31+1, -16777217, -1, 0, 1, 2**31-1], np.int32)
    x = cx.tensor(a, device=device)
    check(cx.abs(x), np.abs(a), device)
    check(builtins.abs(x), np.abs(a), device)


@pytest.mark.parametrize('dtype', ['float32', 'int32'])
@pytest.mark.parametrize('shape,axis', [((), None), ((), 0), ((), -1), ((), ()),
    ((2, 3, 4), None), ((2, 3, 4), 0), ((2, 3, 4), -1), ((2, 3, 4), (0, 2)),
    ((2, 3, 4), (-1, 0)), ((2, 3, 4), ()), ((0, 3), 1), ((2, 0, 3), 0)])
@pytest.mark.parametrize('keepdims', [False, True])
def test_min_parity(device, dtype, shape, axis, keepdims):
    a = np.random.default_rng(42).integers(-500, 500, shape).astype(dtype)
    x = cx.tensor(a, device=device)
    expected = np.min(a, axis=axis, keepdims=keepdims)
    check(x.min(axis, keepdims), expected, device)
    check(cx.min(cx.tensor(a), axis, keepdims), expected, 'cpu')


@pytest.mark.parametrize('dtype', ['float32', 'int32'])
@pytest.mark.parametrize('shape,axis', [((), None), ((), 0), ((), -1), ((2, 3, 4), None),
    ((2, 3, 4), 0), ((2, 3, 4), 1), ((2, 3, 4), -1), ((0, 3), 1), ((2, 0, 3), 0)])
@pytest.mark.parametrize('keepdims', [False, True])
def test_argmax_parity(device, dtype, shape, axis, keepdims):
    a = np.random.default_rng(73).integers(-3, 4, shape).astype(dtype)
    expected = np.asarray(np.argmax(a, axis=axis, keepdims=keepdims), dtype=np.int32)
    check(cx.tensor(a, device=device).argmax(axis, keepdims), expected, device)
    check(cx.argmax(cx.tensor(a), axis, keepdims), expected, 'cpu')


@pytest.mark.parametrize('axis', [None, 0, 1, -1])
def test_reduction_ieee_ties(device, axis):
    a = np.array([[0., -0., 1e-40, -1e-40, np.inf], [np.nan, 2., np.nan, -np.inf, 2.]], np.float32)
    x = cx.tensor(a, device=device)
    check(x.min(axis), np.min(a, axis=axis), device)
    check(x.argmax(axis), np.asarray(np.argmax(a, axis=axis), np.int32), device)


@pytest.mark.parametrize('dtype', ['float32', 'int32'])
@pytest.mark.parametrize('bounds', [(-2, 3), (None, 3), (-2, None), (4, -4), (0, 0)])
@pytest.mark.parametrize('shape', [(), (0,), (2, 3, 4)])
def test_clip_scalar_parity(device, dtype, bounds, shape):
    a = np.random.default_rng(75).integers(-10, 10, shape).astype(dtype)
    x = cx.tensor(a, device=device)
    expected = np.asarray(np.clip(a, *bounds), dtype=dtype)
    check(x.clip(*bounds), expected, device)
    check(cx.clip(cx.tensor(a), *bounds), expected, 'cpu')


@pytest.mark.parametrize('dtype', ['float32', 'int32'])
def test_clip_broadcast_bounds(device, dtype):
    a = np.array([[[-4], [1], [9]]], dtype=dtype)
    lo = np.array([[-1, 2, 7, 8]], dtype=dtype)
    hi = np.array([[[3]], [[10]]], dtype=dtype)
    x, lower, upper = [cx.tensor(v, device=device) for v in (a, lo, hi)]
    result = cx.clip(x, lower, upper)
    check(result, np.clip(a, lo, hi), device)
    assert not _core._shares_storage(result._impl, x._impl)


def test_clip_ieee(device):
    a = np.array([np.nan, -0., 0., 1e-40, -1e-40, np.inf, -np.inf, 1., 2.], np.float32)
    lo = np.array([-1., 0., -0., -2e-40, -2e-40, -1., -1., np.nan, -1.], np.float32)
    hi = np.array([1., 0., -0., 2e-40, 2e-40, 1., 1., 2., np.nan], np.float32)
    result = cx.clip(*[cx.tensor(v, device=device) for v in (a, lo, hi)])
    check(result, np.clip(a, lo, hi), device)
    np.testing.assert_array_equal(result.numpy()[1:7].view(np.uint32), np.clip(a, lo, hi)[1:7].view(np.uint32))


def topk_reference(a, k, axis, largest, sorted):
    # Stable NumPy ordering with NaNs greatest; negate float64 to avoid int32 overflow.
    key = a.astype(np.float64)
    if largest:
        key = -key
        key[np.isnan(key)] = -np.inf
        # NaNs precede +inf as required, using a separate lexicographic key.
        ids = np.argsort(key, axis=axis, stable=True)
        nan_order = np.argsort(~np.take_along_axis(np.isnan(a), ids, axis=axis), axis=axis, stable=True)
        ids = np.take_along_axis(ids, nan_order, axis=axis)
    else:
        ids = np.argsort(key, axis=axis, stable=True)
    ids = np.take(ids, range(k), axis=axis)
    if not sorted:
        ids = np.sort(ids, axis=axis)
    return np.take_along_axis(a, ids, axis=axis), ids.astype(np.int32)


@pytest.mark.parametrize('dtype', ['float32', 'int32'])
@pytest.mark.parametrize('shape,axis,k', [((7,), -1, 0), ((7,), 0, 1), ((7,), -1, 7),
    ((2, 5, 3), 1, 3), ((2, 5, 3), 0, 2), ((2, 5, 3), -1, 2),
    ((0, 5, 3), 1, 2), ((2, 0, 3), 1, 0), ((2, 513), -1, 7)])
@pytest.mark.parametrize('largest,sorted', [(True, True), (False, True), (True, False), (False, False)])
def test_topk_parity(device, dtype, shape, axis, k, largest, sorted):
    a = np.random.default_rng(947).integers(-5, 6, shape).astype(dtype)
    x = cx.tensor(a, device=device)
    values, indices = x.topk(k, axis, largest, sorted)
    expected, ids = topk_reference(a, k, axis, largest, sorted)
    check(values, expected, device); check(indices, ids, device)
    cv, ci = cx.topk(cx.tensor(a), k, axis, largest, sorted)
    check(cv, expected, 'cpu'); check(ci, ids, 'cpu')
    assert not _core._shares_storage(x._impl, values._impl)
    assert not _core._shares_storage(values._impl, indices._impl)
    np.testing.assert_array_equal(x.numpy(), a)


@pytest.mark.parametrize('largest,sorted', [(True, True), (False, True), (True, False), (False, False)])
def test_topk_extremes(device, largest, sorted):
    for a in [np.array([np.inf, np.nan, 0., -0., -np.inf, 1e-40, -1e-40, np.nan, 1., 1.], np.float32),
              np.array([-2**31, 2**31-1, 16777217, 16777216, 0, 0], np.int32)]:
        for k in (1, len(a)-1, len(a)):
            v, i = cx.topk(cx.tensor(a, device=device), k, largest=largest, sorted=sorted)
            ev, ei = topk_reference(a, k, -1, largest, sorted)
            check(v, ev, device);check(i, ei, device)
            np.testing.assert_array_equal(v.numpy().view(np.uint32), a[ei].view(np.uint32))


def test_math_no_host_fallback_and_high_rank(device, monkeypatch):
    x = cx.tensor([[1., 4., 9.], [16., 25., 36.]], device=device)
    lo, hi = cx.tensor(2., device=device), cx.tensor(20., device=device)
    def forbidden(*args, **kwargs):
        raise AssertionError('math exported an input tensor to host')
    with monkeypatch.context() as guard:
        guard.setattr(cx.Tensor, 'numpy', forbidden);guard.setattr(cx.Tensor, 'cpu', forbidden)
        values = [cx.log(x), cx.sqrt(x), cx.abs(x), cx.min(x), cx.argmax(x), cx.clip(x, lo, hi), *cx.topk(x, 2)]
    assert builtins.all(v.device == device for v in values)
    high = x.reshape((1,)*300 + (2, 3))
    check(high.min().reshape(()), np.array(1., np.float32), device)
    v, i = high.topk(2)
    assert v.shape == (1,)*300 + (2, 2)
    check(i.reshape((2, 2)), np.array([[2, 1], [2, 1]], np.int32), device)
    if device != 'cpu':
        with pytest.raises(ValueError, match='device'):
            cx.clip(x, lo.cpu(), hi)


def test_math_errors_and_empty_validation(device):
    x = cx.tensor([[1., 2.]], device=device)
    for name in ['log', 'sqrt', 'abs', 'min', 'argmax', 'clip', 'topk']:
        with pytest.raises(TypeError):
            getattr(cx, name)([1, 2])
    for shape in [(2, 0), (0, 0)]:
        empty = cx.empty(shape, device=device)
        for name in ['min', 'argmax']:
            with pytest.raises(ValueError): getattr(cx, name)(empty, axis=1)
    for name in ['log', 'sqrt']:
        with pytest.raises(ValueError, match='float32'):
            getattr(cx, name)(cx.tensor(np.array([], np.int32), device=device))
    boolean = cx.tensor(np.array([], bool), device=device)
    for name in ['log', 'sqrt', 'abs', 'min', 'argmax']:
        with pytest.raises(ValueError): getattr(cx, name)(boolean)
    with pytest.raises(ValueError): cx.topk(boolean, 0)
    with pytest.raises(ValueError): cx.clip(boolean, False, True)
    for axis in [2, -3, (0, 1), True, 1.5]:
        with pytest.raises((ValueError, TypeError)): cx.argmax(x, axis)
    for k in [-1, 3, 1.5, True, 2**70]:
        with pytest.raises((ValueError, TypeError, OverflowError)): cx.topk(x, k)
    with pytest.raises(ValueError): cx.topk(cx.tensor(1., device=device), 1)
    with pytest.raises(ValueError): cx.clip(x)
    with pytest.raises(ValueError): cx.clip(x, cx.tensor(np.array(1, np.int32), device=device), 2)
    with pytest.raises(ValueError): cx.clip(x, cx.ones((3,), device=device), 2)
    for name in ['min', 'argmax']:
        with pytest.raises(TypeError): getattr(cx, name)(x, keepdims=1)
    with pytest.raises(TypeError): cx.topk(x, 1, largest=1)
    with pytest.raises(TypeError): cx.topk(x, 1, sorted=1)
    huge = cx.empty((0, 2**31), device=device)
    with pytest.raises(ValueError, match='int32'): cx.argmax(huge, axis=1)
    with pytest.raises(ValueError, match='int32'): cx.topk(huge, 0)
    with pytest.raises(ValueError): cx.min(x, (0, 0))


def test_min_signed_zero_first_value(device):
    bits = np.array([[0, 0x80000000], [0x80000000, 0]], np.uint32)
    x = cx.tensor(bits.view(np.float32), device=device)
    np.testing.assert_array_equal(x.min(axis=1).numpy().view(np.uint32), bits[:, 0])
    np.testing.assert_array_equal(x.argmax(axis=1).numpy(), [0, 0])
    for largest in (True, False):
        v, i = x.topk(2, largest=largest)
        np.testing.assert_array_equal(v.numpy().view(np.uint32), bits)
        np.testing.assert_array_equal(i.numpy(), [[0, 1], [0, 1]])


def test_clip_empty_and_topk_copy(device):
    for dtype in ('float32', 'int32'):
        x = cx.tensor(np.empty((0, 2, 1), dtype=dtype), device=device)
        lo = cx.tensor(np.zeros((1, 3), dtype=dtype), device=device)
        check(cx.clip(x, lo, 4), np.empty((0, 2, 3), dtype=dtype), device)
        values = np.array([[3, 3, 1, 2]], dtype=dtype)
        x = cx.tensor(values, device=device)
        v, i = x.topk(3)
        exported_v, exported_i = v.numpy(), i.numpy()
        exported_v[...] = 0; exported_i[...] = 0
        check(v, np.array([[3, 3, 2]], dtype=dtype), device)
        check(i, np.array([[0, 1, 3]], dtype=np.int32), device)
        check(x, values, device)
