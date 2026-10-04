"""NumPy matmul shape semantics, native batch broadcasting and ownership."""
import numpy as np
import pytest

import tensorcx as cx
from tensorcx import _core


@pytest.fixture(params=[(d, p) for d in cx.devices() for p in cx.matmul_backends(d)])
def route(request):
    return request.param


CASES = [
    ((3,), (3,)), ((2, 3), (3,)), ((3,), (3, 4)),
    ((2, 3, 4), (2, 4, 5)), ((2, 3, 4), (4, 5)),
    ((3, 4), (2, 4, 5)), ((4,), (2, 4, 5)), ((2, 3, 4), (4,)),
    ((2, 1, 3, 4), (1, 5, 4, 6)), ((1, 2, 3, 4), (3, 1, 4, 5)),
    ((2, 1, 1, 3, 4), (1, 5, 1, 4, 2)),
    ((2, 17, 19), (1, 19, 33)), ((2, 33, 17), (2, 17, 1)),
    ((0,), (0,)), ((2, 0), (0,)), ((0,), (0, 3)),
    ((2, 3, 0), (1, 0, 4)), ((0, 3, 4), (1, 4, 2)),
    ((1, 3, 4), (0, 4, 2)), ((2, 0, 4), (2, 4, 3)),
    ((2, 3, 4), (2, 4, 0)), ((0, 2, 3, 0), (1, 2, 0, 4)),
    ((2, 1, 0, 3), (1, 5, 3, 4)), ((3,), (0, 3, 4)),
    ((0, 2, 3), (3,)), ((2, 1, 3, 4), (5, 4, 2)),
]


@pytest.mark.parametrize('shapes', CASES)
def test_batched_matmul_numpy_cpu_parity(route, shapes):
    device, preference = route
    rng = np.random.default_rng(741)
    a, b = [rng.uniform(-1, 1, s).astype(np.float32) for s in shapes]
    x, y = [cx.tensor(v, device=device) for v in (a, b)]
    actual = cx.matmul(x, y, backend=preference)
    expected = np.matmul(a, b)
    assert actual.shape == expected.shape and actual.dtype == cx.float32 and actual.device == device
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(actual.numpy(), cx.matmul(cx.tensor(a), cx.tensor(b)).numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(x.numpy(), a)
    np.testing.assert_array_equal(y.numpy(), b)
    # Every result owns new storage and independent NumPy exports.
    assert not _core._shares_storage(actual._impl, x._impl)
    assert not _core._shares_storage(actual._impl, y._impl)
    exported = actual.numpy()
    exported[...] = 123
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-4, atol=1e-4)
    if preference == 'auto':
        np.testing.assert_allclose((x @ y).numpy(), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('shapes', [
    ((), (2, 2)), ((2, 2), ()), ((3,), (4,)),
    ((2, 3, 4), (3, 4, 5)), ((2, 3, 4), (2, 5, 6)),
    ((0, 3, 4), (2, 4, 5)), ((0, 3, 4), (1, 5, 6)),
])
def test_batched_matmul_invalid_shapes(route, shapes):
    device, preference = route
    a, b = [cx.empty(s, device=device) for s in shapes]
    with pytest.raises(ValueError):
        cx.matmul(a, b, backend=preference)


@pytest.mark.parametrize('dtype', [cx.int32, cx.bool])
def test_batched_matmul_rejects_dtype_even_empty(route, dtype):
    device, preference = route
    for shape in [(2, 3, 3), (0, 3, 3)]:
        a = cx.tensor(np.ones(shape, dtype=dtype), device=device)
        b = cx.tensor(np.ones(shape, dtype=np.float32), device=device)
        for left, right in [(a, a), (a, b), (b, a)]:
            with pytest.raises(ValueError, match='float32'):
                cx.matmul(left, right, backend=preference)


def test_batched_matmul_no_host_fallback(route, monkeypatch):
    device, preference = route
    x = cx.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4), device=device)
    y = cx.tensor(np.eye(4, dtype=np.float32), device=device)
    def forbidden(*args, **kwargs):
        raise AssertionError('matmul exported an input tensor to host')
    with monkeypatch.context() as guard:
        guard.setattr(cx.Tensor, 'numpy', forbidden)
        guard.setattr(cx.Tensor, 'cpu', forbidden)
        result = cx.matmul(x, y, backend=preference)
    np.testing.assert_array_equal(result.numpy(), np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    if device != 'cpu':
        with pytest.raises(ValueError, match='device mismatch'):
            cx.matmul(x, y.cpu(), backend=preference)


def test_batched_matmul_high_rank_and_overflow(route):
    device, preference = route
    a = cx.ones((1,) * 300 + (2, 3), device=device)
    b = cx.ones((3, 4), device=device)
    result = cx.matmul(a, b, backend=preference)
    assert result.shape == (1,) * 300 + (2, 4)
    np.testing.assert_array_equal(result.reshape((2, 4)).numpy(), np.full((2, 4), 3, np.float32))
    a = cx.empty((2**62, 0), device=device)
    b = cx.empty((0, 4), device=device)
    with pytest.raises(ValueError):
        cx.matmul(a, b, backend=preference)
    # Leading and trailing zeros make the unused batch product irrelevant.
    a = cx.empty((0, 2**62, 4, 0, 2), device=device)
    b = cx.empty((1, 1, 1, 2, 0), device=device)
    assert cx.matmul(a, b, backend=preference).shape == (0, 2**62, 4, 0, 0)


def test_batched_matmul_special_values(route):
    device, preference = route
    a = np.array([[[np.nan, 1]], [[np.inf, 1]], [[-np.inf, -1]], [[-0., 0.]]], np.float32)
    b = np.array([[1.], [2.]], np.float32)
    with np.errstate(invalid='ignore'):
        expected = a @ b
    actual = cx.matmul(cx.tensor(a, device=device), cx.tensor(b, device=device), backend=preference)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-4, atol=1e-4, equal_nan=True)


def test_batched_matmul_backend_rank_limit(route):
    device, preference = route
    a = cx.ones((2,) * 15 + (1, 1), device=device)
    b = cx.ones((1, 1), device=device)
    if device == 'metal' and preference == 'optimized':
        with pytest.raises(ValueError, match='16'):
            cx.matmul(a, b, backend=preference)
    else:
        result = cx.matmul(a, b, backend=preference)
        assert result.shape == a.shape
        np.testing.assert_array_equal(result.reshape((-1,)).numpy(), np.ones(2**15, np.float32))


def test_batched_matmul_grid_stride(route):
    device, preference = route
    # More batches than the CUDA grid cap, with distinct batch values.
    values = (np.arange(65537, dtype=np.float32) % 17).reshape(-1, 1, 1)
    a = cx.tensor(values, device=device)
    b = cx.tensor([[2.]], device=device)
    result = cx.matmul(a, b, backend=preference)
    np.testing.assert_array_equal(result.numpy(), values * 2)
