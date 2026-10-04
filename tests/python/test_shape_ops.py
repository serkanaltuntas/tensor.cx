"""Shape operations: native contiguous transpose and shared-storage views."""
import gc
import itertools

import numpy as np
import pytest

import tensorcx as cx
from tensorcx import _core


@pytest.fixture(params=cx.devices())
def device_name(request):
    return request.param


@pytest.mark.parametrize("dtype", [cx.float32, cx.int32])
@pytest.mark.parametrize("shape", [(), (1,), (257,), (2, 3), (3, 5, 7), (2, 1, 3, 4),
                                  (0,), (2, 0, 3), (1, 0)])
def test_transpose_numpy_cpu_parity_and_copy(device_name, dtype, shape):
    values = np.arange(np.prod(shape, dtype=int), dtype=dtype).reshape(shape)
    x = cx.tensor(values, device=device_name)
    permutations = {tuple(range(len(shape))), tuple(reversed(range(len(shape))))}
    if len(shape) == 3:
        permutations.update(itertools.permutations(range(3)))
    for axes in permutations:
        expected = values.transpose(axes)
        for y in (x.transpose(axes), cx.transpose(x, axes),
                  x.transpose(tuple(axis - len(shape) for axis in axes))):
            assert y.shape == expected.shape and y.dtype == dtype and y.device == device_name
            assert not _core._shares_storage(x._impl, y._impl)
            np.testing.assert_array_equal(y.numpy(), expected)
            cx.testing.assert_allclose(y, cx.tensor(values).transpose(axes))
            contiguous = np.ascontiguousarray(expected).reshape(expected.shape)
            # Empty NumPy arrays may use zero strides, unlike runtime metadata.
            if contiguous.size:
                assert y.strides == tuple(stride // contiguous.itemsize for stride in contiguous.strides)
    np.testing.assert_array_equal(x.T.numpy(), values.T)
    np.testing.assert_array_equal(cx.transpose(x).numpy(), values.T)
    np.testing.assert_array_equal(x.numpy(), values)
    y = x.T
    del x
    gc.collect()
    np.testing.assert_array_equal(y.numpy(), values.T)


def test_transpose_preserves_bits(device_name):
    bits = np.array([0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 1,
                     0xffc12345, 0x7f812345, 0x80000001], dtype=np.uint32).reshape(3, 3)
    x = cx.tensor(bits.view(np.float32), device=device_name)
    # Construction may quiet a signaling NaN. Reordering must preserve the
    # bits actually stored by the runtime, without further canonicalization.
    stored = x.numpy().view(np.uint32)
    np.testing.assert_array_equal(x.T.numpy().view(np.uint32), stored.T)
    np.testing.assert_array_equal(x.transpose((0, 1)).numpy().view(np.uint32), stored)


@pytest.mark.parametrize("shape,axis", [
    ((1, 2, 1, 3, 1), None), ((1, 2, 1, 3, 1), (0, -1)), ((1, 2, 1), 0),
    ((1, 2, 1), -1), ((1, 2, 1), [2, 0]), ((1, 1), None), ((2, 3), None),
    ((1, 0, 1), None), ((0,), None), ((1, 2), ()), ((), None), ((), ()),
    ((), 0), ((), -1), ((1,), np.int64(0)),
])
@pytest.mark.parametrize("dtype", [cx.float32, cx.int32])
def test_squeeze_numpy_and_shared_storage(device_name, dtype, shape, axis):
    values = np.arange(np.prod(shape, dtype=int), dtype=dtype).reshape(shape)
    x = cx.tensor(values, device=device_name)
    expected = np.squeeze(values, axis=tuple(axis) if isinstance(axis, list) else axis)
    for y in (x.squeeze(axis), cx.squeeze(x, axis)):
        assert y.shape == expected.shape and y.dtype == dtype and y.device == device_name
        assert _core._shares_storage(x._impl, y._impl)
        np.testing.assert_array_equal(y.numpy(), expected)
    assert x.shape == shape
    del x
    gc.collect()
    np.testing.assert_array_equal(y.numpy(), expected)


@pytest.mark.parametrize("shape,axis", [
    ((2, 3), 0), ((2, 3), 1), ((2, 3), -1), ((2, 3), -3),
    ((2, 3), (0, -1)), ((2, 3), (3, 0)), ((2, 3), [0, 2, -1]),
    ((2, 3), ()), ((0, 3), (0, 2)), ((1,), np.int32(1)),
    ((), 0), ((), -1), ((), ()), ((), (0, -1)),
])
@pytest.mark.parametrize("dtype", [cx.float32, cx.int32])
def test_expand_dims_numpy_and_shared_storage(device_name, dtype, shape, axis):
    values = np.arange(np.prod(shape, dtype=int), dtype=dtype).reshape(shape)
    x = cx.tensor(values, device=device_name)
    expected = np.expand_dims(values, axis)
    for y in (x.expand_dims(axis), cx.expand_dims(x, axis)):
        assert y.shape == expected.shape and y.dtype == dtype and y.device == device_name
        assert _core._shares_storage(x._impl, y._impl)
        np.testing.assert_array_equal(y.numpy(), expected)
    assert x.shape == shape
    del x
    gc.collect()
    np.testing.assert_array_equal(y.numpy(), expected)


def test_shape_ops_high_rank(device_name):
    # Rank 300 also exceeds Metal's 4 KiB inline metadata limit.
    shape = (1,) * 298 + (2, 3)
    x = cx.tensor([[1, 2, 3], [4, 5, 6]], device=device_name).reshape(shape)
    y = x.T
    assert y.shape == tuple(reversed(shape))
    np.testing.assert_array_equal(y.reshape((3, 2)).numpy(), [[1, 4], [2, 5], [3, 6]])
    z = y.squeeze().expand_dims(tuple(range(298)))
    assert z.shape == (1,) * 298 + (3, 2)
    assert _core._shares_storage(y._impl, z._impl)


def test_shape_ops_compose_without_host_exports(device_name, monkeypatch):
    x = cx.tensor(np.arange(6, dtype=np.float32).reshape(2, 3), device=device_name)
    weights = cx.tensor(np.arange(8, dtype=np.float32).reshape(2, 4), device=device_name)
    def forbidden(*args, **kwargs):
        raise AssertionError("shape operation exported tensor data")
    with monkeypatch.context() as patch:
        patch.setattr(cx.Tensor, "numpy", forbidden)
        patch.setattr(cx.Tensor, "cpu", forbidden)
        transposed = x.expand_dims((0, -1)).squeeze().T
        y = cx.matmul(transposed, weights)
        z = (transposed + 1).sum(axis=1, keepdims=True)
        copied = transposed.astype(cx.int32)
    expected = np.arange(6, dtype=np.float32).reshape(2, 3).T
    np.testing.assert_allclose(y.numpy(), expected @ weights.numpy())
    np.testing.assert_array_equal(z.numpy(), (expected + 1).sum(axis=1, keepdims=True))
    np.testing.assert_array_equal(copied.numpy(), expected.astype(np.int32))


def test_shape_ops_argument_validation(device_name):
    x = cx.tensor([[1.0, 2.0], [3.0, 4.0]], device=device_name)
    for function in (cx.transpose, cx.squeeze, cx.expand_dims):
        with pytest.raises(TypeError, match="Tensor"):
            function([1.0], 0)
    for axes in (True, np.bool_(True), "01", b"01", 1.5, [True, 1], [0.0, 1], [2**64, 1]):
        for operation in (x.transpose, x.squeeze, x.expand_dims):
            with pytest.raises(ValueError):
                operation(axes)
    for axes in ((), (0,), (0, 0), (0, -2), (0, 2), (-3, 1)):
        with pytest.raises(ValueError):
            x.transpose(axes)
        with pytest.raises(ValueError):
            _core.transpose(x._impl, axes)
    for axes in (0, -1, (0, 1), (0, 0), (2,), (-3,)):
        with pytest.raises(ValueError):
            x.squeeze(axes)
    singleton = x.expand_dims(0)
    for axes in ((0, 0), (0, -3)):
        with pytest.raises(ValueError):
            singleton.squeeze(axes)
    for axes in (None, 3, -4, (0, 0), (0, -4), (0, 4)):
        with pytest.raises(ValueError):
            x.expand_dims(axes)
    scalar = cx.tensor(1.0, device=device_name)
    for axes in ((0,), (-1,), 1, -2):
        with pytest.raises(ValueError):
            scalar.squeeze(axes)
    np.testing.assert_array_equal(x.T.numpy(), [[1, 3], [2, 4]])


def test_shape_ops_integer_protocol_and_iterables(device_name):
    class Axis:
        def __index__(self):
            return 0
    x = cx.tensor([[1.0, 2.0]], device=device_name)
    np.testing.assert_array_equal(x.transpose(iter([np.int64(1), Axis()])).numpy(), [[1], [2]])
    assert x.squeeze(Axis()).shape == (2,)
    assert x.expand_dims(Axis()).shape == (1, 1, 2)
    assert cx.tensor([1], device=device_name).transpose(Axis()).shape == (1,)
    assert x.squeeze(iter([0])).shape == (2,)
    assert x.expand_dims(iter([0, -1])).shape == (1, 1, 2, 1)


def test_transpose_empty_metadata_overflow_is_rejected(device_name):
    x = cx.empty((2**62, 0, 4), device=device_name)
    for axes in ((1, 0, 2), (0, 2, 1)):
        with pytest.raises(ValueError, match="overflow"):
            x.transpose(axes)
        with pytest.raises(ValueError, match="overflow"):
            _core.transpose(x._impl, axes)
    assert x.transpose((2, 1, 0)).shape == (4, 0, 2**62)
