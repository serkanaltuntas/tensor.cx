"""Public multi-axis reductions, shared by all device acceptance gates."""
import gc
import warnings

import numpy as np
import pytest

import tensorcx as cx
from tensorcx import _core


@pytest.fixture(params=cx.devices())
def device_name(request):
    return request.param


CASES = [
    ((2, 3, 4), None), ((2, 3, 4), ()), ((2, 3, 4), (0, 2)),
    ((2, 3, 4), (2, 0)), ((2, 3, 4), (-1, 0)), ((2, 3, 4), (0, 1, 2)),
    ((2, 3, 4), (0, 1)), ((2, 3, 4), (1, 2)), ((2, 3, 4), (1,)),
    ((2, 3, 4), -1), ((1, 1, 1), (0, 2)), ((257, 2, 3), (1, 2)),
    ((2, 3, 257), (0, 2)), ((0, 2, 3), (1, 2)), ((2, 0, 3), (0, 2)),
    ((2, 0, 3), (1, 2)), ((2, 0, 3), None), ((2, 0, 3), ()),
    ((0, 2, 0), (0, 2)), ((0,), None), ((), None), ((), ()), ((), 0), ((), -1),
]


@pytest.mark.parametrize("name", ["sum", "max", "mean"])
@pytest.mark.parametrize("dtype", [cx.float32, cx.int32])
@pytest.mark.parametrize("shape,axis", CASES)
def test_multi_axis_numpy_cpu_parity(device_name, name, dtype, shape, axis):
    values = np.resize(np.array([-5, -1, 0, 2, 7], dtype=dtype), shape)
    x = cx.tensor(values, device=device_name)
    if dtype == cx.int32 and (device_name == "cuda" or name == "mean"):
        for keepdims in (False, True):
            with pytest.raises(ValueError, match="float32"):
                getattr(x, name)(axis, keepdims=keepdims)
        return
    reference_axis = None if not shape and axis in (0, -1) else axis
    for keepdims in (False, True):
        kwargs = {"axis": reference_axis, "keepdims": keepdims}
        if name == "sum":
            kwargs["dtype"] = dtype  # tensor.cx keeps int32 and wraps, unlike NumPy's default.
        try:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                expected = getattr(np, name)(values, **kwargs)
        except ValueError:
            with pytest.raises(ValueError):
                getattr(x, name)(axis, keepdims=keepdims)
            continue
        for y in (getattr(x, name)(axis, keepdims=keepdims),
                  getattr(cx, name)(x, axis=axis, keepdims=np.bool_(keepdims))):
            assert y.shape == expected.shape and y.dtype == dtype and y.device == device_name
            assert not _core._shares_storage(x._impl, y._impl)
            np.testing.assert_allclose(y.numpy(), expected, rtol=1e-5, atol=1e-5, equal_nan=True)
            cx.testing.assert_allclose(y, getattr(cx.tensor(values), name)(axis, keepdims),
                                       rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(x.numpy(), values)


@pytest.mark.parametrize("name", ["sum", "max", "mean"])
def test_multi_axis_default_and_sequence_protocol(device_name, name):
    class Axis:
        def __index__(self):
            return 0
    x = cx.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4), device=device_name)
    np.testing.assert_array_equal(getattr(x, name)().numpy(), getattr(x, name)(None).numpy())
    np.testing.assert_array_equal(getattr(cx, name)(x).numpy(), getattr(x, name)(None).numpy())
    expected = getattr(x, name)((0, 2)).numpy()
    for axes in ([Axis(), np.int32(2)], iter([2, 0]), np.array([0, 2])):
        np.testing.assert_array_equal(getattr(x, name)(axes).numpy(), expected)
    np.testing.assert_array_equal(getattr(x, name)(Axis()).numpy(), getattr(x, name)(0).numpy())


@pytest.mark.parametrize("name", ["sum", "max", "mean"])
def test_multi_axis_validation_even_for_empty_inputs(device_name, name):
    for shape in ((2, 3, 4), (2, 0, 4)):
        x = cx.empty(shape, device=device_name)
        for axes in (True, np.bool_(True), 0.5, "0", b"0", (0, 0), (0, -3),
                     (3,), (-4,), (0, True), (0, 2.0), (2**64,), (-(2**64),)):
            with pytest.raises(ValueError):
                getattr(x, name)(axes)
        for keepdims in (1, None, "yes", []):
            with pytest.raises(TypeError, match="keepdims"):
                getattr(x, name)((0, 2), keepdims=keepdims)
        for axes in ((0, 0), (0, -3), (3,), (-4,), (True,)):
            with pytest.raises(ValueError):
                getattr(_core, f"_{name}_axes")(x._impl, axes)
    scalar = cx.tensor(1.0, device=device_name)
    for axes in ((0,), (-1,), 1, -2):
        with pytest.raises(ValueError):
            getattr(scalar, name)(axes)
    with pytest.raises(TypeError, match="Tensor"):
        getattr(cx, name)([1.0], axis=None)


@pytest.mark.parametrize("name", ["sum", "max", "mean"])
def test_multi_axis_empty_selection_copies_stored_bits(device_name, name):
    bits = np.array([0x80000000, 0x7fc12345, 1, 0x7f800000], dtype=np.uint32)
    x = cx.tensor(bits.view(np.float32), device=device_name)
    stored = x.numpy().view(np.uint32)
    y = getattr(x, name)(())
    np.testing.assert_array_equal(y.numpy().view(np.uint32), stored)
    assert not _core._shares_storage(x._impl, y._impl)
    del x
    gc.collect()
    np.testing.assert_array_equal(y.numpy().view(np.uint32), stored)


@pytest.mark.parametrize("name", ["sum", "max", "mean"])
def test_multi_axis_special_values_and_canonical_order(device_name, name):
    for values in ([[0., -0.], [-0., 0.]], [[np.inf, 2], [-np.inf, 4]],
                   [[1, np.nan], [3, 4]], [[3e38, 3e38], [-3e38, -3e38]],
                   [[1e20, 1], [-1e20, 3]]):
        data = np.array(values, dtype=np.float32)
        x = cx.tensor(data, device=device_name)
        # Flattened CPU reduction is the same one-pass row-major ordering.
        expected = getattr(cx.tensor(data.reshape(-1)), name)(0).numpy()
        for axes in (None, (0, 1), (1, 0), (-1, -2)):
            actual = getattr(x, name)(axes).numpy()
            np.testing.assert_allclose(actual, expected, equal_nan=True)
            if expected == 0:
                assert np.signbit(actual) == np.signbit(expected)
    if name == "mean":
        data = cx.tensor([[3e38, 3e38], [-3e38, -3e38]], device=device_name)
        # Sequential means produce NaN here; one sum then one division gives +inf.
        assert np.isposinf(data.mean((0, 1)).numpy())


def test_multi_axis_int32_wrap_and_dtype_rejection(device_name):
    data = np.array([[2**31-1, 2**31-1], [-2**31, -17]], dtype=np.int32)
    x = cx.tensor(data, device=device_name)
    for axes in (None, (), (0, 1)):
        if device_name == "cuda":
            for name in ("sum", "max"):
                with pytest.raises(ValueError, match="float32"):
                    getattr(x, name)(axes)
        else:
            np.testing.assert_array_equal(x.sum(axes).numpy(), data.sum(axis=axes, dtype=np.int32))
            np.testing.assert_array_equal(x.max(axes).numpy(), data.max(axis=axes))
        with pytest.raises(ValueError, match="float32"):
            x.mean(axes)


@pytest.mark.parametrize("name", ["sum", "max", "mean"])
def test_multi_axis_high_rank_and_empty_metadata(device_name, name):
    x = cx.tensor(np.arange(6, dtype=np.float32), device=device_name).reshape((2,) + (1,) * 298 + (3,))
    y = getattr(x, name)((299, 0), keepdims=True)
    assert y.shape == (1,) * 300
    expected = {"sum": 15., "max": 5., "mean": 2.5}[name]
    assert y.reshape(()).numpy() == expected
    # A selected iteration space can exceed int64 when there are no outputs.
    empty = cx.empty((0, 1, 2**62, 4, 0), device=device_name)
    assert getattr(empty, name)((1, 2, 3)).shape == (0, 0)
    if name == "max":
        with pytest.raises(ValueError):
            getattr(empty, name)(None)
    else:
        value = getattr(empty, name)(None).numpy()
        assert (value == 0) if name == "sum" else np.isnan(value)
    # Removing zero axes must still validate the actual output shape/strides.
    with pytest.raises(ValueError, match="overflow"):
        getattr(empty, name)((0, 4))


def test_multi_axis_composition_stays_native(device_name, monkeypatch):
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    x = cx.tensor(values, device=device_name)
    def forbidden(*args, **kwargs):
        raise AssertionError("reduction unexpectedly exported input data")
    with monkeypatch.context() as patch:
        patch.setattr(cx.Tensor, "numpy", forbidden)
        patch.setattr(cx.Tensor, "cpu", forbidden)
        centered = x - x.mean((0, 2), keepdims=True)
        result = centered.transpose((1, 0, 2)).sum((1, 2)).expand_dims(1)
    np.testing.assert_allclose(centered.numpy(), values - values.mean(axis=(0, 2), keepdims=True))
    np.testing.assert_array_equal(result.numpy(), np.zeros((3, 1), dtype=np.float32))
