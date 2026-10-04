"""Basic indexing and joining: NumPy oracle, ownership and native validation."""
import gc

import numpy as np
import pytest

import tensorcx as cx
from tensorcx import _core


@pytest.fixture(params=cx.devices())
def device_name(request):
    return request.param


@pytest.mark.parametrize("dtype", [cx.float32, cx.int32])
@pytest.mark.parametrize("shape,key", [
    ((3, 4, 5), 1), ((3, 4, 5), -1), ((3, 4, 5), (1, -1, 2)),
    ((3, 4, 5), (slice(None), slice(1, 4, 2), slice(None, None, -1))),
    ((3, 4), (None, Ellipsis, -1, None)), ((3, 4), (Ellipsis, None)),
    ((3, 4), (slice(None, None, -1), slice(3, 0, -2))),
    ((3, 4), (slice(99), slice(-99, 99))), ((3, 4), (slice(2, 1),)),
    ((0, 3), (slice(None, None, -1), 1)), ((2, 0, 3), (1, Ellipsis, None)),
    ((3,), (slice(None, None, 10**100),)), ((3,), (slice(None, None, -(10**100)),)),
    ((3,), (slice(-(10**100), 10**100),)), ((3,), np.int64(1)),
    ((), ()), ((), Ellipsis), ((), (None, Ellipsis, None)),
    ((3, 4), ()), ((3, 4), (1, Ellipsis, 2)),
])
def test_indexing_numpy_parity(device_name, dtype, shape, key):
    values = np.arange(np.prod(shape, dtype=int), dtype=dtype).reshape(shape)
    x = cx.tensor(values, device=device_name)
    y = x[key]
    expected = values[key]
    assert isinstance(y, cx.Tensor)
    assert y.shape == expected.shape and y.device == device_name and y.dtype == dtype
    assert not _core._shares_storage(x._impl, y._impl)
    np.testing.assert_array_equal(y.numpy(), expected)
    np.testing.assert_array_equal(y.numpy(), cx.tensor(values)[key].numpy())
    np.testing.assert_array_equal(y.reshape((-1,)).numpy(), np.asarray(expected).reshape(-1))
    np.testing.assert_array_equal(x.numpy(), values)
    del x
    gc.collect()
    np.testing.assert_array_equal(y.numpy(), expected)


@pytest.mark.parametrize("axis", [0, 1, -1, -2, np.int64(1)])
@pytest.mark.parametrize("dtype", [cx.float32, cx.int32])
@pytest.mark.parametrize("sizes", [(2, 3, 1), (0, 3, 0), (0, 0)])
def test_concat_numpy_parity(device_name, dtype, axis, sizes):
    arrays = [np.arange(4 * size, dtype=dtype).reshape((size, 4) if axis in (0, -2) else (4, size))
              for size in sizes]
    inputs = [cx.tensor(a, device=device_name) for a in arrays]
    result = cx.concat(iter(inputs), axis)
    expected = np.concatenate(arrays, axis)
    assert result.device == device_name and result.dtype == dtype
    np.testing.assert_array_equal(result.numpy(), expected)
    np.testing.assert_array_equal(result.numpy(), cx.concat([cx.tensor(a) for a in arrays], axis).numpy())
    assert all(not _core._shares_storage(x._impl, result._impl) for x in inputs)


@pytest.mark.parametrize("shape", [(), (0,), (2, 3), (2, 0, 3)])
@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize("dtype", [cx.float32, cx.int32])
def test_stack_numpy_parity(device_name, dtype, shape, axis):
    values = np.arange(np.prod(shape, dtype=int), dtype=dtype).reshape(shape)
    x = cx.tensor(values, device=device_name)
    result = cx.stack([x, x, x], axis)
    np.testing.assert_array_equal(result.numpy(), np.stack([values] * 3, axis))
    np.testing.assert_array_equal(result.numpy(), cx.stack([cx.tensor(values)] * 3, axis).numpy())
    assert not _core._shares_storage(x._impl, result._impl)
    if shape:
        single = cx.concat([x])
        assert not _core._shares_storage(x._impl, single._impl)
        np.testing.assert_array_equal(single.numpy(), values)


@pytest.mark.parametrize("sections", [1, 2, 4, np.int64(2), [], [1, 3], [-2], [3, 1],
                                       [-99, 99], [2, 2], [10**100], [-(10**100)]])
@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize("dtype", [cx.float32, cx.int32])
def test_split_numpy_parity(device_name, dtype, sections, axis):
    values = np.arange(16, dtype=dtype).reshape(4, 4)
    x = cx.tensor(values, device=device_name)
    expected = np.split(values, sections, axis)
    for outputs in (cx.split(x, sections, axis), x.split(sections, axis)):
        assert isinstance(outputs, tuple) and len(outputs) == len(expected)
        for i, (part, oracle) in enumerate(zip(outputs, expected)):
            assert part.dtype == dtype and part.device == device_name
            assert not _core._shares_storage(x._impl, part._impl)
            assert all(not _core._shares_storage(other._impl, part._impl) for other in outputs[:i])
            np.testing.assert_array_equal(part.numpy(), oracle)


def test_index_copy_preserves_bits_and_avoids_host_export(device_name, monkeypatch):
    bits = np.array([0, 0x80000000, 0x7fc12345, 0xffc45678, 1, 0x7f800000], dtype=np.uint32)
    x = cx.tensor(bits.view(np.float32), device=device_name)
    stored = x.numpy().view(np.uint32)
    def forbidden(*args, **kwargs):
        raise AssertionError("operation exported tensor to host")
    with monkeypatch.context() as patch:
        patch.setattr(cx.Tensor, "numpy", forbidden)
        patch.setattr(cx.Tensor, "cpu", forbidden)
        sliced = x[::-1]
        joined = cx.concat([x, sliced])
        stacked = cx.stack([x, x])
        parts = joined.split(2)
    np.testing.assert_array_equal(sliced.numpy().view(np.uint32), stored[::-1])
    np.testing.assert_array_equal(joined.numpy().view(np.uint32), np.concatenate([stored, stored[::-1]]))
    np.testing.assert_array_equal(stacked.numpy().view(np.uint32), np.stack([stored, stored]))
    np.testing.assert_array_equal(parts[1].numpy().view(np.uint32), stored[::-1])


def test_high_rank_and_empty_metadata(device_name):
    x = cx.ones((1,) * 300, device=device_name)
    assert x[(slice(None),) * 300].shape == x.shape
    assert cx.stack([x, x], -1).shape == (1,) * 300 + (2,)
    empty = cx.empty((2, 0, 4), device=device_name)
    assert [v.shape for v in empty.split(3, 1)] == [(2, 0, 4)] * 3
    large = cx.empty((2**62, 0, 2), device=device_name)
    assert large[::-1].shape == large.shape
    # Concatenated axis metadata must remain valid even with no allocation.
    with pytest.raises(ValueError, match="overflow"):
        cx.concat([large, large])


@pytest.mark.parametrize("key", [3, -4, (0, 0, 0), (Ellipsis, Ellipsis), True, np.bool_(False),
                                  [1], np.array([1]), np.array(True), "1", 1.5])
def test_invalid_index(device_name, key):
    with pytest.raises(IndexError):
        cx.ones((3, 4), device=device_name)[key]


@pytest.mark.parametrize("key", [slice(None, None, 0), slice(True), slice(0.5), slice(None, None, False)])
def test_invalid_slice(device_name, key):
    with pytest.raises(ValueError):
        cx.ones((3,), device=device_name)[key]


@pytest.mark.parametrize("starts,steps,lengths", [
    ([], [1], [1]), ([0], [], [1]), ([0], [1], []),
    ([0], [0], [0]), ([-1], [1], [1]), ([3], [1], [1]),
    ([0], [1], [4]), ([2], [-1], [4]), ([0], [1], [-1]),
    ([0], [2**63 - 1], [2]), ([2], [-(2**63)], [2]),
])
def test_native_slice_rejects_bad_metadata(device_name, starts, steps, lengths):
    x = cx.ones((3,), device=device_name)
    with pytest.raises(ValueError):
        _core._slice(x._impl, starts, steps, lengths)


@pytest.mark.parametrize("axis", [True, 0.5, "0", 2, -3, 2**100])
def test_join_split_bad_axis(device_name, axis):
    x = cx.ones((2, 3), device=device_name)
    for call in (lambda: cx.concat([x, x], axis), lambda: x.split(1, axis)):
        with pytest.raises(ValueError): call()


def test_joining_validation(device_name):
    x = cx.ones((2, 3), device=device_name)
    for inputs in ([], [x, 1], None, [x, x.astype(cx.int32)]):
        for op in (cx.concat, cx.stack):
            with pytest.raises(ValueError): op(inputs)
    for op in (cx.concat, cx.stack):
        with pytest.raises(ValueError): op([x, cx.ones((2, 4), device=device_name)])
    with pytest.raises(ValueError): cx.concat([x, x.reshape((6,))])
    with pytest.raises(ValueError): cx.concat([cx.tensor(1, device=device_name)])
    with pytest.raises(ValueError): _core._concat(x._impl, [x.astype(cx.int32)._impl], 0)
    with pytest.raises(ValueError): _core._concat(x._impl, [x.reshape((6,))._impl], 0)
    for bad in (0, -1, 3, True, 2.5, [1.5], [True], "12"):
        with pytest.raises(ValueError): x.split(bad)
    if device_name != "cpu":
        with pytest.raises(ValueError): cx.concat([x, x.cpu()])
        with pytest.raises(ValueError): _core._concat(x._impl, [x.cpu()._impl], 0)


def test_random_slices(device_name):
    rng = np.random.default_rng(384)
    values = rng.standard_normal((5, 6, 7), dtype=np.float32)
    x = cx.tensor(values, device=device_name)
    for _ in range(60):
        key = tuple(slice(int(rng.integers(-10, 10)), int(rng.integers(-10, 10)),
                          int(rng.choice([-4, -2, -1, 1, 2, 3]))) for _ in range(3))
        np.testing.assert_array_equal(x[key].numpy(), values[key])
