import numpy as np
import pytest

import cortex_runtime as cx


HUGE_SHAPE = (3_037_000_500, 3_037_000_500)
STRIDE_OVERFLOW_SHAPE = (0, 9_223_372_036_854_775_807, 2)


def _gelu_reference(values):
    values = np.asarray(values, dtype=np.float32)
    inner = np.float32(0.7978845608028654) * (
        values + np.float32(0.044715) * values * values * values
    )
    return np.float32(0.5) * values * (
        np.float32(1.0) + np.tanh(inner)
    )


def _silu_reference(values):
    values = np.asarray(values, dtype=np.float32)
    return values / (np.float32(1.0) + np.exp(-values))


def test_tensor_infers_int32_for_integer_list():
    x = cx.tensor([1, 2, 3], device="cpu")

    assert x.shape == (3,)
    assert x.strides == (1,)
    assert x.dtype == cx.int32
    assert x.device == "cpu"
    assert x.nbytes == 12
    np.testing.assert_array_equal(x.numpy(), np.array([1, 2, 3], dtype=np.int32))


def test_tensor_supports_explicit_float32():
    x = cx.tensor([1, 2, 3], dtype=cx.float32, device="cpu")

    assert x.dtype == cx.float32
    np.testing.assert_allclose(x.numpy(), np.array([1, 2, 3], dtype=np.float32))


def test_zeros_ones_and_empty_metadata():
    zeros = cx.zeros((3,), dtype=cx.float32, device="cpu")
    ones = cx.ones((3,), dtype=cx.int32, device="cpu")
    empty = cx.empty((2,), dtype=cx.float32, device="cpu")

    np.testing.assert_allclose(zeros.numpy(), np.zeros((3,), dtype=np.float32))
    np.testing.assert_array_equal(ones.numpy(), np.ones((3,), dtype=np.int32))
    assert empty.shape == (2,)
    assert empty.dtype == cx.float32
    assert empty.nbytes == 8


def test_factory_rejects_shape_size_overflow():
    with pytest.raises(ValueError, match="shape size overflow"):
        cx.zeros(HUGE_SHAPE, dtype=cx.float32, device="cpu")


def test_factory_rejects_shape_stride_overflow():
    with pytest.raises(ValueError, match="shape stride overflow"):
        cx.zeros(STRIDE_OVERFLOW_SHAPE, dtype=cx.float32, device="cpu")


def test_numpy_preserves_tensor_shape():
    x = cx.zeros((2, 3), dtype=cx.float32, device="cpu")

    assert x.shape == (2, 3)
    np.testing.assert_allclose(x.numpy(), np.zeros((2, 3), dtype=np.float32))


def test_tensor_accepts_nested_rank2_data():
    x = cx.tensor([[1, 2, 3], [4, 5, 6]], dtype=cx.float32, device="cpu")

    assert x.shape == (2, 3)
    assert x.strides == (3, 1)
    np.testing.assert_allclose(
        x.numpy(),
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
    )


def test_add_and_multiply_cpu_int32():
    x = cx.tensor([1, 2, 3], device="cpu")
    y = cx.tensor([4, 5, 6], device="cpu")

    np.testing.assert_array_equal((x + y).numpy(), np.array([5, 7, 9], dtype=np.int32))
    np.testing.assert_array_equal((x * y).numpy(), np.array([4, 10, 18], dtype=np.int32))


def test_add_and_multiply_cpu_float32():
    x = cx.tensor([1.0, 2.0, 3.0], device="cpu")
    y = cx.ones((3,), dtype=cx.float32, device="cpu")

    np.testing.assert_allclose((x + y).numpy(), np.array([2, 3, 4], dtype=np.float32))
    np.testing.assert_allclose((x * y).numpy(), np.array([1, 2, 3], dtype=np.float32))


def test_sum_and_max_cpu_float32_match_numpy():
    x = cx.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=cx.float32, device="cpu")

    cx.testing.assert_allclose(cx.sum(x, axis=1), np.sum(x.numpy(), axis=1), kind="reduction")
    cx.testing.assert_allclose(x.sum(axis=0), np.sum(x.numpy(), axis=0), kind="reduction")
    cx.testing.assert_allclose(cx.max(x, axis=1), np.max(x.numpy(), axis=1), kind="reduction")
    cx.testing.assert_allclose(x.max(axis=0), np.max(x.numpy(), axis=0), kind="reduction")


def test_max_cpu_float32_propagates_nan_like_numpy():
    x = cx.tensor([[np.nan, -1.0], [1.0, 2.0]], dtype=cx.float32, device="cpu")

    cx.testing.assert_allclose(cx.max(x, axis=1), np.max(x.numpy(), axis=1), kind="reduction")


def test_mean_cpu_float32_matches_numpy():
    x = cx.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=cx.float32, device="cpu")

    cx.testing.assert_allclose(cx.mean(x, axis=1), np.mean(x.numpy(), axis=1), kind="reduction")
    cx.testing.assert_allclose(x.mean(axis=0), np.mean(x.numpy(), axis=0), kind="reduction")


def test_mean_cpu_empty_axis_returns_nan():
    x = cx.empty((2, 0), dtype=cx.float32, device="cpu")

    cx.testing.assert_allclose(
        cx.mean(x, axis=1),
        np.array([np.nan, np.nan], dtype=np.float32),
        kind="reduction",
    )


def test_exp_cpu_float32_matches_numpy():
    x = cx.tensor([-2.0, -0.5, 0.0, 1.0, 3.0], dtype=cx.float32, device="cpu")

    cx.testing.assert_allclose(cx.exp(x), np.exp(x.numpy()), kind="elementwise")
    cx.testing.assert_allclose(x.exp(), np.exp(x.numpy()), kind="elementwise")


def test_gelu_and_silu_cpu_float32_match_reference():
    x = cx.tensor([-4.0, -1.0, 0.0, 0.5, 2.0, 5.0], dtype=cx.float32, device="cpu")

    cx.testing.assert_allclose(cx.gelu(x), _gelu_reference(x.numpy()), kind="elementwise")
    cx.testing.assert_allclose(x.gelu(), _gelu_reference(x.numpy()), kind="elementwise")
    cx.testing.assert_allclose(cx.silu(x), _silu_reference(x.numpy()), kind="elementwise")
    cx.testing.assert_allclose(x.silu(), _silu_reference(x.numpy()), kind="elementwise")


def test_sum_and_max_cpu_support_negative_axis():
    x = cx.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[-1.0, -2.0], [5.0, 6.0]]],
        dtype=cx.float32,
        device="cpu",
    )

    cx.testing.assert_allclose(cx.sum(x, axis=-1), np.sum(x.numpy(), axis=-1), kind="reduction")
    cx.testing.assert_allclose(cx.max(x, axis=-2), np.max(x.numpy(), axis=-2), kind="reduction")
    cx.testing.assert_allclose(cx.mean(x, axis=-1), np.mean(x.numpy(), axis=-1), kind="reduction")


def test_sum_and_max_cpu_int32_exact():
    x = cx.tensor([[1, 2, 3], [4, 5, 6]], dtype=cx.int32, device="cpu")

    np.testing.assert_array_equal(cx.sum(x, axis=1).numpy(), np.array([6, 15], dtype=np.int32))
    np.testing.assert_array_equal(cx.max(x, axis=0).numpy(), np.array([4, 5, 6], dtype=np.int32))


def test_rank1_reduction_returns_rank0_scalar():
    x = cx.tensor([1, 2, 3], dtype=cx.int32, device="cpu")

    summed = cx.sum(x, axis=0)
    maximum = cx.max(x, axis=0)

    assert summed.shape == ()
    assert maximum.shape == ()
    assert summed.numpy().item() == 6
    assert maximum.numpy().item() == 3


def test_rank0_scalar_reduction_returns_scalar():
    x = cx.tensor(5, dtype=cx.int32, device="cpu")
    y = cx.tensor(5.0, dtype=cx.float32, device="cpu")

    summed = cx.sum(x, axis=0)
    maximum = cx.max(x, axis=-1)
    mean = cx.mean(y, axis=0)

    assert summed.shape == ()
    assert maximum.shape == ()
    assert mean.shape == ()
    assert summed.numpy().item() == 5
    assert maximum.numpy().item() == 5
    assert mean.numpy().item() == 5.0


def test_exp_rank0_scalar_returns_scalar():
    x = cx.tensor(2.0, dtype=cx.float32, device="cpu")
    actual = cx.exp(x)

    assert actual.shape == ()
    cx.testing.assert_allclose(actual, np.exp(x.numpy()), kind="elementwise")


def test_gelu_and_silu_rank0_scalar_return_scalar():
    x = cx.tensor(2.0, dtype=cx.float32, device="cpu")

    gelu = cx.gelu(x)
    silu = cx.silu(x)

    assert gelu.shape == ()
    assert silu.shape == ()
    cx.testing.assert_allclose(gelu, _gelu_reference(x.numpy()), kind="elementwise")
    cx.testing.assert_allclose(silu, _silu_reference(x.numpy()), kind="elementwise")


def test_unary_empty_tensor_returns_empty():
    x = cx.empty((0,), dtype=cx.float32, device="cpu")

    assert cx.exp(x).shape == (0,)
    assert cx.gelu(x).shape == (0,)
    assert cx.silu(x).shape == (0,)
    np.testing.assert_allclose(cx.exp(x).numpy(), np.array([], dtype=np.float32))
    np.testing.assert_allclose(cx.gelu(x).numpy(), np.array([], dtype=np.float32))
    np.testing.assert_allclose(cx.silu(x).numpy(), np.array([], dtype=np.float32))


def test_sum_cpu_int32_wraps_two_complement():
    x = cx.tensor([[2**31 - 1, 1]], dtype=cx.int32, device="cpu")

    np.testing.assert_array_equal(cx.sum(x, axis=1).numpy(), np.array([-(2**31)], dtype=np.int32))


def test_sum_empty_axis_returns_zero_and_max_rejects_empty_axis():
    x = cx.empty((2, 0), dtype=cx.float32, device="cpu")

    np.testing.assert_allclose(cx.sum(x, axis=1).numpy(), np.zeros((2,), dtype=np.float32))
    with pytest.raises(ValueError, match="max reduction requires a non-empty axis"):
        cx.max(x, axis=1)


def test_reductions_reject_invalid_axis_and_non_tensor_input():
    x = cx.ones((2, 3), dtype=cx.float32, device="cpu")

    with pytest.raises(ValueError, match="reduction axis is out of range"):
        cx.sum(x, axis=2)
    with pytest.raises(ValueError, match="reduction axis is out of range"):
        cx.max(x, axis=-3)
    with pytest.raises(TypeError, match="sum expects a Tensor argument"):
        cx.sum([1, 2, 3], axis=0)
    with pytest.raises(TypeError, match="max expects a Tensor argument"):
        cx.max([1, 2, 3], axis=0)
    with pytest.raises(TypeError, match="mean expects a Tensor argument"):
        cx.mean([1, 2, 3], axis=0)


def test_float32_only_ops_reject_int32_and_non_tensor_input():
    x = cx.tensor([1, 2, 3], dtype=cx.int32, device="cpu")

    with pytest.raises(ValueError, match="mean only supports float32"):
        cx.mean(x, axis=0)
    with pytest.raises(ValueError, match="exp only supports float32"):
        cx.exp(x)
    with pytest.raises(ValueError, match="gelu only supports float32"):
        cx.gelu(x)
    with pytest.raises(ValueError, match="silu only supports float32"):
        cx.silu(x)
    with pytest.raises(TypeError, match="exp expects a Tensor argument"):
        cx.exp([1, 2, 3])
    with pytest.raises(TypeError, match="gelu expects a Tensor argument"):
        cx.gelu([1, 2, 3])
    with pytest.raises(TypeError, match="silu expects a Tensor argument"):
        cx.silu([1, 2, 3])


def test_binary_ops_reject_shape_mismatch():
    x = cx.tensor([1, 2, 3], device="cpu")
    y = cx.tensor([1, 2], device="cpu")

    with pytest.raises(ValueError, match="shape mismatch"):
        _ = x + y


def test_binary_ops_reject_dtype_mismatch():
    x = cx.tensor([1, 2, 3], dtype=cx.int32, device="cpu")
    y = cx.tensor([1, 2, 3], dtype=cx.float32, device="cpu")

    with pytest.raises(ValueError, match="dtype mismatch"):
        _ = x * y


def test_matmul_cpu_float32_matches_numpy():
    x = cx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=cx.float32, device="cpu")
    y = cx.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=cx.float32, device="cpu")

    expected = x.numpy() @ y.numpy()

    np.testing.assert_allclose(cx.matmul(x, y).numpy(), expected)
    np.testing.assert_allclose((x @ y).numpy(), expected)


def test_matmul_cpu_rejects_invalid_inputs():
    x = cx.ones((2, 3), dtype=cx.float32, device="cpu")
    y = cx.ones((2, 3), dtype=cx.float32, device="cpu")
    z = cx.ones((3, 2), dtype=cx.int32, device="cpu")

    with pytest.raises(ValueError, match="matmul shape mismatch"):
        cx.matmul(x, y)
    with pytest.raises(ValueError, match="matmul only supports float32"):
        cx.matmul(x, z)
    with pytest.raises(ValueError, match="matmul requires rank-2"):
        cx.matmul(cx.ones((3,), dtype=cx.float32, device="cpu"), y)


def test_tensor_rejects_unsupported_device():
    with pytest.raises(ValueError, match="unsupported device transfer"):
        cx.tensor([1, 2, 3], device="cuda")


def test_tensor_rejects_int_out_of_int32_range():
    with pytest.raises(ValueError, match="out of range for int32"):
        cx.tensor([2**40])
    with pytest.raises(ValueError, match="out of range for int32"):
        cx.tensor([[2**40, 1], [2, 3]])


def test_scalar_input_preserves_rank0():
    assert cx.tensor(5).shape == ()
    assert cx.tensor(2.5).shape == ()
    assert cx.tensor(5).numpy().item() == 5


def test_fill_rejects_int32_value_out_of_range():
    from cortex_runtime import _core

    with pytest.raises(ValueError, match="fill value is out of range for int32"):
        _core.fill((2,), dtype="int32", value=3e9, device="cpu")
    with pytest.raises(ValueError, match="fill value is out of range for int32"):
        _core.fill((2,), dtype="int32", value=float("nan"), device="cpu")


def test_int32_add_multiply_wrap_two_complement():
    big = cx.tensor([2**31 - 1, 2**31 - 1], dtype=cx.int32, device="cpu")
    addend = cx.tensor([1, 2], dtype=cx.int32, device="cpu")

    np.testing.assert_array_equal(
        (big + addend).numpy(), np.array([-(2**31), -(2**31) + 1], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        (big * cx.tensor([2, 2], dtype=cx.int32, device="cpu")).numpy(),
        np.array([-2, -2], dtype=np.int32),
    )


def test_randn_rejects_negative_shape():
    with pytest.raises(ValueError, match="shape dimensions must be non-negative"):
        cx.randn((-1,))


def test_rank0_scalar_numpy_round_trip():
    # Rank-0 tensors are reachable via the low-level factory; numpy() must emit a
    # 0-d array, not crash on the empty shape.
    from cortex_runtime import _core

    for dtype, np_dtype in (("float32", np.float32), ("int32", np.int32)):
        scalar = cx.Tensor(_core.zeros((), dtype=dtype, device="cpu"))
        array = scalar.numpy()
        assert array.shape == ()
        assert array.dtype == np_dtype
        assert array.item() == 0


def test_native_tensor_factory_rejects_nested_sequence():
    # The native _core.tensor is the flat 1-D factory; nested data must route
    # through the public cx.tensor() wrapper instead of mis-flattening here.
    from cortex_runtime import _core

    with pytest.raises(ValueError, match="flat numeric sequence"):
        _core.tensor([[1, 2], [3, 4]])
