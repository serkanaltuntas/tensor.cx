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


def _softmax_reference(values, axis):
    values = np.asarray(values, dtype=np.float32)
    max_values = np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(values - max_values)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def _rmsnorm_reference(values, axis, eps=1.0e-5):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 0:
        return values / np.sqrt(values * values + np.float32(eps))
    mean_square = np.mean(values * values, axis=axis, keepdims=True)
    return values / np.sqrt(mean_square + np.float32(eps))


def _layernorm_reference(values, axis, eps=1.0e-5):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 0:
        return (values - values) / np.sqrt(np.float32(eps))
    mean = np.mean(values, axis=axis, keepdims=True)
    centered = values - mean
    variance = np.mean(centered * centered, axis=axis, keepdims=True)
    return centered / np.sqrt(variance + np.float32(eps))


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


def test_factory_rejects_invalid_shape_types_as_value_error():
    with pytest.raises(ValueError, match="shape dimension is out of range"):
        cx.zeros((2**80,), dtype=cx.float32, device="cpu")
    with pytest.raises(ValueError, match="shape dimension is out of range"):
        cx.zeros((-(2**80),), dtype=cx.float32, device="cpu")
    with pytest.raises(ValueError, match="shape dimensions must be integers"):
        cx.zeros((1.5,), dtype=cx.float32, device="cpu")
    with pytest.raises(ValueError, match="shape dimensions must be integers"):
        cx.zeros("abc", dtype=cx.float32, device="cpu")
    with pytest.raises(ValueError, match="shape dimensions must be integers"):
        cx.zeros(True, dtype=cx.float32, device="cpu")
    with pytest.raises(ValueError, match="shape dimensions must be integers"):
        cx.zeros((True,), dtype=cx.float32, device="cpu")
    with pytest.raises(ValueError, match="shape must be an int or an iterable of ints"):
        cx.zeros(1.5, dtype=cx.float32, device="cpu")


def test_factory_accepts_numpy_integer_shape_dims():
    assert cx.zeros(np.int64(2), dtype=cx.float32, device="cpu").shape == (2,)
    assert cx.zeros((np.int64(2),), dtype=cx.float32, device="cpu").shape == (2,)


def test_factory_rejects_invalid_dtype_type_as_value_error():
    with pytest.raises(ValueError, match="unsupported dtype"):
        cx.zeros((1,), dtype=123, device="cpu")
    with pytest.raises(ValueError, match="unsupported dtype"):
        cx.tensor([1], dtype=123, device="cpu")


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


def test_softmax_cpu_float32_matches_stable_reference():
    x = cx.tensor(
        [[1000.0, 1001.0, 999.0], [-1000.0, -999.0, -1001.0]],
        dtype=cx.float32,
        device="cpu",
    )

    cx.testing.assert_allclose(cx.softmax(x, axis=1), _softmax_reference(x.numpy(), axis=1), kind="reduction")
    cx.testing.assert_allclose(x.softmax(axis=0), _softmax_reference(x.numpy(), axis=0), kind="reduction")
    assert np.isfinite(cx.softmax(x, axis=1).numpy()).all()


def test_rmsnorm_cpu_float32_matches_reference():
    x = cx.tensor(
        [[1.0, -2.0, 3.0], [4.0, 0.5, -6.0]],
        dtype=cx.float32,
        device="cpu",
    )

    cx.testing.assert_allclose(cx.rmsnorm(x, axis=1), _rmsnorm_reference(x.numpy(), axis=1), kind="reduction")
    cx.testing.assert_allclose(x.rmsnorm(axis=0), _rmsnorm_reference(x.numpy(), axis=0), kind="reduction")
    cx.testing.assert_allclose(
        cx.rmsnorm(x, axis=1, eps=1.0e-3),
        _rmsnorm_reference(x.numpy(), axis=1, eps=1.0e-3),
        kind="reduction",
    )
    cx.testing.assert_allclose(
        cx.rmsnorm(x, axis=1, eps=0.0),
        _rmsnorm_reference(x.numpy(), axis=1, eps=0.0),
        kind="reduction",
    )


def test_layernorm_cpu_float32_matches_reference():
    x = cx.tensor(
        [[1.0, -2.0, 3.0], [4.0, 0.5, -6.0]],
        dtype=cx.float32,
        device="cpu",
    )

    cx.testing.assert_allclose(cx.layernorm(x, axis=1), _layernorm_reference(x.numpy(), axis=1), kind="reduction")
    cx.testing.assert_allclose(x.layernorm(axis=0), _layernorm_reference(x.numpy(), axis=0), kind="reduction")
    cx.testing.assert_allclose(
        cx.layernorm(x, axis=1, eps=1.0e-3),
        _layernorm_reference(x.numpy(), axis=1, eps=1.0e-3),
        kind="reduction",
    )
    cx.testing.assert_allclose(
        cx.layernorm(x, axis=1, eps=0.0),
        _layernorm_reference(x.numpy(), axis=1, eps=0.0),
        kind="reduction",
    )

    constant = cx.tensor([[3.0, 3.0, 3.0]], dtype=cx.float32, device="cpu")
    cx.testing.assert_allclose(cx.layernorm(constant, axis=1), np.zeros((1, 3), dtype=np.float32), kind="reduction")
    zero_eps_constant = cx.layernorm(constant, axis=1, eps=0.0)
    assert zero_eps_constant.shape == (1, 3)
    assert np.isnan(zero_eps_constant.numpy()).all()

    nonfinite_constant = cx.tensor([[np.inf, np.inf]], dtype=cx.float32, device="cpu")
    actual = cx.layernorm(nonfinite_constant, axis=1)
    assert actual.shape == (1, 2)
    assert np.isnan(actual.numpy()).all()


def test_sum_and_max_cpu_support_negative_axis():
    x = cx.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[-1.0, -2.0], [5.0, 6.0]]],
        dtype=cx.float32,
        device="cpu",
    )

    cx.testing.assert_allclose(cx.sum(x, axis=-1), np.sum(x.numpy(), axis=-1), kind="reduction")
    cx.testing.assert_allclose(cx.max(x, axis=-2), np.max(x.numpy(), axis=-2), kind="reduction")
    cx.testing.assert_allclose(cx.mean(x, axis=-1), np.mean(x.numpy(), axis=-1), kind="reduction")
    cx.testing.assert_allclose(cx.softmax(x, axis=-1), _softmax_reference(x.numpy(), axis=-1), kind="reduction")
    cx.testing.assert_allclose(cx.rmsnorm(x, axis=-1), _rmsnorm_reference(x.numpy(), axis=-1), kind="reduction")
    cx.testing.assert_allclose(cx.layernorm(x, axis=-1), _layernorm_reference(x.numpy(), axis=-1), kind="reduction")


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


def test_softmax_rank0_scalar_returns_one():
    x = cx.tensor(2.0, dtype=cx.float32, device="cpu")
    actual = cx.softmax(x, axis=0)

    assert actual.shape == ()
    cx.testing.assert_allclose(actual, np.array(1.0, dtype=np.float32), kind="reduction")


def test_rmsnorm_rank0_scalar_returns_scalar():
    x = cx.tensor(2.0, dtype=cx.float32, device="cpu")
    actual = cx.rmsnorm(x, axis=0)

    assert actual.shape == ()
    cx.testing.assert_allclose(actual, _rmsnorm_reference(x.numpy(), axis=0), kind="reduction")


def test_layernorm_rank0_scalar_returns_scalar():
    x = cx.tensor(2.0, dtype=cx.float32, device="cpu")
    actual = cx.layernorm(x, axis=0)

    assert actual.shape == ()
    cx.testing.assert_allclose(actual, _layernorm_reference(x.numpy(), axis=0), kind="reduction")


def test_unary_empty_tensor_returns_empty():
    x = cx.empty((0,), dtype=cx.float32, device="cpu")

    assert cx.exp(x).shape == (0,)
    assert cx.gelu(x).shape == (0,)
    assert cx.silu(x).shape == (0,)
    np.testing.assert_allclose(cx.exp(x).numpy(), np.array([], dtype=np.float32))
    np.testing.assert_allclose(cx.gelu(x).numpy(), np.array([], dtype=np.float32))
    np.testing.assert_allclose(cx.silu(x).numpy(), np.array([], dtype=np.float32))


def test_softmax_empty_tensor_returns_empty():
    x = cx.empty((2, 0), dtype=cx.float32, device="cpu")

    actual = cx.softmax(x, axis=1)

    assert actual.shape == (2, 0)
    np.testing.assert_allclose(actual.numpy(), np.empty((2, 0), dtype=np.float32))


def test_rmsnorm_empty_tensor_returns_empty():
    x = cx.empty((2, 0), dtype=cx.float32, device="cpu")

    actual = cx.rmsnorm(x, axis=1)

    assert actual.shape == (2, 0)
    np.testing.assert_allclose(actual.numpy(), np.empty((2, 0), dtype=np.float32))


def test_layernorm_empty_tensor_returns_empty():
    x = cx.empty((2, 0), dtype=cx.float32, device="cpu")

    actual = cx.layernorm(x, axis=1)

    assert actual.shape == (2, 0)
    np.testing.assert_allclose(actual.numpy(), np.empty((2, 0), dtype=np.float32))


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
    int_x = cx.ones((2, 3), dtype=cx.int32, device="cpu")

    with pytest.raises(ValueError, match="reduction axis is out of range"):
        cx.sum(x, axis=2)
    with pytest.raises(ValueError, match="reduction axis is out of range"):
        cx.max(x, axis=-3)
    with pytest.raises(ValueError, match="reduction axis is out of range"):
        cx.softmax(x, axis=2)
    with pytest.raises(ValueError, match="reduction axis is out of range"):
        cx.softmax(int_x, axis=2)
    with pytest.raises(ValueError, match="reduction axis is out of range"):
        cx.rmsnorm(x, axis=2)
    with pytest.raises(ValueError, match="reduction axis is out of range"):
        cx.rmsnorm(int_x, axis=2)
    with pytest.raises(ValueError, match="reduction axis is out of range"):
        cx.layernorm(x, axis=2)
    with pytest.raises(ValueError, match="reduction axis is out of range"):
        cx.layernorm(int_x, axis=2)
    with pytest.raises(TypeError, match="sum expects a Tensor argument"):
        cx.sum([1, 2, 3], axis=0)
    with pytest.raises(TypeError, match="max expects a Tensor argument"):
        cx.max([1, 2, 3], axis=0)
    with pytest.raises(TypeError, match="mean expects a Tensor argument"):
        cx.mean([1, 2, 3], axis=0)
    with pytest.raises(TypeError, match="softmax expects a Tensor argument"):
        cx.softmax([1, 2, 3], axis=0)
    with pytest.raises(TypeError, match="rmsnorm expects a Tensor argument"):
        cx.rmsnorm([1, 2, 3], axis=0)
    with pytest.raises(TypeError, match="layernorm expects a Tensor argument"):
        cx.layernorm([1, 2, 3], axis=0)


@pytest.mark.parametrize(
    "op",
    [cx.sum, cx.max, cx.mean, cx.softmax, cx.rmsnorm, cx.layernorm],
)
@pytest.mark.parametrize(
    ("axis", "message"),
    [
        (True, "axis must be an integer"),
        (np.bool_(True), "axis must be an integer"),
        ("1", "axis must be an integer"),
        (1.5, "axis must be an integer"),
        (2**80, "axis is out of range"),
        (-(2**80), "axis is out of range"),
    ],
)
def test_axis_ops_reject_invalid_axis_types_before_native_dispatch(op, axis, message):
    x = cx.ones((2, 3), dtype=cx.float32, device="cpu")

    with pytest.raises(ValueError, match=message):
        op(x, axis=axis)


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
    with pytest.raises(ValueError, match="softmax only supports float32"):
        cx.softmax(x, axis=0)
    with pytest.raises(ValueError, match="rmsnorm only supports float32"):
        cx.rmsnorm(x, axis=0)
    with pytest.raises(ValueError, match="layernorm only supports float32"):
        cx.layernorm(x, axis=0)
    with pytest.raises(TypeError, match="exp expects a Tensor argument"):
        cx.exp([1, 2, 3])
    with pytest.raises(TypeError, match="gelu expects a Tensor argument"):
        cx.gelu([1, 2, 3])
    with pytest.raises(TypeError, match="silu expects a Tensor argument"):
        cx.silu([1, 2, 3])


def test_rmsnorm_rejects_invalid_epsilon():
    x = cx.ones((2, 3), dtype=cx.float32, device="cpu")

    with pytest.raises(ValueError, match="epsilon must be finite and non-negative"):
        cx.rmsnorm(x, axis=1, eps=-1.0)
    with pytest.raises(ValueError, match="epsilon must be finite and non-negative"):
        cx.rmsnorm(x, axis=1, eps=np.inf)
    with pytest.raises(ValueError, match="epsilon must be finite and non-negative"):
        cx.rmsnorm(x, axis=1, eps=np.nan)
    with pytest.raises(ValueError, match="epsilon must be finite and non-negative"):
        cx.rmsnorm(x, axis=1, eps=float(np.finfo(np.float32).max) * 2.0)
    with pytest.raises(ValueError, match="epsilon must be finite and non-negative"):
        cx.rmsnorm(x, axis=1, eps=1.0e-50)


def test_layernorm_rejects_invalid_epsilon():
    x = cx.ones((2, 3), dtype=cx.float32, device="cpu")

    with pytest.raises(ValueError, match="epsilon must be finite and non-negative"):
        cx.layernorm(x, axis=1, eps=-1.0)
    with pytest.raises(ValueError, match="epsilon must be finite and non-negative"):
        cx.layernorm(x, axis=1, eps=np.inf)
    with pytest.raises(ValueError, match="epsilon must be finite and non-negative"):
        cx.layernorm(x, axis=1, eps=np.nan)
    with pytest.raises(ValueError, match="epsilon must be finite and non-negative"):
        cx.layernorm(x, axis=1, eps=float(np.finfo(np.float32).max) * 2.0)
    with pytest.raises(ValueError, match="epsilon must be finite and non-negative"):
        cx.layernorm(x, axis=1, eps=1.0e-50)


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
    with pytest.raises(ValueError, match="device is not available"):
        cx.tensor([1, 2, 3], device="missing")


def test_tensor_rejects_unsupported_device_before_materializing_data():
    class ExplodingArrayLike:
        def __array__(self, dtype=None, copy=None):
            raise AssertionError("array conversion should not run")

    with pytest.raises(ValueError, match="device is not available"):
        cx.tensor(ExplodingArrayLike(), device="missing")
    with pytest.raises(ValueError, match="device is not available"):
        cx.empty((2**62,), dtype=cx.float32, device="missing")
    with pytest.raises(ValueError, match="device is not available"):
        cx.zeros((2**62,), dtype=cx.float32, device="missing")
    with pytest.raises(ValueError, match="device is not available"):
        cx.ones((2**62,), dtype=cx.float32, device="missing")


def test_tensor_rejects_bool_data():
    from cortex_runtime import _core

    with pytest.raises(ValueError, match="bool tensor data is not supported"):
        cx.tensor(True, device="cpu")
    with pytest.raises(ValueError, match="bool tensor data is not supported"):
        cx.tensor(np.bool_(True), device="cpu")
    with pytest.raises(ValueError, match="bool tensor data is not supported"):
        cx.tensor([True, False], device="cpu")
    with pytest.raises(ValueError, match="bool tensor data is not supported"):
        cx.tensor([[True, False]], device="cpu")
    with pytest.raises(ValueError, match="bool tensor data is not supported"):
        cx.tensor([1, True], device="cpu")
    with pytest.raises(ValueError, match="bool tensor data is not supported"):
        _core.tensor([True], device="cpu")
    with pytest.raises(ValueError, match="bool tensor data is not supported"):
        _core.tensor([np.bool_(True)], device="cpu")
    with pytest.raises(ValueError, match="bool tensor data is not supported"):
        _core.tensor_from_flat([True], shape=(1,), dtype="int32", device="cpu")
    with pytest.raises(ValueError, match="bool tensor data is not supported"):
        _core.tensor_from_flat([True], shape=(1,), dtype="float32", device="cpu")
    with pytest.raises(ValueError, match="bool tensor data is not supported"):
        _core.tensor_from_flat([np.bool_(True)], shape=(1,), dtype="int32", device="cpu")
    with pytest.raises(ValueError, match="bool tensor data is not supported"):
        _core.tensor_from_flat([np.bool_(True)], shape=(1,), dtype="float32", device="cpu")


def test_tensor_rejects_int_out_of_int32_range():
    with pytest.raises(ValueError, match="out of range for int32"):
        cx.tensor([2**40])
    with pytest.raises(ValueError, match="out of range for int32"):
        cx.tensor([[2**40, 1], [2, 3]])


def test_tensor_rejects_float32_values_that_cannot_be_cast():
    from cortex_runtime import _core

    with pytest.raises(ValueError, match="not convertible to float32"):
        cx.tensor([object()], dtype=cx.float32)
    with pytest.raises(ValueError, match="not convertible to float32"):
        cx.tensor(["not-a-number"], dtype=cx.float32)
    with pytest.raises(ValueError, match="not convertible to float32"):
        cx.tensor([10**400], dtype=cx.float32)
    with pytest.raises(ValueError, match="not convertible to float32"):
        _core.tensor_from_flat([object()], shape=(1,), dtype="float32", device="cpu")


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
    with pytest.raises(ValueError, match="fill value is out of range for int32"):
        _core.fill((2,), dtype="int32", value=1.9, device="cpu")
    with pytest.raises(ValueError, match="fill value is out of range for int32"):
        _core.fill((2,), dtype="int32", value=-1.9, device="cpu")
    with pytest.raises(ValueError, match="fill value is out of range for int32"):
        _core.fill((2**62,), dtype="int32", value=1.9, device="cpu")


def test_fill_accepts_int32_boundary_values():
    from cortex_runtime import _core

    lower = _core.fill((2,), dtype="int32", value=-(2**31), device="cpu")
    upper = _core.fill((2,), dtype="int32", value=2**31 - 1, device="cpu")

    np.testing.assert_array_equal(
        lower.numpy(), np.array([-(2**31), -(2**31)], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        upper.numpy(), np.array([2**31 - 1, 2**31 - 1], dtype=np.int32)
    )


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


def test_randn_shape_uses_project_error_taxonomy():
    with pytest.raises(ValueError, match="shape must be an int or an iterable of ints"):
        cx.randn(1.5)
    with pytest.raises(ValueError, match="shape dimensions must be integers"):
        cx.randn((1.5,))
    with pytest.raises(ValueError, match="shape dimensions must be integers"):
        cx.randn("abc")
    with pytest.raises(ValueError, match="shape dimensions must be integers"):
        cx.randn((True,))


def test_randn_accepts_numpy_integer_shape():
    x = cx.randn((np.int64(2),), seed=1)

    assert x.shape == (2,)


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


def test_tensor_from_flat_rejects_generator_longer_than_shape():
    # The shape fixes the element count, so an over-long (here: infinite)
    # generator must be rejected after at most numel+1 items instead of being
    # drained until the process runs out of memory. The consumption counter
    # pins the early-termination behavior itself, not just the error message.
    import itertools

    from cortex_runtime import _core

    def counting(iterable, consumed):
        for item in iterable:
            consumed.append(item)
            yield item

    for dtype in ("float32", "int32"):
        consumed = []
        with pytest.raises(ValueError, match="tensor data length does not match shape"):
            _core.tensor_from_flat(
                counting(itertools.count(), consumed), shape=(3,), dtype=dtype
            )
        assert len(consumed) <= 4, (
            f"{dtype}: consumed {len(consumed)} items; early termination is broken"
        )

    consumed = []
    with pytest.raises(ValueError, match="tensor data length does not match shape"):
        _core.tensor_from_flat(
            counting(itertools.count(), consumed), shape=(0,), dtype="float32"
        )
    assert len(consumed) <= 1, "(0,)-shape must reject before consuming a second item"


def test_tensor_from_flat_accepts_exact_length_generator():
    from cortex_runtime import _core

    float_tensor = cx.Tensor(
        _core.tensor_from_flat(
            (float(value) for value in range(6)), shape=(2, 3), dtype="float32"
        )
    )
    int_tensor = cx.Tensor(
        _core.tensor_from_flat(
            (value for value in range(6)), shape=(2, 3), dtype="int32"
        )
    )

    np.testing.assert_array_equal(
        float_tensor.numpy(), np.arange(6, dtype=np.float32).reshape(2, 3)
    )
    np.testing.assert_array_equal(
        int_tensor.numpy(), np.arange(6, dtype=np.int32).reshape(2, 3)
    )


def test_tensor_infers_dtype_from_empty_numpy_array_dtype():
    # An empty array has no elements for the C++ 1-D factory to inspect; the
    # dtype must come from the numpy array's own dtype, not default to int32.
    assert cx.tensor(np.array([], dtype=np.float32)).dtype == cx.float32
    assert cx.tensor(np.array([], dtype=np.int32)).dtype == cx.int32
    # A bare empty list follows numpy's default (float), matching np.asarray([]).
    assert cx.tensor([]).dtype == cx.float32
    # Explicit dtype still wins for empty inputs.
    assert cx.tensor([], dtype=cx.int32).dtype == cx.int32
