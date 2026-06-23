import numpy as np
import pytest

import cortex_runtime as cx


HUGE_SHAPE = (3_037_000_500, 3_037_000_500)
STRIDE_OVERFLOW_SHAPE = (0, 9_223_372_036_854_775_807, 2)
TOO_MANY_METAL_THREADS_SHAPE = (4_294_967_296,)


def test_best_device_add_success_snippet():
    device = cx.best_device()

    x = cx.ones((1_000_000,), dtype=cx.float32, device=device)
    y = cx.ones((1_000_000,), dtype=cx.float32, device=device)
    z = x + y

    np.testing.assert_allclose(z.cpu().numpy()[:5], np.array([2, 2, 2, 2, 2], dtype=np.float32))


def test_binary_ops_reject_device_mismatch():
    if not cx.is_available("metal"):
        pytest.skip("Metal is not available")

    x = cx.ones((3,), dtype=cx.float32, device="cpu")
    y = cx.ones((3,), dtype=cx.float32, device="metal")

    with pytest.raises(ValueError, match="device mismatch"):
        _ = x + y


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_add_float32_matches_cpu():
    x_cpu = cx.tensor([1.0, 2.0, 3.0], dtype=cx.float32, device="cpu")
    y_cpu = cx.tensor([4.0, 5.0, 6.0], dtype=cx.float32, device="cpu")

    z_gpu = x_cpu.to("metal") + y_cpu.to("metal")

    np.testing.assert_allclose(z_gpu.cpu().numpy(), (x_cpu + y_cpu).numpy())


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_multiply_float32_matches_cpu():
    x_cpu = cx.tensor([1.5, 2.0, 3.5], dtype=cx.float32, device="cpu")
    y_cpu = cx.tensor([4.0, 5.5, 6.0], dtype=cx.float32, device="cpu")

    z_gpu = x_cpu.to("metal") * y_cpu.to("metal")

    np.testing.assert_allclose(z_gpu.cpu().numpy(), (x_cpu * y_cpu).numpy())


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_fill_float32_matches_cpu():
    zeros_cpu = cx.zeros((2, 3), dtype=cx.float32, device="cpu")
    ones_cpu = cx.ones((2, 3), dtype=cx.float32, device="cpu")
    zeros = cx.zeros((2, 3), dtype=cx.float32, device="metal")
    ones = cx.ones((2, 3), dtype=cx.float32, device="metal")

    np.testing.assert_allclose(zeros.cpu().numpy(), zeros_cpu.numpy())
    np.testing.assert_allclose(ones.cpu().numpy(), ones_cpu.numpy())


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_fill_rejects_shape_size_overflow():
    with pytest.raises(ValueError, match="shape size overflow"):
        cx.zeros(HUGE_SHAPE, dtype=cx.float32, device="metal")


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_fill_rejects_shape_stride_overflow():
    with pytest.raises(ValueError, match="shape stride overflow"):
        cx.zeros(STRIDE_OVERFLOW_SHAPE, dtype=cx.float32, device="metal")


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_fill_rejects_shapes_larger_than_thread_limit_before_allocation():
    with pytest.raises(ValueError, match="support at most 2\\^32 - 1 elements"):
        cx.zeros(TOO_MANY_METAL_THREADS_SHAPE, dtype=cx.float32, device="metal")


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_fill_rejects_int32_value_out_of_range():
    from cortex_runtime import _core

    with pytest.raises(ValueError, match="fill value is out of range for int32"):
        _core.fill((2,), dtype="int32", value=3e9, device="metal")


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_int32_overflow_wraps_like_cpu():
    big = cx.tensor([2**31 - 1, 2**31 - 1], dtype=cx.int32, device="cpu")
    addend = cx.tensor([1, 2], dtype=cx.int32, device="cpu")

    metal_sum = (big.to("metal") + addend.to("metal")).cpu().numpy()
    metal_prod = (big.to("metal") * addend.to("metal")).cpu().numpy()

    # Pinned to literal two's-complement results so this is independent of the
    # CPU path (both share the same uint round-trip).
    # sum:  [(2^31-1)+1, (2^31-1)+2] wrap to [-2^31, -2^31+1]
    # prod: [(2^31-1)*1, (2^31-1)*2] = [2^31-1, 4294967294 -> -2]
    np.testing.assert_array_equal(metal_sum, np.array([-(2**31), -(2**31) + 1], dtype=np.int32))
    np.testing.assert_array_equal(metal_prod, np.array([2**31 - 1, -2], dtype=np.int32))


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_add_int32_matches_cpu():
    x_cpu = cx.tensor([1, 2, 3], dtype=cx.int32, device="cpu")
    y_cpu = cx.tensor([4, 5, 6], dtype=cx.int32, device="cpu")

    z = x_cpu.to("metal") + y_cpu.to("metal")

    np.testing.assert_array_equal(z.cpu().numpy(), (x_cpu + y_cpu).numpy())


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_multiply_int32_matches_cpu():
    x_cpu = cx.tensor([2, 3, 4], dtype=cx.int32, device="cpu")
    y_cpu = cx.tensor([5, 6, 7], dtype=cx.int32, device="cpu")

    z_gpu = x_cpu.to("metal") * y_cpu.to("metal")

    np.testing.assert_array_equal(z_gpu.cpu().numpy(), (x_cpu * y_cpu).numpy())


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_fill_int32_matches_cpu():
    zeros_cpu = cx.zeros((2, 3), dtype=cx.int32, device="cpu")
    ones_cpu = cx.ones((2, 3), dtype=cx.int32, device="cpu")
    zeros = cx.zeros((2, 3), dtype=cx.int32, device="metal")
    ones = cx.ones((2, 3), dtype=cx.int32, device="metal")

    np.testing.assert_array_equal(zeros.cpu().numpy(), zeros_cpu.numpy())
    np.testing.assert_array_equal(ones.cpu().numpy(), ones_cpu.numpy())


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_sum_and_max_float32_match_cpu_on_non_trivial_axis():
    x_cpu = cx.tensor(
        [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]],
        dtype=cx.float32,
        device="cpu",
    )
    x_metal = x_cpu.to("metal")

    cx.testing.assert_allclose(cx.sum(x_metal, axis=1).cpu(), cx.sum(x_cpu, axis=1), kind="reduction")
    cx.testing.assert_allclose(x_metal.sum(axis=0).cpu(), x_cpu.sum(axis=0), kind="reduction")
    cx.testing.assert_allclose(cx.max(x_metal, axis=1).cpu(), cx.max(x_cpu, axis=1), kind="reduction")
    cx.testing.assert_allclose(x_metal.max(axis=0).cpu(), x_cpu.max(axis=0), kind="reduction")


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_mean_float32_matches_cpu_on_non_trivial_axis():
    x_cpu = cx.tensor(
        [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]],
        dtype=cx.float32,
        device="cpu",
    )
    x_metal = x_cpu.to("metal")

    cx.testing.assert_allclose(cx.mean(x_metal, axis=1).cpu(), cx.mean(x_cpu, axis=1), kind="reduction")
    cx.testing.assert_allclose(x_metal.mean(axis=0).cpu(), x_cpu.mean(axis=0), kind="reduction")


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_exp_float32_matches_cpu():
    x_cpu = cx.tensor([-2.0, -0.5, 0.0, 1.0, 3.0], dtype=cx.float32, device="cpu")
    x_metal = x_cpu.to("metal")

    cx.testing.assert_allclose(cx.exp(x_metal).cpu(), cx.exp(x_cpu), kind="elementwise")
    cx.testing.assert_allclose(x_metal.exp().cpu(), x_cpu.exp(), kind="elementwise")


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_gelu_and_silu_float32_match_cpu():
    x_cpu = cx.tensor([-4.0, -1.0, 0.0, 0.5, 2.0, 5.0], dtype=cx.float32, device="cpu")
    x_metal = x_cpu.to("metal")

    cx.testing.assert_allclose(cx.gelu(x_metal).cpu(), cx.gelu(x_cpu), kind="elementwise")
    cx.testing.assert_allclose(x_metal.gelu().cpu(), x_cpu.gelu(), kind="elementwise")
    cx.testing.assert_allclose(cx.silu(x_metal).cpu(), cx.silu(x_cpu), kind="elementwise")
    cx.testing.assert_allclose(x_metal.silu().cpu(), x_cpu.silu(), kind="elementwise")


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_softmax_float32_matches_cpu_on_non_trivial_axis():
    x_cpu = cx.tensor(
        [[1000.0, 1001.0, 999.0], [-1000.0, -999.0, -1001.0]],
        dtype=cx.float32,
        device="cpu",
    )
    x_metal = x_cpu.to("metal")

    cx.testing.assert_allclose(cx.softmax(x_metal, axis=1).cpu(), cx.softmax(x_cpu, axis=1), kind="reduction")
    cx.testing.assert_allclose(x_metal.softmax(axis=0).cpu(), x_cpu.softmax(axis=0), kind="reduction")
    assert np.isfinite(cx.softmax(x_metal, axis=1).cpu().numpy()).all()


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_rmsnorm_float32_matches_cpu_on_non_trivial_axis():
    x_cpu = cx.tensor(
        [[1.0, -2.0, 3.0], [4.0, 0.5, -6.0]],
        dtype=cx.float32,
        device="cpu",
    )
    x_metal = x_cpu.to("metal")

    cx.testing.assert_allclose(cx.rmsnorm(x_metal, axis=1).cpu(), cx.rmsnorm(x_cpu, axis=1), kind="reduction")
    cx.testing.assert_allclose(x_metal.rmsnorm(axis=0).cpu(), x_cpu.rmsnorm(axis=0), kind="reduction")
    cx.testing.assert_allclose(
        cx.rmsnorm(x_metal, axis=1, eps=1.0e-3).cpu(),
        cx.rmsnorm(x_cpu, axis=1, eps=1.0e-3),
        kind="reduction",
    )
    cx.testing.assert_allclose(
        cx.rmsnorm(x_metal, axis=1, eps=0.0).cpu(),
        cx.rmsnorm(x_cpu, axis=1, eps=0.0),
        kind="reduction",
    )


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_layernorm_float32_matches_cpu_on_non_trivial_axis():
    x_cpu = cx.tensor(
        [[1.0, -2.0, 3.0], [4.0, 0.5, -6.0]],
        dtype=cx.float32,
        device="cpu",
    )
    x_metal = x_cpu.to("metal")

    cx.testing.assert_allclose(cx.layernorm(x_metal, axis=1).cpu(), cx.layernorm(x_cpu, axis=1), kind="reduction")
    cx.testing.assert_allclose(x_metal.layernorm(axis=0).cpu(), x_cpu.layernorm(axis=0), kind="reduction")
    cx.testing.assert_allclose(
        cx.layernorm(x_metal, axis=1, eps=1.0e-3).cpu(),
        cx.layernorm(x_cpu, axis=1, eps=1.0e-3),
        kind="reduction",
    )
    cx.testing.assert_allclose(
        cx.layernorm(x_metal, axis=1, eps=0.0).cpu(),
        cx.layernorm(x_cpu, axis=1, eps=0.0),
        kind="reduction",
    )
    constant = cx.tensor([[3.0, 3.0, 3.0]], dtype=cx.float32, device="metal")
    cx.testing.assert_allclose(
        cx.layernorm(constant, axis=1).cpu(),
        np.zeros((1, 3), dtype=np.float32),
        kind="reduction",
    )
    zero_eps_constant = cx.layernorm(constant, axis=1, eps=0.0).cpu()
    assert zero_eps_constant.shape == (1, 3)
    assert np.isnan(zero_eps_constant.numpy()).all()

    nonfinite_constant = cx.tensor([[np.inf, np.inf]], dtype=cx.float32, device="metal")
    actual = cx.layernorm(nonfinite_constant, axis=1).cpu()
    assert actual.shape == (1, 2)
    assert np.isnan(actual.numpy()).all()


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_reductions_support_negative_axis():
    x_cpu = cx.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[-1.0, -2.0], [5.0, 6.0]]],
        dtype=cx.float32,
        device="cpu",
    )
    x_metal = x_cpu.to("metal")

    cx.testing.assert_allclose(cx.sum(x_metal, axis=-1).cpu(), cx.sum(x_cpu, axis=-1), kind="reduction")
    cx.testing.assert_allclose(cx.max(x_metal, axis=-2).cpu(), cx.max(x_cpu, axis=-2), kind="reduction")
    cx.testing.assert_allclose(cx.mean(x_metal, axis=-1).cpu(), cx.mean(x_cpu, axis=-1), kind="reduction")
    cx.testing.assert_allclose(cx.softmax(x_metal, axis=-1).cpu(), cx.softmax(x_cpu, axis=-1), kind="reduction")
    cx.testing.assert_allclose(cx.rmsnorm(x_metal, axis=-1).cpu(), cx.rmsnorm(x_cpu, axis=-1), kind="reduction")
    cx.testing.assert_allclose(cx.layernorm(x_metal, axis=-1).cpu(), cx.layernorm(x_cpu, axis=-1), kind="reduction")


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_sum_and_max_int32_match_cpu():
    x_cpu = cx.tensor([[1, 2, 3], [4, 5, 6]], dtype=cx.int32, device="cpu")
    x_metal = x_cpu.to("metal")

    np.testing.assert_array_equal(cx.sum(x_metal, axis=1).cpu().numpy(), cx.sum(x_cpu, axis=1).numpy())
    np.testing.assert_array_equal(cx.max(x_metal, axis=0).cpu().numpy(), cx.max(x_cpu, axis=0).numpy())


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_rank1_reduction_returns_rank0_scalar():
    x_cpu = cx.tensor([1, 2, 3], dtype=cx.int32, device="cpu")
    x_metal = x_cpu.to("metal")

    summed = cx.sum(x_metal, axis=0).cpu()
    maximum = cx.max(x_metal, axis=0).cpu()

    assert summed.shape == ()
    assert maximum.shape == ()
    assert summed.numpy().item() == 6
    assert maximum.numpy().item() == 3


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_rank0_scalar_reduction_returns_scalar():
    x_cpu = cx.tensor(5, dtype=cx.int32, device="cpu")
    y_cpu = cx.tensor(5.0, dtype=cx.float32, device="cpu")
    x_metal = x_cpu.to("metal")
    y_metal = y_cpu.to("metal")

    summed = cx.sum(x_metal, axis=0).cpu()
    maximum = cx.max(x_metal, axis=-1).cpu()
    mean = cx.mean(y_metal, axis=0).cpu()

    assert summed.shape == ()
    assert maximum.shape == ()
    assert mean.shape == ()
    assert summed.numpy().item() == 5
    assert maximum.numpy().item() == 5
    assert mean.numpy().item() == 5.0


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_exp_rank0_scalar_returns_scalar():
    x_cpu = cx.tensor(2.0, dtype=cx.float32, device="cpu")

    actual = cx.exp(x_cpu.to("metal")).cpu()

    assert actual.shape == ()
    cx.testing.assert_allclose(actual, cx.exp(x_cpu), kind="elementwise")


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_gelu_and_silu_rank0_scalar_return_scalar():
    x_cpu = cx.tensor(2.0, dtype=cx.float32, device="cpu")
    x_metal = x_cpu.to("metal")

    gelu = cx.gelu(x_metal).cpu()
    silu = cx.silu(x_metal).cpu()

    assert gelu.shape == ()
    assert silu.shape == ()
    cx.testing.assert_allclose(gelu, cx.gelu(x_cpu), kind="elementwise")
    cx.testing.assert_allclose(silu, cx.silu(x_cpu), kind="elementwise")


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_softmax_rank0_scalar_returns_one():
    x_cpu = cx.tensor(2.0, dtype=cx.float32, device="cpu")

    actual = cx.softmax(x_cpu.to("metal"), axis=0).cpu()

    assert actual.shape == ()
    cx.testing.assert_allclose(actual, np.array(1.0, dtype=np.float32), kind="reduction")


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_rmsnorm_rank0_scalar_returns_scalar():
    x_cpu = cx.tensor(2.0, dtype=cx.float32, device="cpu")

    actual = cx.rmsnorm(x_cpu.to("metal"), axis=0).cpu()

    assert actual.shape == ()
    cx.testing.assert_allclose(actual, cx.rmsnorm(x_cpu, axis=0), kind="reduction")


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_layernorm_rank0_scalar_returns_scalar():
    x_cpu = cx.tensor(2.0, dtype=cx.float32, device="cpu")

    actual = cx.layernorm(x_cpu.to("metal"), axis=0).cpu()

    assert actual.shape == ()
    cx.testing.assert_allclose(actual, cx.layernorm(x_cpu, axis=0), kind="reduction")


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_unary_empty_tensor_returns_empty():
    x = cx.empty((0,), dtype=cx.float32, device="metal")

    assert cx.exp(x).shape == (0,)
    assert cx.gelu(x).shape == (0,)
    assert cx.silu(x).shape == (0,)
    np.testing.assert_allclose(cx.exp(x).cpu().numpy(), np.array([], dtype=np.float32))
    np.testing.assert_allclose(cx.gelu(x).cpu().numpy(), np.array([], dtype=np.float32))
    np.testing.assert_allclose(cx.silu(x).cpu().numpy(), np.array([], dtype=np.float32))


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_softmax_empty_tensor_returns_empty():
    x = cx.empty((2, 0), dtype=cx.float32, device="metal")

    actual = cx.softmax(x, axis=1).cpu()

    assert actual.shape == (2, 0)
    np.testing.assert_allclose(actual.numpy(), np.empty((2, 0), dtype=np.float32))


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_rmsnorm_empty_tensor_returns_empty():
    x = cx.empty((2, 0), dtype=cx.float32, device="metal")

    actual = cx.rmsnorm(x, axis=1).cpu()

    assert actual.shape == (2, 0)
    np.testing.assert_allclose(actual.numpy(), np.empty((2, 0), dtype=np.float32))


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_layernorm_empty_tensor_returns_empty():
    x = cx.empty((2, 0), dtype=cx.float32, device="metal")

    actual = cx.layernorm(x, axis=1).cpu()

    assert actual.shape == (2, 0)
    np.testing.assert_allclose(actual.numpy(), np.empty((2, 0), dtype=np.float32))


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_max_float32_negative_infinity_matches_cpu():
    x_cpu = cx.tensor([[-np.inf, -np.inf]], dtype=cx.float32, device="cpu")

    cx.testing.assert_allclose(
        cx.max(x_cpu.to("metal"), axis=1).cpu(),
        cx.max(x_cpu, axis=1),
        kind="reduction",
    )


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_max_float32_nan_matches_cpu_and_numpy():
    x_cpu = cx.tensor([[np.nan, -1.0], [1.0, 2.0]], dtype=cx.float32, device="cpu")

    cx.testing.assert_allclose(
        cx.max(x_cpu.to("metal"), axis=1).cpu(),
        np.max(x_cpu.numpy(), axis=1),
        kind="reduction",
    )


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_sum_int32_wraps_like_cpu():
    x_cpu = cx.tensor([[2**31 - 1, 1]], dtype=cx.int32, device="cpu")

    np.testing.assert_array_equal(
        cx.sum(x_cpu.to("metal"), axis=1).cpu().numpy(),
        np.array([-(2**31)], dtype=np.int32),
    )


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_sum_empty_axis_returns_zero_and_max_rejects_empty_axis():
    x = cx.empty((2, 0), dtype=cx.float32, device="metal")
    xi = cx.empty((2, 0), dtype=cx.int32, device="metal")

    np.testing.assert_allclose(cx.sum(x, axis=1).cpu().numpy(), np.zeros((2,), dtype=np.float32))
    np.testing.assert_array_equal(cx.sum(xi, axis=1).cpu().numpy(), np.zeros((2,), dtype=np.int32))
    cx.testing.assert_allclose(
        cx.mean(x, axis=1).cpu(),
        np.array([np.nan, np.nan], dtype=np.float32),
        kind="reduction",
    )
    with pytest.raises(ValueError, match="max reduction requires a non-empty axis"):
        cx.max(x, axis=1)


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_reductions_reject_invalid_axis():
    x = cx.ones((2, 3), dtype=cx.float32, device="metal")
    int_x = cx.ones((2, 3), dtype=cx.int32, device="metal")

    with pytest.raises(ValueError, match="reduction axis is out of range"):
        cx.sum(x, axis=2)
    with pytest.raises(ValueError, match="reduction axis is out of range"):
        cx.max(x, axis=-3)
    with pytest.raises(ValueError, match="reduction axis is out of range"):
        cx.mean(x, axis=2)
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


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_float32_only_ops_reject_int32():
    x = cx.tensor([1, 2, 3], dtype=cx.int32, device="metal")

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


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_rmsnorm_rejects_invalid_epsilon():
    x = cx.ones((2, 3), dtype=cx.float32, device="metal")

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


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_layernorm_rejects_invalid_epsilon():
    x = cx.ones((2, 3), dtype=cx.float32, device="metal")

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
