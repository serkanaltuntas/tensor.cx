import numpy as np
import pytest

import cortex_runtime as cx


MATMUL_CASES = [
    (2, 3, 4),
    (1, 5, 3),
    (4, 2, 1),
]


def optimized_matmul_available():
    return cx.is_available("metal") and "optimized" in cx.matmul_backends("metal")


def tensor_values(shape, seed):
    return cx.randn(shape, dtype=cx.float32, device="cpu", seed=seed)


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
@pytest.mark.parametrize(("m", "k", "n"), MATMUL_CASES)
def test_metal_custom_matmul_matches_cpu(m, k, n):
    lhs_cpu = tensor_values((m, k), seed=m + k)
    rhs_cpu = tensor_values((k, n), seed=k + n)

    actual = cx.matmul(lhs_cpu.to("metal"), rhs_cpu.to("metal"), backend="custom")

    np.testing.assert_allclose(actual.cpu().numpy(), cx.matmul(lhs_cpu, rhs_cpu).numpy(), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not optimized_matmul_available(), reason="Optimized Metal matmul is not available")
@pytest.mark.parametrize(("m", "k", "n"), MATMUL_CASES)
def test_optimized_metal_matmul_matches_cpu(m, k, n):
    lhs_cpu = tensor_values((m, k), seed=10 + m + k)
    rhs_cpu = tensor_values((k, n), seed=20 + k + n)

    actual = cx.matmul(lhs_cpu.to("metal"), rhs_cpu.to("metal"), backend="optimized")

    np.testing.assert_allclose(actual.cpu().numpy(), cx.matmul(lhs_cpu, rhs_cpu).numpy(), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_matmul_auto_uses_working_accelerated_path():
    lhs_cpu = tensor_values((2, 3), seed=101)
    rhs_cpu = tensor_values((3, 2), seed=202)

    actual = cx.matmul(lhs_cpu.to("metal"), rhs_cpu.to("metal"))

    np.testing.assert_allclose(actual.cpu().numpy(), cx.matmul(lhs_cpu, rhs_cpu).numpy(), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_matmul_zero_inner_dimension_returns_zeros():
    lhs = cx.ones((2, 0), dtype=cx.float32, device="metal")
    rhs = cx.ones((0, 3), dtype=cx.float32, device="metal")
    expected = np.zeros((2, 3), dtype=np.float32)

    np.testing.assert_allclose(cx.matmul(lhs, rhs, backend="custom").cpu().numpy(), expected)
    np.testing.assert_allclose(cx.matmul(lhs, rhs).cpu().numpy(), expected)
    if "optimized" in cx.matmul_backends("metal"):
        np.testing.assert_allclose(cx.matmul(lhs, rhs, backend="optimized").cpu().numpy(), expected)


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_matmul_rejects_output_element_count_overflow_before_allocation():
    lhs = cx.ones((4_294_967_295, 0), dtype=cx.float32, device="metal")
    rhs = cx.ones((0, 4_294_967_295), dtype=cx.float32, device="metal")

    with pytest.raises(ValueError, match="support at most 2\\^32 - 1 elements"):
        cx.matmul(lhs, rhs, backend="custom")
    with pytest.raises(ValueError, match="support at most 2\\^32 - 1 elements"):
        cx.matmul(lhs, rhs)


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_matmul_acceptance_snippet():
    a_cpu = cx.randn((2, 3), device="cpu", seed=1)
    b_cpu = cx.randn((3, 4), device="cpu", seed=2)

    a = a_cpu.to("metal")
    b = b_cpu.to("metal")
    c = cx.matmul(a, b)

    np.testing.assert_allclose(c.cpu().numpy(), cx.matmul(a_cpu, b_cpu).numpy(), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not cx.is_available("metal"), reason="Metal is not available")
def test_metal_matmul_rejects_invalid_inputs():
    lhs = cx.ones((2, 3), dtype=cx.float32, device="metal")
    bad_shape = cx.ones((2, 3), dtype=cx.float32, device="metal")
    bad_dtype = cx.ones((3, 2), dtype=cx.int32, device="metal")

    with pytest.raises(ValueError, match="matmul shape mismatch"):
        cx.matmul(lhs, bad_shape, backend="custom")
    with pytest.raises(ValueError, match="matmul only supports float32"):
        cx.matmul(lhs, bad_dtype, backend="custom")
    with pytest.raises(ValueError, match="unsupported Metal matmul backend"):
        cx.matmul(lhs, bad_shape, backend="unknown")
