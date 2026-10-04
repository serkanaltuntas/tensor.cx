import numpy as np
import pytest

import tensorcx as cx


def test_assert_allclose_accepts_tensors():
    a = cx.tensor([1.0, 2.0, 3.0], dtype=cx.float32, device="cpu")
    b = cx.tensor([1.0, 2.0, 3.0], dtype=cx.float32, device="cpu")

    cx.testing.assert_allclose(a, b)


def test_assert_allclose_int_uses_exact_equality():
    a = cx.tensor([1, 2, 3], dtype=cx.int32, device="cpu")
    b = cx.tensor([1, 2, 3], dtype=cx.int32, device="cpu")

    cx.testing.assert_allclose(a, b)

    with pytest.raises(AssertionError):
        cx.testing.assert_allclose(a, cx.tensor([1, 2, 4], dtype=cx.int32, device="cpu"))


def test_assert_allclose_elementwise_tolerance():
    base = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    # Within the §12.3 elementwise tolerance (1e-6) passes.
    cx.testing.assert_allclose(base + 5e-7, base, kind="elementwise")

    # A 1e-3 perturbation is far outside elementwise tolerance.
    with pytest.raises(AssertionError):
        cx.testing.assert_allclose(base + 1e-3, base, kind="elementwise")


def test_assert_allclose_matmul_tolerance_is_looser_than_elementwise():
    base = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    perturbed = base + 5e-5  # outside elementwise (1e-6) but within matmul (1e-4)

    cx.testing.assert_allclose(perturbed, base, kind="matmul")
    with pytest.raises(AssertionError):
        cx.testing.assert_allclose(perturbed, base, kind="elementwise")


def test_assert_allclose_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown tolerance kind"):
        cx.testing.assert_allclose([1.0], [1.0], kind="bogus")


@pytest.mark.parametrize("dtype", [np.float32, np.int32, np.bool_])
@pytest.mark.parametrize("shapes", [((3,), ()), ((), (3,)), ((0,), ()), ((1, 2), (2,))])
def test_assert_allclose_rejects_shape_mismatch(dtype, shapes):
    actual, expected = (np.ones(shape, dtype=dtype) for shape in shapes)
    with pytest.raises(AssertionError, match="shape mismatch"):
        cx.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("reverse", [False, True])
def test_assert_allclose_integer_values_stay_exact_with_float_reference(reverse):
    integer = cx.tensor([100_000_000], dtype=cx.int32)
    expected = np.array([100_000_000.0], dtype=np.float64)
    cx.testing.assert_allclose(integer, expected)
    inputs = (integer, expected + 1)
    if reverse:
        inputs = inputs[::-1]
    with pytest.raises(AssertionError):
        cx.testing.assert_allclose(*inputs, rtol=1, atol=10)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "actual, expected",
    [
        (np.array([2**53 + 1], dtype=np.int64), np.array([2**53], dtype=np.float64)),
        (np.array([2**63 + 1], dtype=np.uint64), np.array([2**63], dtype=np.float64)),
        (np.array([2**63 + 1], dtype=np.uint64), np.array([2**63 - 1], dtype=np.int64)),
    ],
)
def test_assert_allclose_exact_comparison_avoids_lossy_promotion(actual, expected, reverse):
    if reverse:
        actual, expected = expected, actual
    with pytest.raises(AssertionError):
        cx.testing.assert_allclose(actual, expected)


def test_assert_allclose_accepts_exact_large_mixed_values():
    cx.testing.assert_allclose(
        np.array([2**53, 2**53 + 2], dtype=np.int64),
        np.array([2**53, 2**53 + 2], dtype=np.float64),
    )
