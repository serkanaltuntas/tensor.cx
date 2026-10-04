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
