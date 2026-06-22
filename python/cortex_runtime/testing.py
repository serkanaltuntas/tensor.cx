"""Test helpers with dtype-aware tolerances.

Implements the ``cx.testing.assert_allclose`` referenced by PROJECT.md §12.3 and
AGENTS.md "Testing Requirements". Tolerances follow the table in §12.3: integer
and boolean results are compared for exact equality, while float comparisons use
operation-class defaults (elementwise is tight; reductions and matmul are looser
because GPU and CPU accumulate in different orders).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# (rtol, atol) per operation class, from PROJECT.md §12.3.
_FLOAT_TOLERANCES: dict[str, tuple[float, float]] = {
    "elementwise": (1e-6, 1e-6),
    "reduction": (1e-5, 1e-5),
    "matmul": (1e-4, 1e-4),
}


def _as_array(value: Any) -> np.ndarray:
    """Accept a Cortex Tensor (via ``.numpy()``) or anything array-like."""
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def assert_allclose(
    actual: Any,
    expected: Any,
    *,
    kind: str = "elementwise",
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """Assert two results match within the §12.3 tolerance for ``kind``.

    ``actual`` and ``expected`` may be Cortex ``Tensor`` objects or array-likes.
    Integer/boolean dtypes are always compared exactly; ``rtol``/``atol`` only
    apply to floating-point comparisons and override the ``kind`` default.
    """
    if kind not in _FLOAT_TOLERANCES:
        raise ValueError(
            f"unknown tolerance kind {kind!r}; expected one of "
            f"{sorted(_FLOAT_TOLERANCES)}"
        )

    actual_array = _as_array(actual)
    expected_array = _as_array(expected)

    integer_like = {"i", "u", "b"}
    if actual_array.dtype.kind in integer_like and expected_array.dtype.kind in integer_like:
        np.testing.assert_array_equal(actual_array, expected_array)
        return

    default_rtol, default_atol = _FLOAT_TOLERANCES[kind]
    np.testing.assert_allclose(
        actual_array,
        expected_array,
        rtol=default_rtol if rtol is None else rtol,
        atol=default_atol if atol is None else atol,
    )
