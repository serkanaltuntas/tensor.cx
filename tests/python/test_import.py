from importlib.metadata import version

import pytest

import cortex_runtime as cx


def test_version_matches_package_metadata():
    assert cx.__version__ == version("cortex-runtime")
    assert cx.version() == cx.__version__


def test_backend_contract_smoke_test():
    assert cx._core._backend_contract_smoke_test() is True


def test_cpu_backend_contract_smoke_test():
    assert cx._core._cpu_backend_contract_smoke_test() is True


def test_metal_backend_contract_smoke_test():
    if not cx.is_available("metal"):
        pytest.skip("Metal backend is not available")

    assert cx._core._metal_backend_contract_smoke_test() is True
