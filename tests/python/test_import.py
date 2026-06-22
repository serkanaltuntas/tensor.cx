from importlib.metadata import version

import cortex_runtime as cx


def test_version_matches_package_metadata():
    assert cx.__version__ == version("cortex-runtime")
    assert cx.version() == cx.__version__
