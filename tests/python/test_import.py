import cortex_runtime as cx


def test_version_matches_package_metadata():
    assert cx.__version__ == "0.0.0"
    assert cx.version() == cx.__version__
