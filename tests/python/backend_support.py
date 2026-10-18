"""Shared operation gates for broad contracts and intentionally partial backends."""
import pytest
import tensorcx as cx


def require_operations(device, *operations):
    missing = sorted(set(operations) - cx.capabilities(device).keys())
    if missing and device.partition(":")[0] in {"cpu", "metal", "cuda"}:
        pytest.fail(f"{device} lost required operation contracts: {', '.join(missing)}")
    if missing:
        pytest.skip(f"{device} does not declare operations: {', '.join(missing)}")
