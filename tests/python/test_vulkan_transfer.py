"""Vulkan-only copy diagnostics and real heterogeneous pooled placement."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import numpy as np
import pytest
import tensorcx as cx


def test_vulkan_clone_uses_device_copy_without_upload_or_download():
    if not cx.is_available('vulkan'): pytest.skip('Vulkan unavailable')
    value = cx.tensor([True, False, True], device='vulkan')
    before = cx.diagnostics()['backends']['vulkan']['execution']
    result = value.to('vulkan:0', copy=True)
    after = cx.diagnostics()['backends']['vulkan']['execution']
    assert after['submissions'] == before['submissions'] + 1
    assert after['uploads'] == before['uploads'] and after['downloads'] == before['downloads']
    assert after['dispatches'] == before['dispatches']
    np.testing.assert_array_equal(result.numpy(), np.array([True,False,True]))
    assert cx._core._vulkan_validation_errors() == 0


def test_transfer_threads_pools_and_explicit_dlpack_bridge():
    selected = [device for device in ('cuda', 'vulkan') if cx.is_available(device)]
    if len(selected) < 2: pytest.skip('CUDA and Vulkan required for heterogeneous acceptance')
    def forward(i):
        data = np.arange(257, dtype=np.int32) + i
        with ExitStack() as scope:
            for device in selected: scope.enter_context(cx.memory_pool(device))
            source = cx.from_dlpack(data, copy=False).to(selected[i % 2])
            result = source.to(selected[1-i % 2])
            # Exchange only supported CPU/CUDA DLPack storage after placement.
            exported = cx.from_dlpack(result.to('cuda'), copy=False)
            del source, result
        np.testing.assert_array_equal(exported.numpy(), data)
        return exported
    with ThreadPoolExecutor(max_workers=4) as workers:
        values = list(workers.map(forward, range(16)))
    for i, value in enumerate(values): np.testing.assert_array_equal(value.numpy(), np.arange(257, dtype=np.int32) + i)
    assert cx._core._vulkan_validation_errors() == 0
