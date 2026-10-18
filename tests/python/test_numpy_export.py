"""NumPy's rank limit must not abort the runtime or silently lose shape."""
import json
import subprocess
import sys

import numpy as np
import pytest

import tensorcx as cx


_EXPORT_PROBE = r'''
import json
import sys
import numpy as np
import tensorcx as cx

device, ranks = sys.argv[1], json.loads(sys.argv[2])
checks = 0
for dtype in (cx.float32, cx.int32, cx.bool):
    for rank in ranks:
        for empty in (False, True):
            if rank == 0 and empty:
                continue
            shape = ((0,) + (1,) * (rank - 1)) if empty else (1,) * rank
            x = cx.zeros(shape, dtype=dtype, device=device) if empty else cx.ones(shape, dtype=dtype, device=device)
            assert x.shape == shape
            try:
                reference = np.ones(shape, dtype=dtype)
            except ValueError:
                for export in (x.numpy, x.cpu()._impl.numpy):
                    try:
                        export()
                    except ValueError:
                        pass
                    else:
                        raise AssertionError(f'unsupported NumPy rank {rank} was accepted')
            else:
                for export in (x.numpy, x.cpu()._impl.numpy):
                    actual = export()
                    assert actual.shape == reference.shape and actual.dtype == reference.dtype
                    np.testing.assert_array_equal(actual, reference)
                    if not empty:
                        actual[...] = 0
            # Export rejection/mutation must not affect the tensor, its context,
            # or the ability to reshape high-rank internal metadata for export.
            count = 0 if empty else 1
            np.testing.assert_array_equal(x.reshape((count,)).numpy(), np.ones(count, dtype=dtype))
            assert x.shape == shape
            checks += 1
assert cx.ones((2,), device=device).numpy().shape == (2,)
if device.partition(':')[0] == 'vulkan':
    assert cx._core._vulkan_validation_errors() == 0
print(json.dumps({'device':device, 'numpy':np.__version__, 'checks':checks}))
'''


@pytest.mark.parametrize("device", cx.devices())
@pytest.mark.parametrize("ranks", [(0, 1, 31, 32, 33, 63, 64), (65, 100, 128), (129, 256, 300)])
def test_numpy_export_rank_boundary_and_recovery(device, ranks):
    # Keep both the pre-fix abort and the wrong-shape fallback in children so
    # these regressions produce ordinary test failures rather than kill pytest.
    result = subprocess.run([sys.executable, "-I", "-c", _EXPORT_PROBE, device, json.dumps(ranks)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads(result.stdout)
    assert evidence == {"device":device, "numpy":np.__version__,
                        "checks":6 * len(ranks) - (3 if 0 in ranks else 0)}


@pytest.mark.parametrize("device", cx.devices())
def test_numpy_export_high_rank_private_copy_and_stored_bits(device):
    bits = np.array([0x80000000, 1, 0x7f812345, 0xffc12345, 0x7f800000, 0x3f800001], np.uint32)
    x = cx.from_dlpack(bits.view(np.float32), copy=False).to(device)
    shape = (1,) * 62 + (2, 3)
    high = x.reshape(shape)
    exported = high.numpy()
    assert exported.shape == shape and exported.dtype == np.float32
    np.testing.assert_array_equal(exported.reshape(-1).view(np.uint32), bits)
    second = high.numpy()
    assert not np.shares_memory(exported, second)
    exported[...] = 0
    del x, high
    np.testing.assert_array_equal(second.reshape(-1).view(np.uint32), bits)
