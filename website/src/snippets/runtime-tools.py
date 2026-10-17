"""Inspect CPU/CUDA host work and opt into bounded CUDA allocation reuse."""
import argparse
from contextlib import nullcontext
import json

import numpy as np
import tensorcx as cx


def run(device):
    x = cx.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    scope = cx.memory_pool(device) if device == "cuda" else nullcontext(None)
    with scope as pool:
        with cx.profile() as profile:
            result = (x + x) * x
            scratch = cx.empty(x.shape, dtype=cx.float32, device=device)
            assert scratch.shape == x.shape  # Initial contents are unspecified.
            del scratch
        # The host copy and CPU comparison are outside the profiling scope.
        np.testing.assert_array_equal(result.numpy(), [2.0, 8.0, 18.0, 32.0])
        del result
        before_close = pool.stats() if pool is not None else None
    report = profile.report()
    assert report["complete"]
    return {
        "device": device, "cpu_parity": True, "profile_complete": report["complete"],
        "native_regions": profile.summary(), "pool_before_close": before_close,
        "pool_after_close": pool.stats() if pool is not None else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    print(json.dumps(run(parser.parse_args().device), indent=2))
