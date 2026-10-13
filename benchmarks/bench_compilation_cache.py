"""Measure uncached versus cached public compile calls, including module loading."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import tempfile
import time
import numpy as np
import tensorcx as cx


@cx.experimental.kernel
def cached_expression(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        total = a[i] + b[i]
        out[i] = total * b[i]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("cpu", "cuda", "metal"), default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("repeats must be positive")
    a, b = (cx.ones((257,), device=args.target) for _ in range(2))
    output = cx.zeros((257,), device=args.target)
    compiler = "auto" if args.target == "metal" else "mlir"
    results = {"target": args.target, "version": cx.version(),
               "measurement": "public compile latency including module loading; not kernel throughput",
               "uncached_ms": [], "warm_ms": []}
    previous = {key: os.environ.get(key) for key in ("TENSORCX_CACHE_DIR", "TENSORCX_DISABLE_CACHE")}
    try:
        with tempfile.TemporaryDirectory(prefix="tensorcx-cache-benchmark-") as directory:
            os.environ["TENSORCX_CACHE_DIR"] = str(Path(directory) / "cache")
            os.environ["TENSORCX_DISABLE_CACHE"] = "0"
            for _ in range(args.repeats):
                start = time.perf_counter()
                artifact = cached_expression.compile(target=args.target, compiler=compiler, cache=False)
                results["uncached_ms"].append((time.perf_counter() - start) * 1000)
                np.testing.assert_array_equal(artifact.launch(a, b, output, 257).numpy(), 2)
                assert not artifact.cache_hit
            start = time.perf_counter()
            first = cached_expression.compile(target=args.target, compiler=compiler)
            results["initial_cache_miss_ms"] = (time.perf_counter() - start) * 1000
            assert not first.cache_hit
            for _ in range(args.repeats):
                start = time.perf_counter()
                artifact = cached_expression.compile(target=args.target, compiler=compiler)
                results["warm_ms"].append((time.perf_counter() - start) * 1000)
                assert artifact.cache_hit
                np.testing.assert_array_equal(artifact.launch(a, b, output, 257).numpy(), 2)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    results["uncached_median_ms"] = statistics.median(results["uncached_ms"])
    results["warm_median_ms"] = statistics.median(results["warm_ms"])
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
