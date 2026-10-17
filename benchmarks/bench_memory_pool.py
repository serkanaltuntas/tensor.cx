"""CUDA allocation reuse: synchronous public wall time, bounded retention, CPU parity."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np
import tensorcx as cx
from tensorcx.profiling import _atomic_write
from bench_runtime_profile import DEFAULT_SIZES, positive_int, sizes_arg, summary, metadata, gpu_snapshot, forward, public_report, ROOT


def measure(function, validate, args):
    context = (cx.memory_pool("cuda", max_cached_bytes=args.max_cached_bytes,
                              max_cached_blocks=args.max_cached_blocks)
               if args.pool == "on" else nullcontext(None))
    samples = []
    with context as pool:
        for iteration in range(args.warmup + args.repeats):
            start = time.perf_counter_ns()
            result = function()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            validate(result)
            del result  # Returned-result cleanup is outside the call timer.
            if iteration >= args.warmup:
                samples.append(elapsed)
        # Separate diagnostic call, never one of the latency samples.
        with cx.profile() as p:
            result = function()
        validate(result)
        del result
        report = p.report()
        if not report["complete"]:
            raise RuntimeError("diagnostic profile truncated")
        regions = p.summary()
        before_close = pool.stats() if pool is not None else None
    return dict(**summary(samples), native_regions=regions,
                pool_before_close=before_close, pool_after_close=pool.stats() if pool is not None else None)


def run(args):
    if not cx.is_available("cuda"):
        raise RuntimeError("real CUDA required; no CPU fallback")
    details = metadata("cuda")
    # The imported helper has its own hash; identify this entry point as well.
    import hashlib
    details["memory_pool_benchmark_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    details["pool_available"] = hasattr(cx, "memory_pool")
    details["gpu_before"] = gpu_snapshot()
    rng = np.random.default_rng(701)
    rows = []

    def record(case, shape, function, validate):
        rows.append(dict(case=case, shape=list(shape), **measure(function, validate, args)))

    def parity(expected):
        reference = expected.numpy()
        return lambda result: np.testing.assert_allclose(result.numpy(), reference, rtol=1e-4, atol=1e-4)

    for size in args.sizes:
        def allocation_cycle():
            tensor = cx.empty((size,), dtype=cx.float32, device="cuda")
            return tensor.shape, tensor.dtype, tensor.device
        def validate_allocation(result):
            assert result == ((size,), "float32", "cuda")
        record("allocation_cycle", (size,), allocation_cycle, validate_allocation)
        a, b = (cx.tensor(rng.uniform(-1, 1, size).astype(np.float32)) for _ in range(2))
        x, y = a.to("cuda"), b.to("cuda")
        expected = (a + b) * b
        record("expression_resident", (size,), lambda: (x + y) * y, parity(expected))
        def end_to_end():
            left, right = a.to("cuda"), b.to("cuda")
            return ((left + right) * right).cpu()
        record("expression_end_to_end", (size,), end_to_end, parity(expected))
    host = [cx.tensor(rng.normal(size=shape).astype(np.float32)) for shape in ((32, 64), (64, 128), (128, 10))]
    inputs = [x.to("cuda") for x in host]
    record("mlp_resident", (32, 64, 128, 10), lambda: forward(*inputs), parity(forward(*host)))
    details["gpu_after"] = gpu_snapshot()
    details["other_compute_processes_observed"] = bool(details["gpu_before"]["other_compute_processes"] or
                                                       details["gpu_after"]["other_compute_processes"])
    return public_report(dict(schema_version=1, metadata=details, config=dict(pool=args.pool, sizes=list(args.sizes),
                              repeats=args.repeats, warmup=args.warmup, max_cached_bytes=args.max_cached_bytes,
                              max_cached_blocks=args.max_cached_blocks), results=rows,
                              measurement="unprofiled public-call host wall time; separate native profile; allocation_cycle validates metadata only, other outputs against CPU"), root=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", choices=("off", "on"), required=True)
    parser.add_argument("--sizes", type=sizes_arg, default=DEFAULT_SIZES)
    parser.add_argument("--repeats", type=positive_int, default=7)
    parser.add_argument("--warmup", type=positive_int, default=1)
    parser.add_argument("--max-cached-bytes", type=positive_int, default=64 * 1024 * 1024)
    parser.add_argument("--max-cached-blocks", type=positive_int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(run(args), indent=2, allow_nan=False) + "\n"
    if args.output:
        _atomic_write(args.output, text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
