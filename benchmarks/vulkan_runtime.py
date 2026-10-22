"""Paired synchronous Vulkan latency, allocation/reuse counts and clock reports."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import tensorcx as cx


def positive(value):
    result = int(value)
    if result < 1: raise argparse.ArgumentTypeError("must be positive")
    return result


def counters(device):
    info = cx.diagnostics()["backends"][device]
    return {group: {k: v for k, v in info[group].items() if isinstance(v, int)}
            for group in ("allocations", "execution")}


def difference(after, before):
    return {group: {k: v - before[group][k] for k, v in values.items()}
            for group, values in after.items()}


def validate(value, expected, device, allocation=False):
    assert value.device == device and value.dtype == cx.float32
    if allocation:
        assert value.shape == expected.shape and value.nbytes == expected.nbytes
    else:
        np.testing.assert_allclose(value.numpy(), expected.numpy(), rtol=2e-4, atol=2e-5, equal_nan=True)


def run(*, device="vulkan", elements=4096, iterations=50, rounds=6, warmup=5,
        mode="paired", max_cached_bytes=64 * 1024 * 1024, max_cached_blocks=64):
    selected = cx.device(device)
    if selected.type != "vulkan": raise ValueError("this benchmark requires a Vulkan route")
    device = str(selected)
    if min(elements, iterations, rounds) < 1 or warmup < 0: raise ValueError("invalid benchmark counts")
    if mode not in ("direct", "pooled", "paired"): raise ValueError("invalid benchmark mode")
    cx.capabilities(device)  # Fail explicitly before constructing a workload.
    rng = np.random.default_rng(901)
    a = cx.tensor(rng.uniform(.5, 2, (elements,)).astype(np.float32)); x = a.to(device)
    ha = cx.tensor(rng.uniform(-1, 1, (8, 16)).astype(np.float32))
    hw = cx.tensor(rng.uniform(-1, 1, (16, 8)).astype(np.float32))
    gx, gw = ha.to(device), hw.to(device)
    cases = {
        "allocation": (lambda: cx.empty((elements,), device=device), cx.empty((elements,)), device),
        "fill": (lambda: cx.ones((elements,), device=device), cx.ones((elements,)), device),
        "upload": (lambda: a.to(device), a, device),
        "download": (lambda: x.cpu(), a, "cpu"),
        "arithmetic": (lambda: (x + 1) * .5, (a + 1) * .5, device),
        "inference": (lambda: cx.matmul(gx.layernorm(-1).gelu(), gw).softmax(-1),
                      cx.matmul(ha.layernorm(-1).gelu(), hw).softmax(-1), device),
    }
    modes = ["direct", "pooled"] if mode == "paired" else [mode]
    result = {"schema_version": 1, "kind": "vulkan_runtime_performance", "device": device,
              "cpu_parity": True, "elements": elements, "iterations": iterations, "rounds": rounds,
              "warmup": warmup, "mode": mode, "cases": {},
              "environment": {"system": platform.system(), "machine": platform.machine(),
                              "python": platform.python_version(), "numpy": np.__version__},
              "measurement": {"clock": "perf_counter_ns_host", "execution": "synchronous",
                  "order": "alternate mode order each round; fresh pool per sample",
                  "cold": "first call in the scope; process/device/toolchain already initialized",
                  "warm": "iterations after cold parity check and warmup; excludes scope entry/exit",
                  "counters": "whole sample including parity download and warmup; not per-iteration counts",
                  "device_intervals": "separate profiled capture; no alignment to host clock",
                  "limits": "shared desktop, one device/driver; no cross-GPU or isolated-host claim"}}

    def scope(which):
        return (cx.memory_pool(device, max_cached_bytes=max_cached_bytes, max_cached_blocks=max_cached_blocks)
                if which == "pooled" else nullcontext())

    for name, (fn, expected, output_device) in cases.items():
        row = {"samples": {which: [] for which in modes}, "summary": {}, "profiles": {}}
        for repeat in range(rounds):
            order = modes if repeat % 2 == 0 else list(reversed(modes))
            for which in order:
                before = counters(device)
                context = scope(which)
                started = time.perf_counter_ns(); pool = context.__enter__()
                enter_ns = time.perf_counter_ns() - started
                try:
                    started = time.perf_counter_ns(); value = fn()
                    cold_ns = time.perf_counter_ns() - started
                    validate(value, expected, output_device, name == "allocation"); del value
                    for _ in range(warmup): value = fn(); del value
                    started = time.perf_counter_ns()
                    for _ in range(iterations): value = fn(); del value
                    elapsed = time.perf_counter_ns() - started
                finally:
                    started = time.perf_counter_ns(); context.__exit__(None, None, None)
                    exit_ns = time.perf_counter_ns() - started
                row["samples"][which].append({"round": repeat, "order": order.index(which),
                    "cold_ns": cold_ns, "mean_ns": elapsed / iterations,
                    "scope_enter_ns": enter_ns, "scope_exit_ns": exit_ns,
                    "counters": difference(counters(device), before),
                    "pool": pool.stats() if pool else None})
        for which in modes:
            values = [sample["mean_ns"] for sample in row["samples"][which]]
            q1, _, q3 = np.percentile(values, [25, 50, 75])
            row["summary"][which] = {"median_ns": statistics.median(values),
                "min_ns": min(values), "max_ns": max(values), "iqr_ns": float(q3 - q1)}
            # Capture instrumentation separately: query creation/recording is
            # deliberately excluded from the unprofiled latency samples above.
            with cx.profile(device_timing=True) as profile:
                with scope(which):
                    for _ in range(3):
                        value = fn(); validate(value, expected, output_device, name == "allocation"); del value
            p = profile.report()
            row["profiles"][which] = {"clock": p["clock"], "device_clock": p["device_clock"],
                "host_complete": p["host_complete"], "device_complete": p["device_complete"],
                "device_errors": p["device_errors"], "host_summary": profile.summary(),
                "device_summary": profile.device_summary()}
        if mode == "paired":
            row["direct_over_pooled_ratio"] = row["summary"]["direct"]["median_ns"] / row["summary"]["pooled"]["median_ns"]
        result["cases"][name] = row
    diagnostic = cx.diagnostics()["backends"][device]
    result["environment"].update({k: diagnostic[k] for k in ("name", "api_version", "driver_version_raw")})
    result["timestamps"] = diagnostic["timestamps"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="vulkan")
    parser.add_argument("--elements", type=positive, default=4096)
    parser.add_argument("--iterations", type=positive, default=50)
    parser.add_argument("--rounds", type=positive, default=6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--mode", choices=("direct", "pooled", "paired"), default="paired")
    parser.add_argument("--max-cached-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-cached-blocks", type=positive, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.max_cached_bytes < 0: parser.error("warmup/cache bytes must be nonnegative")
    report = run(**{k: v for k, v in vars(args).items() if k != "output"})
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("CPU parity verified; synchronous medians in microseconds:")
    for name, row in report["cases"].items():
        print(name, {mode: round(s["median_ns"] / 1000, 2) for mode, s in row["summary"].items()})


if __name__ == "__main__": main()
