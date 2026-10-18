"""Manual CUDA fusion experiment: balanced unprofiled calls, separate diagnostics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import permutations
import json
from pathlib import Path
import sys
import time

import numpy as np
import tensorcx as cx
from tensorcx import _core
from tensorcx._compiler.cpu import toolchain
from tensorcx.profiling import _atomic_write

from bench_runtime_profile import (DEFAULT_SIZES, ROOT, command, digest, gpu_snapshot,
                                   metadata, positive_int, public_report, sizes_arg, summary)


@cx.experimental.kernel
def add(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = a[i] + b[i]


@cx.experimental.kernel
def multiply(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = a[i] * b[i]


@cx.experimental.kernel
def subtract(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = a[i] - b[i]


@cx.experimental.kernel
def fused_two(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = (a[i] + b[i]) * b[i]


@cx.experimental.kernel
def fused_four(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        value = (a[i] + b[i]) * b[i]
        out[i] = (value - a[i]) * b[i]


def compile_kernels(target):
    # Compilation and module loading are excluded; no persistent-cache hits.
    return {name: kernel.compile(target=target, compiler="mlir", cache=False)
            for name, kernel in (("add", add), ("multiply", multiply),
                                 ("subtract", subtract), ("two", fused_two), ("four", fused_four))}


def providers(kernels, a, b, workload):
    """Generated outputs use the left input as an immutable full-length template."""
    n = a.shape[0]

    def native():
        value = (a + b) * b
        return (value - a) * b if workload == "four" else value

    def separate():
        value = kernels["add"].launch(a, b, a, n)
        value = kernels["multiply"].launch(value, b, value, n)
        if workload == "four":
            value = kernels["subtract"].launch(value, a, value, n)
            value = kernels["multiply"].launch(value, b, value, n)
        return value

    def fused():
        return kernels[workload].launch(a, b, a, n)

    return dict(native_separate=native, mlir_separate=separate, mlir_fused=fused)


def end_to_end(kernels, a, b, workload, name):
    left, right = a.to("cuda"), b.to("cuda")
    return providers(kernels, left, right, workload)[name]().cpu()


def validate_against(expected):
    def validate(result):
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6, atol=1e-6, equal_nan=False)
    return validate


def measure(functions, validate, repeats, warmup):
    """All six provider permutations; validation and returned-result cleanup untimed."""
    orders = list(permutations(functions))
    if repeats <= 0 or repeats % len(orders):
        raise ValueError("measured repeats must be a positive multiple of the provider permutation count")
    samples = {name: [] for name in functions}
    measured_orders = []
    for iteration in range(warmup + repeats):
        # Restart the balanced order cycle after warmup, independent of its length.
        order = orders[(iteration if iteration < warmup else iteration - warmup) % len(orders)]
        for name in order:
            start = time.perf_counter_ns()
            result = functions[name]()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            validate(result)
            del result
            if iteration >= warmup:
                samples[name].append(elapsed)
        if iteration >= warmup:
            measured_orders.append(list(order))
    return dict(providers={name: summary(values) for name, values in samples.items()},
                provider_orders=measured_orders)


def diagnose(function, validate, expected_launches, repeats, max_events):
    """Separate instrumented calls, never used to compute unprofiled speed ratios."""
    reports = []
    for _ in range(repeats):
        with cx.profile(device_timing=True, max_events=max_events) as profile:
            result = function()
        report = profile.report()
        if not report["complete"]:
            raise RuntimeError("incomplete fusion diagnostic: CUDA event error or event limit")
        if len(report["device_events"]) != expected_launches:
            raise RuntimeError("unexpected CUDA kernel count in fusion diagnostic")
        validate(result)
        del result
        reports.append(report)
    keys = sorted({(e["backend"], e["category"]) for r in reports for e in r["events"]})
    groups = []
    for backend, category in keys:
        selected = [[e for e in r["events"] if (e["backend"], e["category"]) == (backend, category)]
                    for r in reports]
        groups.append(dict(backend=backend, category=category,
                           calls=[len(events) for events in selected],
                           recorded_bytes=[sum(e["bytes"] or 0 for e in events) for events in selected],
                           self_time=summary([sum(e["self_ns"] for e in events) / 1e6 for events in selected])))
    return dict(complete=True, kernel_counts=[len(r["device_events"]) for r in reports],
                device_interval_sum=summary([sum(e["elapsed_ms"] for e in r["device_events"]) for r in reports]),
                host_self_groups=groups,
                device_events=[[dict(name=e["name"], operation=e["operation"], elapsed_ms=e["elapsed_ms"])
                                for e in r["device_events"]] for r in reports])


def run(args):
    if not cx.is_available("cuda"):
        raise RuntimeError("CUDA is required; this experiment never skips or falls back")
    _core._cuda_kernel_support()
    llvm = toolchain()
    details = metadata("cuda")
    details.update(benchmark_sha256=digest(__file__), seed=731,
                   helper_sha256=digest(ROOT / "benchmarks/bench_runtime_profile.py"),
                   llvm={name: command([str(path), "--version"]) for name, path in llvm.items()},
                   gpu_before=gpu_snapshot())
    kernels = compile_kernels("cuda")
    rng = np.random.default_rng(731)
    rows = []
    for size in args.sizes:
        a, b = (cx.tensor(rng.uniform(-1, 1, size).astype(np.float32), device="cpu") for _ in range(2))
        left, right = a.to("cuda"), b.to("cuda")
        for workload, operations in (("two", 2), ("four", 4)):
            expected = (a + b) * b
            if workload == "four":
                expected = (expected - a) * b
            validate = validate_against(expected.numpy())
            resident = providers(kernels, left, right, workload)
            for boundary in ("resident", "end_to_end"):
                functions = resident if boundary == "resident" else {
                    name: lambda name=name: end_to_end(kernels, a, b, workload, name) for name in resident}
                print(f"measuring {workload}/{boundary}/{size}", file=sys.stderr, flush=True)
                wall = measure(functions, validate, args.repeats, args.warmup)
                diagnostics = {name: diagnose(function, validate, 1 if name == "mlir_fused" else operations,
                                               args.diagnostic_repeats, args.max_events)
                               for name, function in functions.items()}
                # Check mutation after every case, including the aliased output template.
                np.testing.assert_array_equal(left.numpy(), a.numpy())
                np.testing.assert_array_equal(right.numpy(), b.numpy())
                medians = {name: values["median_ms"] for name, values in wall["providers"].items()}
                rows.append(dict(workload=workload, operations=operations, size=size, boundary=boundary,
                                 cpu_parity=True, inputs_unchanged=True, wall=wall, diagnostics=diagnostics,
                                 ratios={name + "_over_fused": medians[name] / medians["mlir_fused"]
                                         for name in ("native_separate", "mlir_separate")}))
    details.update(gpu_after=gpu_snapshot(), finished_utc=datetime.now(timezone.utc).isoformat())
    details["other_compute_processes_observed"] = bool(details["gpu_before"]["other_compute_processes"] or
                                                       details["gpu_after"]["other_compute_processes"])
    return public_report(dict(schema_version=1, metadata=details,
                              config=dict(sizes=list(args.sizes), repeats=args.repeats, warmup=args.warmup,
                                          diagnostic_repeats=args.diagnostic_repeats, max_events=args.max_events),
                              measurement="balanced unprofiled synchronous public-call host wall times; separate instrumented diagnostics",
                              scope={"fusion": "explicitly authored DSL expression, no automatic graph fusion",
                                     "compile": "outside timing; cache=False; OS/driver caches retained",
                                     "pool": "not enabled by this CLI",
                                     "output": "private generated result with full output-template copy per launch",
                                     "cleanup": "returned result cleanup and validation excluded; temporaries inside calls included",
                                     "diagnostics": "event intervals include submission gaps/instrumentation/contention; not isolated kernel times; copy event timing absent",
                                     "recorded_bytes": "per host category; nested byte records must not be added as physical traffic"},
                              results=rows), root=ROOT)


def balanced_repeats(value):
    count = positive_int(value)
    if count % 6:
        raise argparse.ArgumentTypeError("must be a positive multiple of six (balanced provider order)")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=sizes_arg, default=DEFAULT_SIZES)
    parser.add_argument("--repeats", type=balanced_repeats, default=30)
    parser.add_argument("--warmup", type=positive_int, default=6)
    parser.add_argument("--diagnostic-repeats", type=positive_int, default=3)
    parser.add_argument("--max-events", type=positive_int, default=10000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(run(args), indent=2, allow_nan=False) + "\n"
    if args.output:
        _atomic_write(args.output, text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
