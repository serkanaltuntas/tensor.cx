"""Paired unprofiled/profiled CPU or CUDA workloads and native host-span breakdown."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import tensorcx as cx
from tensorcx import _core
from tensorcx.profiling import _atomic_write

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
try:
    from publication_report import public_report
finally:
    sys.path.pop(0)

DEFAULT_SIZES = (1024, 16384, 262144, 1048576, 16777216)


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def sizes_arg(value):
    try:
        sizes = tuple(positive_int(part) for part in value.split(","))
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError("expected comma-separated positive sizes") from error
    if len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("sizes must be unique")
    return sizes


def summary(samples):
    return dict(samples_ms=samples, median_ms=float(np.median(samples)),
                p10_ms=float(np.percentile(samples, 10)), p90_ms=float(np.percentile(samples, 90)))


def measure(function, validate, repeats, warmup, max_events, device_timing=False):
    """Alternate modes; setup/export/validation/returned-result cleanup are untimed."""
    wall = {"unprofiled": [], "profiled": []}
    profiles = []
    for iteration in range(repeats + warmup):
        modes = ("unprofiled", "profiled") if iteration % 2 == 0 else ("profiled", "unprofiled")
        for mode in modes:
            if mode == "profiled":
                with cx.profile(max_events=max_events, device_timing=device_timing) as profile:
                    start = time.perf_counter_ns()
                    result = function()
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                report = profile.report()
                if not report["complete"]:
                    if report.get("device_errors", 0):
                        errors = sorted({e["error"] for e in report.get("device_events", []) if e["error"]})
                        raise RuntimeError("CUDA event measurement failed: " + ", ".join(errors))
                    raise RuntimeError("runtime profile truncated; increase max_events")
            else:
                start = time.perf_counter_ns()
                result = function()
                elapsed = (time.perf_counter_ns() - start) / 1e6
            validate(result)
            del result
            if iteration >= warmup:
                wall[mode].append(elapsed)
                if mode == "profiled":
                    profiles.append(report)
    # Self times use one host clock and exclude nested spans. Inclusive times
    # overlap and must never be summed into a purported workload/device total.
    keys = sorted({(event["backend"], event["category"]) for report in profiles for event in report["events"]})
    breakdown = []
    for backend, category in keys:
        samples = [sum(e["self_ns"] for e in r["events"] if (e["backend"], e["category"]) == (backend, category)) / 1e6
                   for r in profiles]
        breakdown.append(dict(backend=backend, category=category, **summary(samples)))
    device_breakdown = []
    if device_timing:
        keys = sorted({(e["name"], e["operation"] or "") for r in profiles for e in r["device_events"]})
        for name, operation in keys:
            samples = [sum(e["elapsed_ms"] for e in r["device_events"]
                           if (e["name"], e["operation"] or "") == (name, operation)) for r in profiles]
            device_breakdown.append(dict(name=name, operation=operation or None, **summary(samples)))
    return dict(wall={mode: summary(values) for mode, values in wall.items()},
                self_time_breakdown=breakdown, device_time_breakdown=device_breakdown,
                profiles=profiles, cpu_parity=True)


def command(arguments):
    return subprocess.run(arguments, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(device):
    result = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                  benchmark_sha256=digest(__file__), extension_sha256=digest(_core.__file__),
                  profiler_sha256=digest(inspect.getsourcefile(cx.profile)),
                  python=platform.python_version(), numpy=np.__version__, system=platform.system(),
                  machine=platform.machine(), tensorcx=cx.__version__, device=device,
                  device_name=cx.device_name(device), build=_core._build_info(), seed=701)
    if device == "cuda":
        result["cuda"] = _core._cuda_device_info()
    try:
        result.update(revision=command(["git", "rev-parse", "HEAD"]),
                      git_dirty=bool(command(["git", "status", "--porcelain"])),
                      tracked_diff_sha256=hashlib.sha256(command(["git", "diff", "HEAD"]).encode()).hexdigest())
    except (OSError, subprocess.CalledProcessError):
        result.update(revision=None, git_dirty=None, tracked_diff_sha256=None)
    return result


def gpu_snapshot():
    processes = command(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"])
    others = []
    for line in processes.splitlines():
        pid, memory = line.split(",", 1)
        if int(pid) != os.getpid():
            others.append(dict(memory_mib=memory.strip()))
    return dict(gpu=command(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.used,"
                            "temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu", "--format=csv"]),
                other_compute_processes=others)


def forward(x, first, second):
    return cx.softmax(cx.matmul(cx.gelu(cx.layernorm(cx.matmul(x, first), axis=-1)), second), axis=-1)


def run(args):
    if args.device not in ("cpu", "cuda") or not cx.is_available(args.device):
        raise RuntimeError("requested CPU/CUDA profiling backend is unavailable; no fallback")
    device_timing = getattr(args, "device_timing", False)
    if device_timing and args.device != "cuda":
        raise ValueError("device timing benchmark requires --device cuda")
    details = metadata(args.device)
    if args.device == "cuda":
        details["gpu_before"] = gpu_snapshot()
    rng = np.random.default_rng(701)
    rows = []

    def record(case, shape, function, expected, tolerance=1e-6):
        reference = expected.numpy()
        def validate(result):
            np.testing.assert_allclose(result.numpy(), reference, rtol=tolerance, atol=tolerance)
        row = measure(function, validate, args.repeats, args.warmup, args.max_events, device_timing)
        row["profile_samples"] = [dict(elapsed_ns=r["elapsed_ns"], unattributed_ns=r["unattributed_ns"],
                                       event_count=len(r["events"]), complete=r["complete"],
                                       dropped_events=r["dropped_events"],
                                       device_event_count=len(r["device_events"]), device_errors=r["device_errors"],
                                       dropped_device_events=r["dropped_device_events"],
                                       device_complete=r["device_complete"]) for r in row["profiles"]]
        if not getattr(args, "include_events", False):
            del row["profiles"]
        rows.append(dict(case=case, shape=list(shape), **row))

    for size in args.sizes:
        a, b = (cx.tensor(rng.uniform(-1, 1, size).astype(np.float32)) for _ in range(2))
        x, y = a.to(args.device), b.to(args.device)
        expected = (a + b) * b
        record("expression_resident", (size,), lambda: (x + y) * y, expected)
        if args.device == "cuda":
            def end_to_end():
                left, right = a.to("cuda"), b.to("cuda")
                return ((left + right) * right).cpu()
            record("expression_end_to_end", (size,), end_to_end, expected)
        else:
            record("expression_with_copies", (size,), lambda: ((x.astype(cx.float32) + y.astype(cx.float32)) * y).astype(cx.float32), expected)
    host = [cx.tensor(rng.normal(size=shape).astype(np.float32)) for shape in ((32, 64), (64, 128), (128, 10))]
    inputs = [x.to(args.device) for x in host]
    record("mlp_resident", (32, 64, 128, 10), lambda: forward(*inputs), forward(*host), 1e-4)
    if args.device == "cuda":
        details["gpu_after"] = gpu_snapshot()
        details["other_compute_processes_observed"] = bool(details["gpu_before"]["other_compute_processes"] or
                                                           details["gpu_after"]["other_compute_processes"])
    details["finished_utc"] = datetime.now(timezone.utc).isoformat()
    return public_report(dict(schema_version=1, metadata=details,
                              config=dict(sizes=list(args.sizes), repeats=args.repeats, warmup=args.warmup,
                                          max_events=args.max_events, include_events=getattr(args, "include_events", False)),
                              device_timing=device_timing,
                              measurement=("paired public-call host wall time, native self spans and separate CUDA event intervals"
                                           if device_timing else "paired public-call host wall time and native self spans; no device timing"),
                              results=rows), root=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--sizes", type=sizes_arg, default=DEFAULT_SIZES)
    parser.add_argument("--repeats", type=positive_int, default=7)
    parser.add_argument("--warmup", type=positive_int, default=1)
    parser.add_argument("--max-events", type=positive_int, default=10000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-events", action="store_true", help="include per-span timelines in local reports")
    parser.add_argument("--device-timing", action="store_true", help="include CUDA kernel event intervals separately from host spans")
    args = parser.parse_args()
    text = json.dumps(run(args), indent=2, allow_nan=False) + "\n"
    if args.output:
        _atomic_write(args.output, text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
