"""Compile/cache latency, separate cProfile attribution and resident public launches."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import cProfile
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import tensorcx as cx
from tensorcx import _core

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(cx.__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))
try:
    from publication_report import public_report
finally:
    sys.path.pop(0)
DEFAULT_SIZES = (1024, 16384, 262144, 1048576, 16777216)


@cx.experimental.kernel
def cached_expression(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        total = a[i] + b[i]
        out[i] = total * b[i]


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def sizes_arg(value):
    try:
        sizes = tuple(positive_int(part) for part in value.split(","))
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError("expected comma-separated positive sizes") from error
    if len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("sizes must be unique")
    return sizes


def summary(samples):
    return {"samples_ms": samples, "median_ms": float(np.median(samples)),
            "p10_ms": float(np.percentile(samples, 10)),
            "p90_ms": float(np.percentile(samples, 90))}


def measure(functions, repeats, warmup, validate, prepare=None):
    """Alternate order; retain results until the clock stops, then check/release."""
    samples = {name: [] for name in functions}
    names = list(functions)
    for iteration in range(warmup + repeats):
        for name in names if iteration % 2 == 0 else names[::-1]:
            if prepare is not None:
                prepare(name)
            start = time.perf_counter_ns()
            result = functions[name]()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            validate(name, result)
            del result
            if iteration >= warmup:
                samples[name].append(elapsed)
    return {name: summary(values) for name, values in samples.items()}


@contextmanager
def private_cache(directory):
    previous = {key: os.environ.get(key) for key in ("TENSORCX_CACHE_DIR", "TENSORCX_DISABLE_CACHE")}
    try:
        os.environ["TENSORCX_CACHE_DIR"] = str(directory)
        os.environ["TENSORCX_DISABLE_CACHE"] = "0"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def inputs(target, count):
    rng = np.random.default_rng(17)
    arrays = [rng.uniform(-0.75, 0.75, count).astype(np.float32) for _ in range(2)]
    cpu_a, cpu_b = (cx.tensor(array, device="cpu") for array in arrays)
    expected = ((cpu_a + cpu_b) * cpu_b).numpy()
    a, b = (cx.tensor(array, device=target) for array in arrays)
    poison = np.full(count, np.nan, dtype=np.float32)
    output = cx.tensor(poison, device=target)

    def validate(artifact):
        # A fresh poisoned output prevents stale in-place Metal results passing.
        fresh = cx.tensor(poison, device=target)
        np.testing.assert_allclose(artifact.launch(a, b, fresh, count).numpy(), expected,
                                   rtol=1e-6, atol=1e-6)

    return (a, b, output, count), expected, validate


def compile_call(target, *, cache=True):
    return cached_expression.compile(target=target, compiler="auto" if target == "metal" else "mlir", cache=cache)


def check_artifact(name, artifact, validate):
    expected_hit = name in {"warm", "restart"}
    if artifact.cache_hit is not expected_hit:
        raise AssertionError(f"{name}: unexpected cache_hit={artifact.cache_hit}")
    validate(artifact)


def profile_compile(call, validate):
    """Keep only repo-relative function labels; never save a raw profiler dump."""
    profiler = cProfile.Profile()
    start = time.perf_counter_ns()
    result = profiler.runcall(call)
    elapsed = (time.perf_counter_ns() - start) / 1e6
    validate(result)  # Outside both the profile and the clock.
    del result
    stats = pstats.Stats(profiler).stats
    rows = []
    subprocess_row = {"calls": 0, "cumulative_ms": 0.0}
    for (filename, line, name), (primitive, calls, own, cumulative, _) in stats.items():
        if (filename, line, name) == (subprocess.run.__code__.co_filename,
                                      subprocess.run.__code__.co_firstlineno, "run"):
            subprocess_row = {"calls": calls, "cumulative_ms": cumulative * 1000}
        path = Path(filename)
        if not path.is_absolute():
            continue
        try:
            relative = "python/tensorcx/" + path.resolve().relative_to(PACKAGE_ROOT).as_posix()
        except ValueError:
            if path.resolve() != Path(__file__).resolve():
                continue
            relative = "benchmarks/bench_compilation_cache.py"
        rows.append({"function": f"{relative}:{name}", "line": line,
                     "calls": calls, "primitive_calls": primitive,
                     "self_ms": own * 1000, "cumulative_ms": cumulative * 1000})
    return {"instrumented_wall_ms": elapsed, "subprocess_run": subprocess_row,
            "functions": sorted(rows, key=lambda row: row["cumulative_ms"], reverse=True)}


def worker(target, mode):
    # A fresh interpreter/import/device setup is deliberately outside the clock.
    cached_expression.parse_ir()
    _, _, validate = inputs(target, 257)
    if mode == "warm":
        seed = compile_call(target)
        check_artifact("miss", seed, validate)
        del seed
    check = lambda artifact: check_artifact(mode, artifact, validate)
    if mode == "restart":
        return measure({"restart": lambda: compile_call(target)}, 1, 0,
                       lambda name, artifact: check_artifact(name, artifact, validate))["restart"]
    return profile_compile(lambda: compile_call(target, cache=mode != "uncached"), check)


def worker_identity():
    return {"extension_sha256": hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def child(target, mode, directory):
    env = {**os.environ, "TENSORCX_CACHE_DIR": str(directory), "TENSORCX_DISABLE_CACHE": "0"}
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--target", target,
                             "--_worker", mode], env=env, capture_output=True, text=True,
                            check=True, timeout=600)
    report = json.loads(result.stdout)
    if report.pop("worker_identity") != worker_identity():
        raise RuntimeError("benchmark worker loaded a different extension or script")
    return report


def host_observation(target):
    result = {"load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None}
    if target == "cuda" and shutil.which("nvidia-smi"):
        try:
            def query(fields):
                return subprocess.run(["nvidia-smi", *fields, "--format=csv,noheader,nounits"],
                                      check=True, capture_output=True, text=True, timeout=10).stdout.strip()
            result["gpu"] = query(["--query-gpu=name,compute_cap,driver_version,memory.used,utilization.gpu"])
            rows = csv.reader(query(["--query-compute-apps=pid,used_gpu_memory"]).splitlines())
            # PIDs are used only to exclude this process and never enter a report.
            result["other_compute_processes"] = [{"memory_mib": memory.strip()}
                                                 for pid, memory in rows if int(pid) != os.getpid()]
        except (OSError, ValueError, subprocess.SubprocessError):
            result["gpu_observation_unavailable"] = True
    return result


def metadata():
    def command(*args):
        return subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()

    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    return {"started_utc": datetime.now(timezone.utc).isoformat(),
            "revision": command("git", "rev-parse", "HEAD"),
            "git_dirty": bool(command("git", "status", "--porcelain")),
            "tracked_diff_sha256": hashlib.sha256(command("git", "diff", "HEAD").encode()).hexdigest(),
            "benchmark_sha256": digest(__file__), "extension_sha256": digest(_core.__file__),
            "tensorcx": cx.__version__, "python": platform.python_version(),
            "system": platform.system(), "machine": platform.machine(), "libc": platform.libc_ver(),
            "numpy": np.__version__}


def run(args):
    report = {"schema_version": 2, "target": args.target, "metadata": metadata(),
              "parameters": {"repeats": args.repeats, "warmup": args.warmup,
                             "restart_repeats": args.restart_repeats, "seed": 17},
              "boundaries": {
                  "compile": "public compile wall latency; IR primed; validation/identity plus CPU/CUDA loading/JIT; Metal metallib only",
                  "restart": "first compile in each new process; excludes interpreter/tensorcx import/device/IR setup; lazy adapter imports included",
                  "launch": "synchronous public launch with resident inputs/artifact, output poisoned before clock; no device-only claim",
                  "excluded": "validation, result cleanup, tensor setup and subprocess startup",
                  "profile": "separate cProfile calls; cumulative times overlap and must not be summed; not benchmark samples",
                  "host": "OS/driver caches are not cleared; shared host, no exclusivity or universal speedup claim"}}
    report["metadata"]["host_before"] = host_observation(args.target)
    cached_expression.parse_ir()
    _, _, validate = inputs(args.target, 257)
    with tempfile.TemporaryDirectory(prefix="tensorcx-cache-benchmark-") as directory:
        root = Path(directory)
        with private_cache(root / "cache"):
            initial = measure({"miss": lambda: compile_call(args.target)}, 1, 0,
                              lambda name, value: check_artifact(name, value, validate))
            report["initial_cache_miss_ms"] = initial["miss"]["samples_ms"][0]
            report["compile"] = measure(
                {"uncached": lambda: compile_call(args.target, cache=False),
                 "warm": lambda: compile_call(args.target)}, args.repeats, args.warmup,
                lambda name, value: check_artifact(name, value, validate))
            restarts = [child(args.target, "restart", root / "cache")["samples_ms"][0]
                        for _ in range(args.restart_repeats)]
            report["compile"]["restart"] = summary(restarts)
            artifact = compile_call(args.target)
            check_artifact("warm", artifact, validate)
            launches = []
            for count in args.sizes:
                arguments, expected, _ = inputs(args.target, count)
                arguments = list(arguments)
                poison = np.full(count, np.nan, dtype=np.float32)
                def prepare_launch(name):
                    arguments[2] = cx.tensor(poison, device=args.target)
                def check_launch(name, value):
                    np.testing.assert_allclose(value.numpy(), expected, rtol=1e-6, atol=1e-6)
                timings = measure({"resident": lambda: artifact.launch(*arguments)},
                                  args.repeats, args.warmup, check_launch, prepare_launch)
                launches.append({"elements": count, "cpu_parity": True, **timings["resident"]})
            report["launch"] = launches
            del artifact
            if args.profile:
                report["profiles"] = {
                    mode: child(args.target, mode, root / mode)
                    for mode in ("uncached", "miss", "warm")}
                # Same persistent artifact, fresh identity/hash memoization.
                report["profiles"]["restart"] = child(args.target, "restart_profile", root / "cache")
    report["metadata"]["host_after"] = host_observation(args.target)
    report["cpu_parity"] = True
    return public_report(report, root=ROOT)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("cpu", "cuda", "metal"), default="cpu")
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--restart-repeats", type=positive_int, default=3)
    parser.add_argument("--sizes", type=sizes_arg, default=DEFAULT_SIZES)
    parser.add_argument("--profile", action="store_true", help="separate cProfile runs; no raw paths/dumps")
    parser.add_argument("--output", type=Path, help="write complete JSON only after all checks pass")
    parser.add_argument("--_worker", choices=("uncached", "miss", "warm", "restart", "restart_profile"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    if args._worker:
        # restart_profile is a hit without this child's warm-up compilation.
        mode = args._worker
        if mode == "restart_profile":
            cached_expression.parse_ir()
            _, _, validate = inputs(args.target, 257)
            report = profile_compile(lambda: compile_call(args.target),
                                     lambda value: check_artifact("restart", value, validate))
        else:
            report = worker(args.target, mode)
        report["worker_identity"] = worker_identity()
    else:
        report = run(args)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
