"""Strict CUDA/MLIR benchmark: synchronous public API latency, with CPU parity."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
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
from tensorcx._compiler.cpu import toolchain


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
try:
    from publication_report import public_report
finally:
    sys.path.pop(0)
DEFAULT_SIZES = (1024, 16384, 262144, 1048576, 16777216)


@cx.experimental.kernel
def fused(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        total = a[i] + b[i]
        out[i] = total * b[i]


def forward(x, first, second):
    return cx.softmax(cx.matmul(cx.gelu(cx.layernorm(cx.matmul(x, first), axis=-1)), second), axis=-1)


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


def measure(functions, repeats, warmup, validate):
    """Alternate provider order to reduce ordering bias; exclude result cleanup.

    APIs synchronize before returning. Retain each result until its timer stops,
    then validate and release it outside the measured region. No CUDA event or
    device-only throughput claim is made.
    """
    samples = {name: [] for name in functions}
    names = list(functions)
    for iteration in range(warmup + repeats):
        for name in names if iteration % 2 == 0 else names[::-1]:
            start = time.perf_counter_ns()
            result = functions[name]()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            validate(result)
            del result
            if iteration >= warmup:
                samples[name].append(elapsed)
    return {name: summary(values) for name, values in samples.items()}


def command(args):
    return subprocess.run(args, cwd=ROOT, check=True, text=True,
                          capture_output=True).stdout.strip()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_snapshot():
    return command(["nvidia-smi", "--query-gpu=index,name,compute_cap,driver_version,"
                    "memory.used,temperature.gpu,pstate,clocks.sm,clocks.mem,power.draw,utilization.gpu",
                    "--format=csv"])


def compute_processes():
    raw = command(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                   "--format=csv,noheader,nounits"])
    # PIDs are used only to exclude this benchmark, never written to reports.
    return [{"memory_mib": memory.strip()}
            for pid, memory in csv.reader(raw.splitlines())
            if int(pid) != os.getpid()]


def parity(expected, kind="elementwise"):
    # NumPy copy of CPU result: validation adds no CPU runtime dispatch to timings.
    array = expected.numpy()
    tolerance = {"elementwise": 1e-6, "reduction": 1e-5, "matmul": 1e-4}[kind]

    def validate(result):
        actual = result if isinstance(result, np.ndarray) else result.numpy()
        np.testing.assert_allclose(actual, array, rtol=tolerance, atol=tolerance)
    return validate


def run(args):
    if not cx.is_available("cuda"):
        raise RuntimeError("CUDA is required; this benchmark never skips or falls back")
    # The same gate as compilation: require the validated hardware/toolchain.
    _core._cuda_kernel_support()
    llvm = toolchain()
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "revision": command(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(command(["git", "status", "--porcelain"])),
        "tracked_diff_sha256": hashlib.sha256(command(["git", "diff", "HEAD"]).encode()).hexdigest(),
        "benchmark_sha256": digest(__file__), "extension_sha256": digest(_core.__file__),
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "tensorcx": cx.__version__, "nvcc": command(["nvcc", "--version"]),
        "llvm": {name: command([str(path), "--version"]) for name, path in llvm.items()},
        "gpu_before": gpu_snapshot(), "seed": 17,
        "other_compute_processes_before": compute_processes(),
    }
    rows = []

    def record(name, shape, functions, expected, kind="elementwise", **extra):
        print(f"measuring {name} {shape}", file=sys.stderr, flush=True)
        values = measure(functions, args.repeats, args.warmup, parity(expected, kind))
        rows.append({"case": name, "shape": list(shape), "cpu_parity": True,
                     "providers": values, **extra})

    # Populate the Python IR cache first. Each compile still lowers MLIR and
    # loads a fresh native module; driver/OS caches are deliberately not cleared.
    fused.parse_ir()
    probe = cx.ones((257,), device="cuda")
    expected_probe = (probe.cpu() + probe.cpu()) * probe.cpu()
    check_probe = parity(expected_probe)
    compilation = measure(
        {"mlir_cuda": lambda: fused.compile(target="cuda", compiler="mlir")},
        args.compile_repeats, 1,
        lambda artifact: check_probe(artifact.launch(probe, probe, probe, 257)),
    )
    rows.append({"case": "compile", "shape": [], "cpu_parity": True,
                 "providers": compilation})
    compiled = fused.compile(target="cuda", compiler="mlir")
    one = cx.ones((1,), device="cuda")
    record("launch_one_element", (1,),
           {"mlir_cuda": lambda: compiled.launch(one, one, one, 1)},
           cx.tensor([2.0], device="cpu"))
    rng = np.random.default_rng(17)
    for count in args.sizes:
        a = cx.tensor(rng.uniform(-1, 1, count).astype(np.float32), device="cpu")
        b = cx.tensor(rng.uniform(-1, 1, count).astype(np.float32), device="cpu")
        ga, gb = a.to("cuda"), b.to("cuda")
        template = cx.zeros((count,), device="cuda")
        record("host_to_device", (count,), {"cuda": lambda: a.to("cuda")}, a,
               payload_bytes=count * 4)
        record("device_to_host", (count,), {"cuda": lambda: ga.cpu()}, a,
               payload_bytes=count * 4)
        record("add", (count,), {"cpu": lambda: a + b, "cuda": lambda: ga + gb}, a + b)
        for name in ("exp", "gelu", "silu"):
            operation = getattr(cx, name)
            record(name, (count,), {"cpu": lambda: operation(a), "cuda": lambda: operation(ga)}, operation(a))
        expected = (a + b) * b
        record("expression_resident", (count,), {
            "cpu": lambda: (a + b) * b,
            "cuda_separate": lambda: (ga + gb) * gb,
            "cuda_fused": lambda: compiled.launch(ga, gb, template, count),
        }, expected)

        def separate_end_to_end():
            da, db = a.to("cuda"), b.to("cuda")
            return ((da + db) * db).cpu()

        def fused_end_to_end():
            da, db = a.to("cuda"), b.to("cuda")
            # Reuse the left input as the immutable output template; native
            # launch allocates a private result and leaves both inputs intact.
            return compiled.launch(da, db, da, count).cpu()

        record("expression_end_to_end", (count,), {
            "cuda_separate": separate_end_to_end, "cuda_fused": fused_end_to_end,
        }, expected)
        # Check the documented non-mutation contract after every measured case.
        parity(a)(ga)
        parity(b)(gb)
        np.testing.assert_array_equal(template.numpy(), 0)

    # Include contiguous and strided axes, plus a long single slice to expose
    # serial-reduction bottlenecks. Positive bounded inputs avoid cancellation.
    for shape, axis in [((32, 128), 1), ((32, 4096), 1), ((32, 4096), 0), ((1, 65536), 1)]:
        a = cx.tensor(rng.uniform(0, 1, shape).astype(np.float32), device="cpu")
        ga = a.to("cuda")
        for name in ("sum", "max", "mean", "softmax", "rmsnorm", "layernorm"):
            operation = getattr(cx, name)
            record(name, shape, {"cpu": lambda: operation(a, axis=axis),
                                "cuda": lambda: operation(ga, axis=axis)},
                   operation(a, axis=axis), "reduction", axis=axis)
    for m, k, n in ((32, 64, 128), (129, 67, 65), (256, 256, 256)):
        a = cx.tensor(rng.uniform(-1, 1, (m, k)).astype(np.float32), device="cpu")
        b = cx.tensor(rng.uniform(-1, 1, (k, n)).astype(np.float32), device="cpu")
        ga, gb = a.to("cuda"), b.to("cuda")
        record("matmul", (m, k, n), {"cpu": lambda: cx.matmul(a, b),
                                    "cuda_custom": lambda: cx.matmul(ga, gb, backend="custom")},
               cx.matmul(a, b), "matmul")
    host = [cx.tensor(rng.normal(size=shape).astype(np.float32), device="cpu")
            for shape in ((32, 64), (64, 128), (128, 10))]
    inputs = [t.to("cuda") for t in host]
    expected = forward(*host)
    record("mlp_resident", (32, 64, 128, 10),
           {"cpu": lambda: forward(*host), "cuda": lambda: forward(*inputs)}, expected, "matmul")
    record("mlp_end_to_end", (32, 64, 128, 10),
           {"cuda": lambda: forward(*(t.to("cuda") for t in host)).cpu()}, expected, "matmul")
    metadata["gpu_after"] = gpu_snapshot()
    metadata["other_compute_processes_after"] = compute_processes()
    metadata["other_compute_processes_observed"] = bool(
        metadata["other_compute_processes_before"] or metadata["other_compute_processes_after"])
    if metadata["other_compute_processes_observed"]:
        print("WARNING: other GPU compute processes observed; timings are from a shared GPU.", file=sys.stderr)
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    return public_report({"schema_version": 1, "metadata": metadata,
            "config": {"sizes": list(args.sizes), "repeats": args.repeats,
                       "warmup": args.warmup, "compile_repeats": args.compile_repeats},
            "measurement": "synchronous public API wall time; validation and returned-result cleanup excluded",
            "results": rows}, root=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=sizes_arg, default=DEFAULT_SIZES)
    parser.add_argument("--repeats", type=positive_int, default=31)
    parser.add_argument("--warmup", type=positive_int, default=5)
    parser.add_argument("--compile-repeats", type=positive_int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(run(args), indent=2, allow_nan=False) + "\n"
    if args.output:
        # Only publish a report after all measured outputs passed CPU parity.
        args.output.write_text(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
