"""Research-only MLIR/PTX toolchain probe; does not enable a runtime target."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
import tensorcx as cx
from tensorcx._compiler.cpu import toolchain

HERE = Path(__file__).resolve().parent


@cx.experimental.kernel
def reference_add(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = a[i] + b[i]


def run(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed ({result.returncode}):\n{result.stderr[-4000:]}")
    return result.stdout


def extract_ptx(serialized: str) -> str:
    # Read the one assembly string in this fixed research fixture, not arbitrary
    # MLIR. Never eval generated text. MLIR escapes bytes as two hex digits.
    matches = re.findall(r'assembly = "((?:\\[0-9A-Fa-f]{2}|\\["\\]|[^"\\])*)"', serialized)
    if len(matches) != 1:
        raise ValueError("expected exactly one serialized PTX assembly object")
    ptx = re.sub(r'\\([0-9A-Fa-f]{2}|["\\])',
                 lambda m: chr(int(m[1], 16)) if len(m[1]) == 2 else m[1], matches[0])
    if not re.search(r"^\.version 7\.8$", ptx, re.M):
        raise ValueError("expected PTX ISA 7.8")
    if not re.search(r"^\.target sm_52$", ptx, re.M):
        raise ValueError("expected sm_52")
    if not re.search(r"^\.address_size 64$", ptx, re.M):
        raise ValueError("expected 64-bit device pointers")
    entry = re.search(r"\.visible \.entry tensorcx_add\((.*?)\)", ptx, re.S)
    if not entry:
        raise ValueError("missing tensorcx_add entry")
    params = re.findall(r"\.param \.([a-z0-9]+)", entry[1])
    if params != ["u64", "u64", "u64", "u32"]:
        raise ValueError(f"unexpected device parameter ABI: {params}")
    return ptx


def lower(root: Path) -> Path:
    opt = toolchain()["mlir-opt"]
    lowered = root / "nvvm.mlir"
    serialized = root / "serialized.mlir"
    run([opt, str(HERE / "cuda_add.mlir"), "--convert-scf-to-cf",
         "--convert-gpu-to-nvvm=index-bitwidth=64",
         "--reconcile-unrealized-casts", "-o", str(lowered)])
    run([opt, str(lowered), "--nvvm-attach-target=chip=sm_52 features=+ptx78 O=2",
         "--gpu-module-to-binary=format=assembly", "-o", str(serialized)])
    ptx = root / "add.ptx"
    ptx.write_text(extract_ptx(serialized.read_text()))
    return ptx


def verify_case(case):
    size, threads, block = case["size"], case["threads"], case["block"]
    a = cx.tensor(np.arange(size, dtype=np.float32) * 0.25, device="cpu")
    b = cx.tensor((np.arange(size) % 7).astype(np.float32) * 0.5, device="cpu")
    out = a if case["alias"] else cx.tensor(np.full(size, -11, dtype=np.float32), device="cpu")
    expected = reference_add.reference(a, b, out, threads, thread_count=threads, block_size=block)
    np.testing.assert_allclose(np.array(case["result"], dtype=np.float32),
                               expected.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(case["result"][:threads], dtype=np.float32),
                               (a + b).numpy()[:threads], rtol=1e-6, atol=1e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-only", action="store_true", help="Check device lowering without CUDA/GPU")
    parser.add_argument("--cxx", default="g++-13", help="Host compiler with CUDA headers/libraries on its search path")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="tensorcx-cuda-probe-") as directory:
        root = Path(directory)
        ptx = lower(root)
        if args.compile_only:
            print("PASS: LLVM 21.1.8 -> NVVM -> PTX 7.8, sm_52, pointer/u32 ABI")
            return
        assembler = shutil.which("ptxas")
        if not assembler or "release 12.4," not in run([assembler, "--version"]):
            raise RuntimeError("hardware probe requires CUDA 12.4 ptxas")
        run([assembler, "-arch=sm_52", str(ptx), "-o", str(root / "add.cubin")])
        compiler = shutil.which(args.cxx)
        if not compiler:
            raise RuntimeError(f"missing host compiler: {args.cxx}")
        executable = root / "probe"
        run([compiler, "-std=c++20", "-O2", str(HERE / "cuda_driver_probe.cpp"),
             "-lcuda", "-lcudart", "-o", str(executable)])
        report = json.loads(run([str(executable), str(ptx)]))
        for case in report["cases"]:
            verify_case(case)
        print(json.dumps({"device": report["device"], "cases": len(report["cases"]),
                          "context_restored": report["context_restored"],
                          "runtime_version": report["runtime_version"],
                          "driver_api_version": report["driver_api_version"],
                          "ptx": "7.8", "sm": "52", "result": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()
