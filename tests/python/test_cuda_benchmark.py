"""Benchmark measurement boundaries and strict, machine-readable CUDA workflow."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import cortex_runtime as cx
from cortex_runtime import _core
from cortex_runtime._compiler.cpu import toolchain


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/bench_cuda.py"
spec = importlib.util.spec_from_file_location("bench_cuda", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_measure_excludes_validation_and_cleanup_and_alternates(monkeypatch):
    clock = [0]
    events = []

    class Result:
        def __del__(self):
            clock[0] += 1000000000
            events.append("release")

    def call(name):
        events.append(name)
        clock[0] += 2000000
        return Result()

    def validate(result):
        assert isinstance(result, Result)
        clock[0] += 3000000000
        events.append("validate")

    monkeypatch.setattr(bench.time, "perf_counter_ns", lambda: clock[0])
    measured = bench.measure({"a": lambda: call("a"), "b": lambda: call("b")}, 3, 1, validate)
    assert [event for event in events if event in ("a", "b")] == ["a", "b", "b", "a", "a", "b", "b", "a"]
    assert events[1::3] == ["validate"] * 8
    assert events[2::3] == ["release"] * 8
    assert all(row["samples_ms"] == [2.0] * 3 for row in measured.values())


def test_incorrect_output_cannot_produce_measurements():
    validate = bench.parity(cx.tensor([2.0], device="cpu"))
    with pytest.raises(AssertionError):
        bench.measure({"broken": lambda: np.array([3.0], dtype=np.float32)}, 1, 1, validate)


def test_compute_process_snapshot_excludes_self(monkeypatch):
    monkeypatch.setattr(bench, "command", lambda args: f"{os.getpid()}, 350, GPU-test\n42, 100, GPU-other")
    assert bench.compute_processes() == [{"pid": 42, "memory_mib": "100", "gpu_uuid": "GPU-other"}]
    monkeypatch.setattr(bench, "command", lambda args: "")
    assert bench.compute_processes() == []


@pytest.mark.parametrize("value", ["", "0", "-1", "1,,2", "x", "1,1"])
def test_bad_sizes(value):
    with pytest.raises(argparse.ArgumentTypeError):
        bench.sizes_arg(value)


def run_script(output, env):
    return subprocess.run([sys.executable, str(SCRIPT), "--sizes", "17", "--warmup", "1",
                           "--repeats", "3", "--compile-repeats", "1", "--output", str(output)],
                          cwd=ROOT, env=env, text=True, capture_output=True, timeout=180)


def test_missing_cuda_is_error_without_report(tmp_path):
    output = tmp_path / "unavailable.json"
    result = run_script(output, {**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert result.returncode != 0
    assert "CUDA is required" in result.stderr
    assert not output.exists()


def test_real_cuda_report(tmp_path):
    try:
        if not hasattr(_core, "_cuda_kernel_support"):
            raise RuntimeError("CUDA build unavailable")
        _core._cuda_kernel_support()
        toolchain()
    except RuntimeError as error:
        if os.environ.get("CORTEX_REQUIRE_MLIR_CUDA"):
            pytest.fail(str(error))
        pytest.skip(str(error))
    output = tmp_path / "report.json"
    result = run_script(output, os.environ.copy())
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["schema_version"] == 1
    assert report["metadata"]["gpu_before"]
    assert isinstance(report["metadata"]["other_compute_processes_observed"], bool)
    assert len(report["metadata"]["extension_sha256"]) == 64
    cases = {row["case"] for row in report["results"]}
    assert {"compile", "launch_one_element", "host_to_device", "device_to_host", "add",
            "exp", "gelu", "silu", "sum", "max", "mean", "softmax", "rmsnorm", "layernorm",
            "matmul", "expression_resident", "expression_end_to_end", "mlp_resident", "mlp_end_to_end"} == cases
    for row in report["results"]:
        assert row["cpu_parity"]
        for provider in row["providers"].values():
            assert len(provider["samples_ms"]) == (1 if row["case"] == "compile" else 3)
            assert 0 < provider["p10_ms"] <= provider["median_ms"] <= provider["p90_ms"]
    missing_tools = tmp_path / "missing-tools.json"
    result = run_script(missing_tools, {**os.environ, "CORTEX_LLVM_BIN": str(tmp_path)})
    assert result.returncode != 0
    assert "MLIR requires executable" in result.stderr
    assert not missing_tools.exists()
