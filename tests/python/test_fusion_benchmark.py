"""Manual fusion parity, measurement boundaries and fail-closed publication."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import tensorcx as cx
from tensorcx import _core
from tensorcx._compiler.cpu import toolchain

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/bench_cuda_fusion.py"
sys.path.insert(0, str(SCRIPT.parent))
try:
    spec = importlib.util.spec_from_file_location("bench_cuda_fusion", SCRIPT)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
finally:
    sys.path.pop(0)


def test_fusion_measure_balances_all_positions_and_excludes_cleanup(monkeypatch):
    clock, calls = [0], []

    class Result:
        def __del__(self):
            clock[0] += 1_000_000_000
            calls.append("cleanup")

    def call(name):
        calls.append(name)
        clock[0] += 2_000_000
        return Result()

    def validate(result):
        assert isinstance(result, Result)
        clock[0] += 3_000_000_000
        calls.append("validate")

    monkeypatch.setattr(bench.time, "perf_counter_ns", lambda: clock[0])
    result = bench.measure({name: lambda name=name: call(name) for name in "abc"}, validate, 12, 1)
    assert all(row["samples_ms"] == [2.] * 12 for row in result["providers"].values())
    orders = result["provider_orders"]
    assert len({tuple(order) for order in orders}) == 6
    assert orders[:6] == orders[6:]
    for name in "abc":
        assert [sum(order[position] == name for order in orders) for position in range(3)] == [4] * 3
    assert calls[1::3] == ["validate"] * 39 and calls[2::3] == ["cleanup"] * 39


def run_script(output, env, *extra):
    return subprocess.run([sys.executable, str(SCRIPT), "--sizes", "17", "--repeats", "6", "--warmup", "1",
                           "--diagnostic-repeats", "1", "--output", str(output), *extra],
                          cwd=ROOT, env=env, text=True, capture_output=True, timeout=180)


def test_fusion_missing_cuda_preserves_existing_report(tmp_path):
    output = tmp_path / "report.json"
    output.write_text("previous accepted report")
    result = run_script(output, {**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert result.returncode != 0 and "CUDA is required" in result.stderr
    assert output.read_text() == "previous accepted report"


@pytest.mark.parametrize("repeats", [1, 5, 7])
def test_fusion_unbalanced_repeats_rejected(repeats):
    with pytest.raises(argparse.ArgumentTypeError, match="multiple of six"):
        bench.balanced_repeats(str(repeats))
    with pytest.raises(ValueError, match="permutation count"):
        bench.measure(dict.fromkeys("abc"), lambda result: None, repeats, 1)


@pytest.mark.parametrize("failure", ["parity", "diagnostic"])
def test_fusion_failed_run_cannot_replace_report(monkeypatch, tmp_path, failure):
    output = tmp_path / "report.json"
    output.write_text("previous accepted report")
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output", str(output)])

    def fail(args):
        if failure == "parity":
            bench.measure({"broken": lambda: cx.tensor([3.])}, bench.validate_against(np.array([2.])), 1, 1)
        else:
            class Profile:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def report(self): return {"complete": False}
            monkeypatch.setattr(bench.cx, "profile", lambda **kwargs: Profile())
            bench.diagnose(lambda: None, lambda result: None, 1, 1, 1)

    monkeypatch.setattr(bench, "run", fail)
    with pytest.raises((RuntimeError, AssertionError)):
        bench.main()
    assert output.read_text() == "previous accepted report"


def test_fusion_wrong_launch_count_fails(monkeypatch):
    class Profile:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def report(self): return {"complete": True, "device_events": []}
    monkeypatch.setattr(bench.cx, "profile", lambda **kwargs: Profile())
    with pytest.raises(RuntimeError, match="kernel count"):
        bench.diagnose(lambda: None, lambda result: None, 1, 1, 1000)


@pytest.mark.parametrize("sizes", ["0", "-1", "1,1", "x"])
def test_fusion_bad_sizes(sizes):
    with pytest.raises(argparse.ArgumentTypeError):
        bench.sizes_arg(sizes)


def require_target(target):
    try:
        if target == "cuda":
            if not cx.is_available("cuda"):
                raise RuntimeError("CUDA unavailable")
            _core._cuda_kernel_support()
        toolchain()
    except RuntimeError as error:
        required = "TENSORCX_REQUIRE_MLIR_CUDA" if target == "cuda" else "TENSORCX_REQUIRE_MLIR"
        if os.environ.get(required):
            pytest.fail(str(error))
        pytest.skip(str(error))


@pytest.fixture(scope="module", params=["cpu", "cuda"])
def compiled(request):
    require_target(request.param)
    return request.param, bench.compile_kernels(request.param)


@pytest.mark.parametrize("size", [0, 1, 17, 257])
@pytest.mark.parametrize("workload", ["two", "four"])
def test_fusion_provider_parity_and_immutable_alias_template(compiled, size, workload):
    device, kernels = compiled
    rng = np.random.default_rng(791)
    arrays = [rng.uniform(-1, 1, size).astype(np.float32) for _ in range(2)]
    a, b = [cx.tensor(values, device=device) for values in arrays]
    cpu_a, cpu_b = [cx.tensor(values, device="cpu") for values in arrays]
    expected = (cpu_a + cpu_b) * cpu_b
    if workload == "four":
        expected = (expected - cpu_a) * cpu_b
    validate = bench.validate_against(expected.numpy())
    for call in bench.providers(kernels, a, b, workload).values():
        output = call()
        validate(output)
        np.testing.assert_array_equal(a.numpy(), arrays[0])
        np.testing.assert_array_equal(b.numpy(), arrays[1])


def test_fusion_real_cuda_report_and_failures(tmp_path):
    require_target("cuda")
    output = tmp_path / "report.json"
    result = run_script(output, os.environ.copy())
    assert result.returncode == 0, result.stderr
    text = output.read_text()
    report = json.loads(text)
    assert report["publication"]["format"] == "cortex-public-report-v1"
    assert str(ROOT) not in text and "GPU-" not in text
    assert len(report["metadata"]["benchmark_sha256"]) == 64
    assert len(report["results"]) == 4
    for row in report["results"]:
        assert row["cpu_parity"] and row["inputs_unchanged"]
        assert len({tuple(order) for order in row["wall"]["provider_orders"]}) == 6
        for name, wall in row["wall"]["providers"].items():
            assert len(wall["samples_ms"]) == 6 and 0 < wall["p10_ms"] <= wall["median_ms"] <= wall["p90_ms"]
            diagnostic = row["diagnostics"][name]
            assert diagnostic["complete"]
            assert diagnostic["kernel_counts"] == [1 if name == "mlir_fused" else row["operations"]]
            assert diagnostic["device_interval_sum"]["median_ms"] >= 0
    original = output.read_bytes()
    for extra, env, message in (([], {"TENSORCX_LLVM_BIN": str(tmp_path)}, "MLIR requires executable"),
                                (["--max-events", "1"], {}, "incomplete fusion diagnostic")):
        result = run_script(output, {**os.environ, **env}, *extra)
        assert result.returncode != 0 and message in result.stderr
        assert output.read_bytes() == original
