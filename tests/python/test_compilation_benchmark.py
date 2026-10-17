"""Check benchmark clocks, cache state, fresh-process isolation and report safety."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import tensorcx as cx
from tensorcx import _core
from tensorcx._compiler.cpu import toolchain

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/bench_compilation_cache.py"
spec = importlib.util.spec_from_file_location("bench_compilation_cache", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_measure_alternates_and_excludes_validation_and_destruction(monkeypatch):
    clock, events = [0], []

    class Result:
        def __del__(self):
            clock[0] += 10_000_000
            events.append("release")

    def call(name):
        events.append(name)
        clock[0] += 2_000_000
        return Result()

    def prepare(name):
        clock[0] += 500_000_000

    def validate(name, result):
        assert isinstance(result, Result)
        events.append("validate")
        clock[0] += 100_000_000

    monkeypatch.setattr(bench.time, "perf_counter_ns", lambda: clock[0])
    measured = bench.measure({name: lambda name=name: call(name) for name in ("a", "b")}, 3, 1, validate, prepare)
    assert [event for event in events if event in ("a", "b")] == ["a", "b", "b", "a", "a", "b", "b", "a"]
    assert events[1::3] == ["validate"] * 8
    assert events[2::3] == ["release"] * 8
    assert all(row["samples_ms"] == [2.0] * 3 for row in measured.values())


def test_wrong_output_or_cache_state_cannot_pass():
    _, expected, validate = bench.inputs("cpu", 17)

    class Wrong:
        cache_hit = True

        def launch(self, *args):
            return cx.zeros(expected.shape, device="cpu")

    with pytest.raises(AssertionError):
        bench.check_artifact("warm", Wrong(), validate)
    with pytest.raises(AssertionError, match="cache_hit"):
        bench.check_artifact("miss", Wrong(), lambda value: pytest.fail("validation reached"))


def test_cache_environment_restored_on_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORCX_DISABLE_CACHE", "on")
    monkeypatch.delenv("TENSORCX_CACHE_DIR", raising=False)
    with pytest.raises(RuntimeError):
        with bench.private_cache(tmp_path):
            assert os.environ["TENSORCX_CACHE_DIR"] == str(tmp_path)
            assert os.environ["TENSORCX_DISABLE_CACHE"] == "0"
            raise RuntimeError("fail")
    assert "TENSORCX_CACHE_DIR" not in os.environ
    assert os.environ["TENSORCX_DISABLE_CACHE"] == "on"


def test_profiler_has_no_raw_paths_and_excludes_its_validator(monkeypatch):
    clock = [0]
    monkeypatch.setattr(bench.time, "perf_counter_ns", lambda: clock[0])

    def measured():
        clock[0] += 2_000_000
        return bench.summary([1, 2, 3])

    def validate(result):
        assert result["median_ms"] == 2
        clock[0] += 100_000_000

    profile = bench.profile_compile(measured, validate)
    assert profile["instrumented_wall_ms"] == 2
    assert profile["subprocess_run"]["calls"] == 0
    assert any(row["function"].endswith(":summary") for row in profile["functions"])
    assert all(row["function"].startswith("benchmarks/") for row in profile["functions"])
    assert not any("validate" in row["function"] for row in profile["functions"])
    assert str(ROOT) not in json.dumps(profile)


def test_fresh_child_preserves_private_cache_and_forces_enable(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"samples_ms": [3], "worker_identity": bench.worker_identity()}))

    monkeypatch.setenv("TENSORCX_DISABLE_CACHE", "1")
    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    assert bench.child("cpu", "restart", tmp_path)["samples_ms"] == [3]
    command, kwargs = calls[0]
    assert command[0] == sys.executable
    assert command[-2:] == ["--_worker", "restart"]
    assert kwargs["env"]["TENSORCX_CACHE_DIR"] == str(tmp_path)
    assert kwargs["env"]["TENSORCX_DISABLE_CACHE"] == "0"
    assert kwargs["check"] and kwargs["timeout"] == 600
    assert os.environ["TENSORCX_DISABLE_CACHE"] == "1"


@pytest.mark.parametrize("value", ["", "0", "-1", "1,,2", "x", "1,1"])
def test_invalid_sizes(value):
    with pytest.raises(argparse.ArgumentTypeError):
        bench.sizes_arg(value)


def require_benchmark_checkout():
    # Portable sdist/runtime probes intentionally have neither .git nor git CLI.
    # The reporting CLI documents a Git source checkout as its prerequisite.
    if not shutil.which("git") or not (ROOT / ".git").exists():
        pytest.skip("benchmark reporting requires a Git source checkout")


def test_missing_toolchain_does_not_publish_report(tmp_path):
    require_benchmark_checkout()
    output = tmp_path / "failed.json"
    result = subprocess.run([sys.executable, str(SCRIPT), "--output", str(output)],
                            env={**os.environ, "TENSORCX_LLVM_BIN": str(tmp_path)},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    reason = "MLIR requires executable" if _core._cpu_kernel_supported() else "MLIR CPU runtime requires Linux x86_64"
    assert reason in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("target", ["cpu", "cuda", "metal"])
def test_real_report(target, tmp_path):
    require_benchmark_checkout()
    if target == "cpu" and not _core._cpu_kernel_supported():
        pytest.skip("MLIR CPU runtime requires Linux x86_64")
    if not cx.is_available(target):
        if target == "cuda" and os.environ.get("TENSORCX_REQUIRE_MLIR_CUDA"):
            pytest.fail("CUDA required")
        pytest.skip(f"{target} unavailable")
    if target == "metal":
        if not shutil.which("xcrun"):
            pytest.skip("Metal tools unavailable")
    else:
        try:
            toolchain()
        except RuntimeError as error:
            if os.environ.get("TENSORCX_REQUIRE_MLIR") or (target == "cuda" and os.environ.get("TENSORCX_REQUIRE_MLIR_CUDA")):
                pytest.fail(str(error))
            pytest.skip(str(error))
    output = tmp_path / "report.json"
    command = [sys.executable, str(SCRIPT), "--target", target, "--sizes", "1,257",
               "--repeats", "2", "--warmup", "1", "--restart-repeats", "1", "--output", str(output)]
    if target != "metal":
        command.append("--profile")
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["schema_version"] == 2 and report["target"] == target
    assert report["cpu_parity"]
    assert len(report["metadata"]["extension_sha256"]) == 64
    assert str(ROOT) not in output.read_text() and str(tmp_path) not in output.read_text()
    assert "GPU-" not in output.read_text()
    assert report["initial_cache_miss_ms"] > 0
    for name, row in report["compile"].items():
        assert len(row["samples_ms"]) == (1 if name == "restart" else 2)
        assert 0 < row["p10_ms"] <= row["median_ms"] <= row["p90_ms"]
    assert [row["elements"] for row in report["launch"]] == [1, 257]
    assert all(row["cpu_parity"] and len(row["samples_ms"]) == 2 for row in report["launch"])
    if target != "metal":
        assert set(report["profiles"]) == {"uncached", "miss", "warm", "restart"}
        for mode, row in report["profiles"].items():
            assert row["instrumented_wall_ms"] > 0
            assert row["subprocess_run"]["calls"] > 0
            assert any(item["function"].endswith("cache.py:descriptor") for item in row["functions"]) or mode == "uncached"


def test_installed_package_profile_labels_are_retained_and_safe(monkeypatch, tmp_path):
    package = tmp_path / "site-packages" / "tensorcx"
    monkeypatch.setattr(bench, "PACKAGE_ROOT", package)

    def installed_function():
        return 17

    installed_function.__code__ = installed_function.__code__.replace(
        co_filename=str(package / "_compiler" / "cache.py"))
    profile = bench.profile_compile(installed_function, lambda value: None)
    assert any(row["function"] == "python/tensorcx/_compiler/cache.py:installed_function"
               for row in profile["functions"])
    assert str(tmp_path) not in json.dumps(profile)


def test_child_identity_mismatch_is_rejected(monkeypatch, tmp_path):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(
            {"worker_identity": {"extension_sha256": "wrong"}, "samples_ms": [1]}))
    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="different extension or script"):
        bench.child("cpu", "restart", tmp_path)


def test_repeated_in_place_noop_cannot_reuse_prior_correct_output():
    _, expected, validate = bench.inputs("cpu", 17)

    class InPlace:
        calls = 0

        def launch(self, a, b, out, count):
            self.calls += 1
            if self.calls == 1:
                # Model Metal mutating the supplied output, then a later no-op.
                out._impl = cx.tensor(expected, device="cpu")._impl
            return out

    artifact = InPlace()
    validate(artifact)
    with pytest.raises(AssertionError):
        validate(artifact)
