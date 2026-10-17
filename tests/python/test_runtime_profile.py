"""Numerical safety, timing accounting, boundedness, lifecycle and thread isolation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import importlib.util
import json
import os
from pathlib import Path
import threading

import numpy as np
import pytest

import tensorcx as cx
from tensorcx import _core
from tensorcx._compiler.cpu import toolchain
import tensorcx.profiling as profiling

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("bench_runtime_profile", ROOT / "benchmarks/bench_runtime_profile.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.fixture(params=[name for name in cx.devices() if name in ("cpu", "cuda")])
def device_name(request):
    return request.param


def check_accounting(report):
    events = report["events"]
    assert report["clock"] == "steady_clock_host" and report["timing"] == "synchronous_host_spans"
    assert report["coverage"]["metal"] == "not instrumented"
    assert sum(e["self_ns"] for e in events) + report["unattributed_ns"] == report["elapsed_ns"]
    assert report["unattributed_ns"] >= 0
    assert [e["id"] for e in events] == list(range(1, len(events) + 1))
    by_id = {e["id"]: e for e in events}
    for event in events:
        assert 0 <= event["self_ns"] <= event["duration_ns"]
        assert 0 <= event["start_ns"] <= event["start_ns"] + event["duration_ns"] <= report["elapsed_ns"]
        if event["parent_id"]:
            parent = by_id[event["parent_id"]]
            assert parent["id"] < event["id"] and parent["start_ns"] <= event["start_ns"]
            assert event["start_ns"] + event["duration_ns"] <= parent["start_ns"] + parent["duration_ns"]
        if report["complete"]:
            children = sum(e["duration_ns"] for e in events if e["parent_id"] == event["id"])
            assert event["self_ns"] + children == event["duration_ns"]


@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (1,), (257,), (2, 3, 5)])
@pytest.mark.parametrize("dtype", [cx.float32, cx.int32])
def test_profile_primitive_parity_and_accounting(device_name, shape, dtype):
    a = np.resize(np.array([-17, 0, 19, 16777217], dtype=dtype), shape)
    b = np.full(shape, 3, dtype=dtype)
    x, y = cx.tensor(a, device=device_name), cx.tensor(b, device=device_name)
    with cx.profile() as profile:
        result = ((x + y) * y).sum()
    expected = ((cx.tensor(a) + cx.tensor(b)) * cx.tensor(b)).sum()
    np.testing.assert_array_equal(result.numpy(), expected.numpy())
    np.testing.assert_array_equal(x.numpy(), a)
    np.testing.assert_array_equal(y.numpy(), b)
    report = profile.report()
    check_accounting(report)
    assert report["complete"] and report["dropped_events"] == 0
    operations = {e["operation"] for e in report["events"] if e["category"] == "dispatch"}
    assert {"add", "multiply", "sum"} <= operations
    assert any(e["category"] == "allocation" and e["bytes"] == 4 for e in report["events"])
    assert sum(row["self_ns"] for row in profile.summary()) == sum(e["self_ns"] for e in report["events"])


def test_profile_transfer_and_temporary_cleanup(device_name):
    host = cx.ones((257,))
    with cx.profile() as profile:
        x = host.to(device_name)
        result = (x + x).cpu()
        del x
    np.testing.assert_array_equal(result.numpy(), np.full(257, 2))
    report = profile.report()
    check_accounting(report)
    if device_name == "cuda":
        copies = {e["name"] for e in report["events"] if e["category"] == "copy_api"}
        assert {"host_to_device", "device_to_host"} <= copies
        assert any(e["category"] == "deallocation" for e in report["events"])
        assert any(e["category"] == "copy_api" and e["bytes"] == 257 * 4 for e in report["events"])
        assert any(e["category"] == "synchronize" for e in report["events"])
    else:
        assert all(e["backend"] == "cpu" for e in report["events"])


def test_profile_boolean_broadcast_and_composite(device_name):
    a = cx.tensor([[-5, 7, 2]], device=device_name)
    b = cx.tensor([[1], [9]], device=device_name)
    with cx.profile() as p:
        values = cx.where(a < b, a, b)
        truth = (values > 0).all()
        floats = values.astype(cx.float32)
        result = cx.linear(floats, cx.ones((4, 3), device=device_name))
    np.testing.assert_array_equal(values.numpy(), [[-5, 1, 1], [-5, 7, 2]])
    assert not truth.numpy()
    np.testing.assert_array_equal(result.numpy(), [[-3] * 4, [4] * 4])
    check_accounting(p.report())


def test_profile_cpu_work_categories_and_imports():
    with cx.profile() as imports:
        x = cx.tensor(np.arange(12, dtype=np.float32).reshape(3, 4))
        listed = cx.tensor([True, False, True])
        native = _core.tensor([1., 2., 3.], None, "cpu")
    assert listed.numpy().tolist() == [True, False, True]
    assert len(imports.report()["events"]) >= 3
    assert all(e["category"] == "copy_allocate" for e in imports.report()["events"])
    check_accounting(imports.report())
    with cx.profile() as p:
        product = x @ cx.transpose(x)
        all_true = listed.all()
        any_true = listed.any()
        selected = x[listed]
        joined = cx.concat([x, x], axis=0)
        sliced = x[:, 1:3]
    np.testing.assert_array_equal(product.numpy(), np.arange(12).reshape(3, 4) @ np.arange(12).reshape(3, 4).T)
    assert not all_true.numpy() and any_true.numpy()
    np.testing.assert_array_equal(selected.numpy(), x.numpy()[[True, False, True]])
    np.testing.assert_array_equal(joined.numpy(), np.concatenate([x.numpy(), x.numpy()]))
    np.testing.assert_array_equal(sliced.numpy(), x.numpy()[:, 1:3])
    work = {(e["operation"], e["category"]) for e in p.report()["events"]}
    assert {(op, "compute_sync") for op in ("matmul", "all", "any", "masked_select", "slice")} <= work
    assert ("concat", "copy_api") in work
    check_accounting(p.report())


def test_profile_cuda_memset_is_api_submission(device_name):
    with cx.profile() as p:
        result = cx.ones((257,), dtype=cx.bool, device=device_name)
    assert result.numpy().all()
    if device_name == "cuda":
        events = p.report()["events"]
        assert any(e["name"] == "cudaMemset" and e["category"] == "memset_api" for e in events)
        assert any(e["category"] == "synchronize" for e in events)
    check_accounting(p.report())


@pytest.mark.parametrize("bad", [True, np.bool_(False), 0, -1, 100001, 2**100, 1.5, "5", None])
def test_profile_limit_validation(bad):
    with pytest.raises(ValueError, match="max_events"):
        cx.profile(max_events=bad)
    with pytest.raises((ValueError, TypeError, OverflowError)):
        _core._profile_begin(bad)


def test_profile_nested_wrong_token_and_exception_cleanup(device_name):
    x = cx.ones((3,), dtype=cx.int32, device=device_name)
    with cx.profile() as p:
        with pytest.raises(RuntimeError, match="nested"):
            with cx.profile():
                pytest.fail("nested profile entered")
        with pytest.raises(RuntimeError, match="owning thread"):
            _core._profile_end(0)
        result = x + x
    assert result.numpy().tolist() == [2, 2, 2]
    with pytest.raises(ValueError, match="float32"):
        with cx.profile() as failed:
            x / x
    assert failed.report()["events"]
    check_accounting(failed.report())
    with cx.profile() as following:
        -x
    check_accounting(following.report())
    with pytest.raises(RuntimeError, match="single-use"):
        p.__enter__()


def test_profile_cap_is_reported_without_changing_results(device_name):
    with cx.profile(max_events=np.int64(1)) as p:
        x = cx.ones((257,), device=device_name)
        y = (x + x) * x
    assert y.numpy().tolist() == [2.0] * 257
    report = p.report()
    assert len(report["events"]) == 1 and report["dropped_events"] > 0 and not report["complete"]
    check_accounting(report)
    with cx.profile() as complete:
        x + x
    assert complete.report()["complete"]


def test_profile_threads_are_isolated_and_wrong_thread_cannot_close():
    barrier = threading.Barrier(2)
    def worker(size):
        with cx.profile() as p:
            barrier.wait(timeout=10)
            x = cx.ones((size,))
            y = x + x
        assert y.numpy().tolist() == [2] * size
        return p.report()
    with cx.profile() as main, ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, size) for size in (3, 7)]
        for size, future in zip((3, 7), futures):
            report = future.result(timeout=30)
            check_accounting(report)
            assert {e["bytes"] for e in report["events"] if e["category"] == "allocation"} == {size * 4}
        error = executor.submit(main.__exit__, None, None, None)
        with pytest.raises(RuntimeError, match="owning thread"):
            error.result(timeout=30)
    assert main.report()["events"] == []


def test_profile_report_snapshot_export_and_abandonment(tmp_path, monkeypatch):
    unused = cx.profile()
    with pytest.raises(RuntimeError, match="after context exit"):
        unused.report()
    p = unused.__enter__()
    cx.ones((3,))
    p.__exit__(None, None, None)
    snapshot = p.report()
    p.report()["events"].clear()
    assert p.report() == snapshot
    trace = tmp_path / "trace.json"
    p.export_chrome_trace(trace)
    value = json.loads(trace.read_text())
    assert all(e["pid"] == e["tid"] == 0 for e in value["traceEvents"])
    assert len(value["traceEvents"]) == len(snapshot["events"])
    assert str(ROOT) not in trace.read_text() and "GPU-" not in trace.read_text()
    original = trace.read_text()
    with monkeypatch.context() as patch:
        patch.setattr(profiling.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("write failed")))
        with pytest.raises(OSError, match="write failed"):
            p.export_chrome_trace(trace)
    assert trace.read_text() == original
    assert not list(tmp_path.glob(".tensorcx-profile-*"))
    abandoned = cx.profile()
    abandoned.__enter__()
    del abandoned
    gc.collect()
    with cx.profile() as clean:
        pass
    assert clean.report()["events"] == []


def test_profile_metal_coverage_is_explicit():
    with cx.profile() as p:
        if cx.is_available("metal"):
            result = cx.ones((3,), device="metal") + cx.ones((3,), device="metal")
        else:
            result = cx.ones((3,)) + cx.ones((3,))
    assert result.numpy().tolist() == [2] * 3
    assert p.report()["coverage"]["metal"] == "not instrumented"
    assert not any(e["backend"] == "metal" for e in p.report()["events"])


@cx.experimental.kernel
def profiled_generated(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = (a[i] + b[i]) * b[i]


def test_profile_generated_kernel(device_name):
    required = os.environ.get("TENSORCX_REQUIRE_MLIR")
    try:
        toolchain()
        if device_name == "cpu" and not _core._cpu_kernel_supported():
            raise RuntimeError("CPU native generated target unavailable")
        if device_name == "cuda":
            _core._cuda_kernel_support()
    except RuntimeError:
        if required:
            raise
        pytest.skip("optional native LLVM target unavailable")
    artifact = profiled_generated.compile(target=device_name, compiler="mlir", cache=False)
    x = cx.ones((257,), device=device_name)
    output = cx.zeros((257,), device=device_name)
    with cx.profile() as p:
        result = artifact.launch(x, x, output, 257, thread_count=257, block_size=7)
    assert result.numpy().tolist() == [2] * 257
    assert output.numpy().tolist() == [0] * 257
    check_accounting(p.report())
    assert any(e["name"] == "generated_kernel" for e in p.report()["events"])


def test_runtime_profile_benchmark_modes_and_untimed_cleanup(monkeypatch):
    clock, calls = [0], []
    class Result:
        def __del__(self):
            clock[0] += 100_000_000
    class FakeProfile:
        def __enter__(self):
            calls.append("profiled"); clock[0] += 100_000_000; return self
        def __exit__(self, *args):
            clock[0] += 100_000_000
        def report(self):
            return dict(complete=True, events=[], elapsed_ns=0)
    def function():
        clock[0] += 2_000_000; return Result()
    def validate(result):
        clock[0] += 100_000_000
    monkeypatch.setattr(bench.time, "perf_counter_ns", lambda: clock[0])
    monkeypatch.setattr(bench.cx, "profile", lambda **kwargs: FakeProfile())
    result = bench.measure(function, validate, repeats=3, warmup=1, max_events=10)
    assert all(value["samples_ms"] == [2] * 3 for value in result["wall"].values())
    assert len(result["profiles"]) == 3


def test_runtime_profile_benchmark_parity_and_safe_report(device_name):
    result = bench.run(argparse.Namespace(device=device_name, sizes=(17,), repeats=1, warmup=1, max_events=10000,
                                         include_events=True))
    assert len(result["results"]) == 3
    assert str(ROOT) not in json.dumps(result)
    for row in result["results"]:
        assert row["cpu_parity"] and row["self_time_breakdown"]
        for report in row["profiles"]:
            check_accounting(report)


def test_runtime_profile_benchmark_compact_report():
    result = bench.run(argparse.Namespace(device="cpu", sizes=(17,), repeats=1, warmup=1, max_events=10000))
    assert not result["config"]["include_events"]
    for row in result["results"]:
        assert "profiles" not in row and len(row["profile_samples"]) == 1
        assert row["profile_samples"][0]["complete"] and row["profile_samples"][0]["event_count"] > 0


def test_runtime_profile_benchmark_refuses_missing_backend_or_truncation(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(bench.cx, "is_available", lambda name: False)
        with pytest.raises(RuntimeError, match="no fallback"):
            bench.run(argparse.Namespace(device="cuda"))
    with pytest.raises(RuntimeError, match="truncated"):
        bench.measure(lambda: cx.ones((17,)), lambda value: None, 1, 0, 1)


def test_runtime_profile_failed_benchmark_preserves_output(tmp_path, monkeypatch):
    target = tmp_path / "report.json"
    target.write_text("previous")
    monkeypatch.setattr(bench.sys, "argv", ["bench", "--device", "cpu", "--output", str(target)])
    monkeypatch.setattr(bench, "run", lambda args: (_ for _ in ()).throw(AssertionError("wrong output")))
    with pytest.raises(AssertionError, match="wrong output"):
        bench.main()
    assert target.read_text() == "previous"
