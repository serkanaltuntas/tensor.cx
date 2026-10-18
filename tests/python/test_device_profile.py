"""Device event acceptance: separate clocks, parity, lifetime and bounded reports."""
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import math
import os
import threading

import numpy as np
import pytest

import tensorcx as cx
from tensorcx import _core
from tensorcx._compiler.cpu import toolchain


cuda_only = pytest.mark.skipif(not cx.is_available("cuda"), reason="CUDA is unavailable")


def check_report(profile, *, names=()):
    r = profile.report()
    assert r["device_timing"] and r["device_clock"] == "cuda_events_elapsed_ms"
    assert r["complete"] and r["host_complete"] and r["device_complete"]
    assert r["device_errors"] == r["dropped_device_events"] == 0
    assert sum(e["self_ns"] for e in r["events"]) + r["unattributed_ns"] == r["elapsed_ns"]
    host = {e["id"]: e for e in r["events"]}
    assert [e["id"] for e in r["device_events"]] == list(range(1, len(r["device_events"]) + 1))
    for e in r["device_events"]:
        assert e["backend"] == "cuda" and e["error"] is None
        assert math.isfinite(e["elapsed_ms"]) and e["elapsed_ms"] >= 0
        assert host[e["host_event_id"]]["category"] == "compute_sync"
        assert host[e["host_event_id"]]["name"] == e["name"]
        assert "start_ns" not in e and "duration_ns" not in e and "self_ns" not in e
    assert set(names) <= {e["name"] for e in r["device_events"]}
    assert sum(row["self_ns"] for row in profile.summary()) == sum(e["self_ns"] for e in r["events"])
    assert sum(row["calls"] for row in profile.device_summary()) == len(r["device_events"])
    assert all(row["calls"] == row["measured_calls"] and row["failed_calls"] == 0
               for row in profile.device_summary())
    assert sum(row["elapsed_ms"] for row in profile.device_summary()) == pytest.approx(
        sum(e["elapsed_ms"] for e in r["device_events"]))
    return r


@pytest.mark.parametrize("bad", [None, 0, 1, "true", [], np.bool_(True)])
def test_device_profile_option_is_strict_bool(bad):
    with pytest.raises(ValueError, match="device_timing"):
        cx.profile(device_timing=bad)
    with pytest.raises((ValueError, TypeError), match="device_timing"):
        _core._profile_begin(10, bad)


def test_device_profile_cpu_and_disabled_are_explicit(tmp_path):
    with cx.profile(device_timing=True) as p:
        result = cx.ones((17,)) + cx.ones((17,))
    assert result.numpy().tolist() == [2] * 17
    assert check_report(p)["device_events"] == []
    assert p.report()["device_coverage"]["cpu"] == "not instrumented"
    assert p.report()["device_coverage"]["metal"] == "not instrumented"
    with cx.profile() as off:
        cx.ones((17,))
    assert not off.report()["device_timing"] and off.report()["device_complete"] is None
    assert off.device_summary() == [] and off.report()["device_events"] == []
    p.export_chrome_trace(tmp_path / "trace.json")
    trace = json.loads((tmp_path / "trace.json").read_text())
    assert len(trace["traceEvents"]) == len(p.report()["events"])
    assert trace["metadata"]["device_events"] == []


@cuda_only
@pytest.mark.parametrize("dtype", [cx.float32, cx.int32])
@pytest.mark.parametrize("shape", [(), (1,), (257,), (2, 3, 5)])
def test_device_profile_cuda_parity(dtype, shape):
    values = np.resize(np.array([-17, 0, 19, 16777217], dtype=dtype), shape)
    x = cx.tensor(values, device="cuda")
    with cx.profile(device_timing=True) as p:
        y = ((x + x) * x).sum()
    cpu = cx.tensor(values)
    expected = ((cpu + cpu) * cpu).sum().numpy()
    if dtype == cx.int32:
        np.testing.assert_array_equal(y.numpy(), expected)
    else:
        np.testing.assert_allclose(y.numpy(), expected, rtol=1e-5)
    np.testing.assert_array_equal(x.numpy(), values)
    r = check_report(p, names=("launch_binary",))
    assert {e["name"] for e in r["device_events"]} & {"launch_axis", "launch_reduce_axes"}
    assert {e["operation"] for e in r["device_events"]} == {"add", "multiply", "sum"}
    assert len(r["device_events"]) == 3


@cuda_only
def test_device_profile_cuda_all_launch_families_and_waits():
    x = cx.tensor(np.arange(12, dtype=np.float32).reshape(3, 4), device="cuda")
    indices = cx.tensor([0, 2], dtype=cx.int32, device="cuda")
    def work():
        a = cx.ones((3, 4), device="cuda")
        binary = x + a
        b = (x + cx.ones((1, 4), device="cuda")) * 2
        c = cx.transpose(b)
        d = cx.concat([x, a])
        cast = x.astype(cx.int32).astype(cx.float32)
        neg = -x
        small = x.sum(axis=(0, 1))
        mm = x @ c
        mask = x > 4
        truth = mask.all()
        boolean = mask.astype(cx.float32).astype(cx.bool)
        selected = x[mask]
        root = cx.sqrt(x)
        embed = cx.embedding(indices, x)
        attention = cx.scaled_dot_product_attention(x, x, x)
        softmax = cx.softmax(x, axis=-1)
        return a, binary, b, c, d, cast, neg, small, mm, mask, truth, boolean, selected, root, embed, attention, softmax
    with cx.profile() as off:
        expected = work()
    with cx.profile(device_timing=True) as on:
        actual = work()
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-6, atol=1e-6)
    names = ("launch_fill", "launch_binary", "launch_broadcast_binary", "launch_scalar", "launch_transpose",
             "launch_concat", "launch_validate_int32_cast", "launch_cast", "launch_unary", "launch_reduce_axes",
             "launch_matmul", "launch_axis", "launch_predicate", "launch_bool_reduce", "launch_bool_cast",
             "launch_mask_prefix", "launch_prefix_step", "launch_mask_copy", "launch_math",
             "launch_embedding_validate", "launch_embedding_gather", "launch_attention_softmax")
    r = check_report(on, names=names)
    assert off.report()["device_events"] == []
    waits = lambda report: [e["name"] for e in report["events"] if e["category"] == "synchronize"]
    assert waits(r) == waits(off.report())


@cuda_only
def test_device_profile_cuda_empty_copies_and_bounds():
    empty = cx.ones((0,), device="cuda")
    source = cx.ones((17,), device="cuda")
    with cx.profile(device_timing=True) as p:
        blank = empty + empty
        copied = source.astype(cx.float32).cpu()
        boolean = cx.ones((17,), dtype=cx.bool, device="cuda")
    assert blank.numpy().size == 0 and copied.numpy().tolist() == [1] * 17 and boolean.numpy().all()
    assert check_report(p)["device_events"] == []
    with cx.profile(max_events=1, device_timing=True) as clipped:
        a = source + source
        b = a * a
    assert b.numpy().tolist() == [4] * 17
    r = clipped.report()
    assert not r["complete"] and not r["host_complete"] and not r["device_complete"]
    assert r["dropped_events"] > 0 and r["dropped_device_events"] == 1
    assert len(r["events"]) == len(r["device_events"]) == 1
    assert r["device_events"][0]["host_event_id"] == 0 and r["device_events"][0]["error"] is None
    assert r["device_events"][0]["operation"] == "add"


@cuda_only
def test_device_profile_cuda_snapshot_trace_and_exception(tmp_path):
    x = cx.ones((257,), device="cuda")
    with pytest.raises(ValueError, match="body failed"):
        with cx.profile(device_timing=True) as p:
            y = x + x
            raise ValueError("body failed")
    assert y.numpy().tolist() == [2] * 257
    r = check_report(p)
    p.report()["device_events"].clear()
    assert p.report() == r
    p.export_chrome_trace(tmp_path / "cuda-trace.json")
    text = (tmp_path / "cuda-trace.json").read_text()
    trace = json.loads(text)
    assert trace["metadata"]["device_events"] == r["device_events"]
    assert len(trace["traceEvents"]) == len(r["events"])
    assert all(e["pid"] == e["tid"] == 0 for e in trace["traceEvents"])
    assert "GPU-" not in text and str(tmp_path) not in text
    with cx.profile(device_timing=True) as following:
        x + x
    check_report(following)


@cuda_only
@pytest.mark.parametrize("failure", ["event_error", "device_cap"])
def test_device_profile_cuda_partial_measurement_is_explicit(monkeypatch, failure):
    original = _core._profile_end
    def end(token):
        r = original(token)
        if failure == "event_error":
            r["device_events"][0].update(elapsed_ms=None, error="elapsed_failed")
        else:
            r["dropped_device_events"] = 1
        return r
    monkeypatch.setattr(_core, "_profile_end", end)
    x = cx.ones((17,), device="cuda")
    with cx.profile(device_timing=True) as p:
        a = x + x
        b = x + x
    assert a.numpy().tolist() == b.numpy().tolist() == [2] * 17
    r = p.report()
    assert r["host_complete"] and not r["complete"] and not r["device_complete"]
    assert r["device_errors"] == (failure == "event_error")
    row, = p.device_summary()
    assert row["calls"] == 2 and row["failed_calls"] == (failure == "event_error")
    assert row["measured_calls"] + row["failed_calls"] == 2


@cuda_only
def test_device_profile_cuda_threads_and_abandonment():
    x = cx.ones((257,), device="cuda")
    barrier = threading.Barrier(2)
    def work(count):
        with cx.profile(device_timing=True) as p:
            barrier.wait(timeout=10)
            for _ in range(count):
                y = x + x
        assert y.numpy().tolist() == [2] * 257
        assert len(check_report(p)["device_events"]) == count
    with cx.profile(device_timing=True) as p, ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(work, n) for n in (2, 3)]
        for f in futures:
            f.result(timeout=30)
    assert check_report(p)["device_events"] == []
    abandoned = cx.profile(device_timing=True)
    abandoned.__enter__()
    x + x
    del abandoned
    gc.collect()
    with cx.profile(device_timing=True) as fresh:
        x + x
    assert len(check_report(fresh)["device_events"]) == 1


@cx.experimental.kernel
def generated(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = cx.experimental.silu(a[i] * b[i])


@cuda_only
def test_device_profile_cuda_generated_and_zero_work():
    try:
        toolchain()
        _core._cuda_kernel_support()
    except RuntimeError:
        if os.environ.get("TENSORCX_REQUIRE_MLIR_CUDA"):
            raise
        pytest.skip("optional MLIR CUDA target unavailable")
    compiled = generated.compile(target="cuda", compiler="mlir")
    x = cx.ones((257,), device="cuda")
    out = cx.zeros((257,), device="cuda")
    with cx.profile(device_timing=True) as p:
        y = compiled.launch(x, x, out, 257, block_size=7)
    cx.testing.assert_allclose(y, cx.silu(x), rtol=1e-5, atol=1e-6)
    assert out.numpy().tolist() == [0] * 257
    r = check_report(p, names=("generated_kernel",))
    assert len(r["device_events"]) == 1
    with cx.profile(device_timing=True) as zero:
        y = compiled.launch(x, x, out, 0, thread_count=0, block_size=7)
    assert y.numpy().tolist() == [0] * 257
    assert check_report(zero)["device_events"] == []
