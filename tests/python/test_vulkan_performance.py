"""Vulkan clocks, owned-memory accounting and bounded synchronous reuse."""
from concurrent.futures import ThreadPoolExecutor
import gc
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

import tensorcx as cx
from test_vulkan import vulkan_device


def info(device):
    return cx.diagnostics()["backends"][device]


def bounded(stats):
    assert 0 <= stats["cached_bytes"] <= stats["max_cached_bytes"]
    assert 0 <= stats["cached_blocks"] <= stats["max_cached_blocks"]
    assert stats["in_use_bytes"] >= 0 and stats["peak_cached_bytes"] <= stats["max_cached_bytes"]
    assert 0 <= stats["cached_frames"] <= 1


@pytest.mark.parametrize("dtype", [cx.float32, cx.int32, cx.bool])
@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (1,), (3,), (257,), (2, 3, 5)])
def test_vulkan_pool_reuse_values_and_bounds(vulkan_device, dtype, shape):
    before = info(vulkan_device)["allocations"]["physical_blocks"]
    with cx.memory_pool(vulkan_device) as pool:
        for value in (1, 0, 1):
            tensor = (cx.ones if value else cx.zeros)(shape, dtype=dtype, device=vulkan_device)
            np.testing.assert_array_equal(tensor.numpy(), np.full(shape, value, dtype=dtype))
            del tensor
            bounded(pool.stats())
        stats = pool.stats()
        if np.prod(shape, dtype=int):
            assert stats["hits"] >= 2 and stats["frame_hits"] > 0 and stats["in_use_bytes"] == 0
        else:
            assert stats["allocation_calls"] == 0 and stats["frame_hits"] == stats["frame_misses"] == 0
    assert not pool.stats()["active"] and pool.stats()["cached_bytes"] == pool.stats()["cached_frames"] == 0
    assert info(vulkan_device)["allocations"]["physical_blocks"] == before


@pytest.mark.parametrize("shape", [(0, 2**62, 4), (2**62, 4, 0)])
def test_vulkan_invalid_zero_shape_never_enters_pool(vulkan_device, shape):
    with cx.memory_pool(vulkan_device) as pool:
        with cx.profile() as profile:
            with pytest.raises(ValueError, match="overflow"):
                cx.empty(shape, device=vulkan_device)
        assert not profile.report()["events"]
        assert pool.stats()["hits"] == pool.stats()["misses"] == pool.stats()["allocation_calls"] == 0


@pytest.mark.parametrize("limit,blocks", [(0, 2), (3, 2), (4, 1), (32, 2), (1024, 2)])
def test_vulkan_pool_limits_trim_and_escaped_views(vulkan_device, limit, blocks):
    with cx.memory_pool(vulkan_device, max_cached_bytes=limit, max_cached_blocks=blocks) as pool:
        values = [cx.ones((n,), device=vulkan_device) for n in (1, 3, 5, 7)]
        while values:
            values.pop()
            bounded(pool.stats())
        value = cx.ones((7,), device=vulkan_device)
        view = value.reshape((1, 7))
        del value
        assert pool.stats()["in_use_bytes"] >= view.nbytes
        pool.trim()
        assert pool.stats()["cached_bytes"] == pool.stats()["cached_frames"] == 0
        assert pool.stats()["in_use_bytes"] >= view.nbytes
    np.testing.assert_array_equal(view.numpy(), np.ones((1, 7), np.float32))
    with cx.memory_pool(vulkan_device) as other:
        del view
        assert other.stats()["hits"] == other.stats()["in_use_bytes"] == 0
    assert pool.stats()["in_use_bytes"] == 0


def test_vulkan_pool_frames_and_driver_allocation_reduction(vulkan_device):
    def work():
        x = cx.ones((257,), device=vulkan_device)
        return (x + 1).sum()
    base = info(vulkan_device)
    for _ in range(4):
        y = work(); del y
    direct = info(vulkan_device)
    with cx.memory_pool(vulkan_device) as pool:
        for _ in range(4):
            y = work(); del y
        assert pool.stats()["frame_misses"] == 1 and pool.stats()["frame_hits"] > 0
        assert pool.stats()["hits"] > pool.stats()["misses"]
    reused = info(vulkan_device)
    for group, key in (("allocations", "allocation_calls"), ("execution", "command_buffer_creations"),
                       ("execution", "fence_creations"), ("execution", "descriptor_pool_creations")):
        assert reused[group][key] - direct[group][key] < direct[group][key] - base[group][key]
    assert reused["allocations"]["physical_blocks"] == base["allocations"]["physical_blocks"]


def test_vulkan_owned_stats_and_heap_contract(vulkan_device):
    base = info(vulkan_device)
    with cx.memory_pool(vulkan_device) as pool:
        x = cx.ones((3, 5), device=vulkan_device)
        held = info(vulkan_device)
        assert held["allocations"]["live"]["tensor"]["requested_bytes"] == base["allocations"]["live"]["tensor"]["requested_bytes"] + 60
        assert held["allocations"]["live"]["internal"]["requested_bytes"] == 0
        assert held["allocations"]["live"]["staging"]["requested_bytes"] == 0
        del x
        cached = info(vulkan_device)
        assert cached["allocations"]["cached_blocks"] == pool.stats()["cached_blocks"]
        assert cached["allocations"]["cached_bytes"] == pool.stats()["cached_bytes"]
        assert cached["allocations"]["scope"] == "tensorcx_owned_vulkan_allocations"
        assert cx.memory_info(vulkan_device) == base["memory"]
    assert info(vulkan_device)["allocations"]["physical_blocks"] == base["allocations"]["physical_blocks"]


def test_vulkan_profile_device_clocks_and_export(vulkan_device, tmp_path):
    x = cx.tensor(np.arange(17, dtype=np.float32), device=vulkan_device)
    with cx.profile(device_timing=True) as profile:
        with cx.memory_pool(vulkan_device):
            y = ((x + 1) * .5).sum()
            result = y.cpu()
    np.testing.assert_allclose(result.numpy(), ((np.arange(17, dtype=np.float32) + 1) * .5).sum())
    r = profile.report()
    assert r["clock"] == "steady_clock_host" and r["device_clock"] == "vulkan_timestamps_elapsed_ms"
    assert r["device_clocks"]["vulkan"] == r["device_clock"]
    assert {e["backend"] for e in r["device_events"]} == {"vulkan"}
    assert {e["name"] for e in r["device_events"]} >= {"vulkan_compute", "vulkan_upload", "vulkan_download"}
    by_id = {e["id"]: e for e in r["events"]}
    for event in r["device_events"]:
        assert event["host_event_id"] in by_id and by_id[event["host_event_id"]]["backend"] == "vulkan"
        if info(vulkan_device)["timestamps"]["supported"]:
            assert event["error"] is None and np.isfinite(event["elapsed_ms"]) and event["elapsed_ms"] >= 0
        else:
            assert event["error"] is not None and event["elapsed_ms"] is None
    assert sum(row["self_ns"] for row in profile.summary()) == sum(e["self_ns"] for e in r["events"])
    assert sum(row["calls"] for row in profile.device_summary()) == len(r["device_events"])
    profile.export_chrome_trace(tmp_path / "trace.json")
    trace = json.loads((tmp_path / "trace.json").read_text())
    assert trace["metadata"]["device_clock"] == r["device_clock"]
    assert len(trace["traceEvents"]) == len(r["events"])
    assert {e["name"] for e in r["events"]} >= {"vkAllocateMemory", "vkFreeMemory", "vkQueueSubmit", "vkWaitForFences"}
    assert info(vulkan_device)["allocations"]["live"]["internal"]["requested_bytes"] == 0


def test_vulkan_profile_disabled_zero_work_and_bounded_records(vulkan_device):
    before = info(vulkan_device)["execution"]["query_pool_creations"]
    with cx.profile() as plain:
        y = cx.ones((17,), device=vulkan_device) + 1
    assert plain.report()["device_events"] == []
    assert info(vulkan_device)["execution"]["query_pool_creations"] == before
    with cx.profile(device_timing=True) as empty:
        z = cx.empty((2**40, 0), device=vulkan_device).softmax(-1)
    assert not z.nbytes and not empty.report()["device_events"]
    assert info(vulkan_device)["execution"]["query_pool_creations"] == before
    with cx.profile(max_events=2, device_timing=True) as bounded:
        for _ in range(5): y = y + 1
    r = bounded.report()
    assert len(r["events"]) <= 2 and len(r["device_events"]) <= 2
    assert r["dropped_events"] > 0 and r["dropped_device_events"] > 0 and not r["complete"]


def test_vulkan_profile_mixed_cuda_clocks(vulkan_device):
    if not cx.is_available("cuda"): pytest.skip("CUDA unavailable for mixed-clock acceptance")
    a, b = cx.ones((17,), device=vulkan_device), cx.ones((17,), device="cuda")
    with cx.profile(device_timing=True) as profile:
        x, y = a + 1, b + 1
    r = profile.report()
    assert {e["backend"] for e in r["device_events"]} == {"vulkan", "cuda"}
    assert r["device_clock"] == "backend_device_intervals"
    assert r["device_clocks"] == {"cuda": "cuda_events_elapsed_ms", "vulkan": "vulkan_timestamps_elapsed_ms"}
    np.testing.assert_array_equal(x.numpy(), y.numpy())


def test_vulkan_pool_scope_errors_and_recovery(vulkan_device):
    pool = cx.memory_pool(vulkan_device)
    with pytest.raises(RuntimeError): pool.stats()
    with pytest.raises(RuntimeError): pool.trim()
    with pool:
        with pytest.raises(ValueError, match="nested"):
            with cx.memory_pool(vulkan_device): pass
        with ThreadPoolExecutor(max_workers=1) as workers:
            with pytest.raises(RuntimeError, match="owning thread"):
                workers.submit(pool.trim).result()
        with pytest.raises(ValueError): cx.embedding(cx.tensor([-1], dtype=cx.int32, device=vulkan_device), cx.ones((2, 0), device=vulkan_device))
        assert cx.ones((3,), device=vulkan_device).sum().numpy() == 3
    with pytest.raises(RuntimeError):
        with pool: pass
    def escaped_scope():
        with cx.memory_pool(vulkan_device) as p:
            return p, cx.ones((257,), device=vulkan_device)
    p, value = escaped_scope()
    assert not p.stats()["active"] and p.stats()["in_use_bytes"] >= value.nbytes
    pending = [value]
    del value
    def release_escaped():
        value = pending.pop()
        del value
    with ThreadPoolExecutor(max_workers=1) as workers:
        workers.submit(release_escaped).result()
    gc.collect()
    assert p.stats()["in_use_bytes"] == 0


def test_vulkan_pool_last_release_on_foreign_thread(vulkan_device):
    with cx.memory_pool(vulkan_device) as origin:
        pending = [cx.ones((257,), device=vulkan_device)]
        assert origin.stats()["in_use_bytes"] >= 257 * 4
        def release():
            with cx.memory_pool(vulkan_device) as foreign:
                value = pending.pop()
                del value
                gc.collect()
                assert foreign.stats()["cached_blocks"] == foreign.stats()["in_use_bytes"] == 0
                assert foreign.stats()["hits"] == foreign.stats()["misses"] == 0
            return foreign.stats()
        with ThreadPoolExecutor(max_workers=1) as workers:
            workers.submit(release).result()
        assert origin.stats()["in_use_bytes"] == 0 and origin.stats()["cached_blocks"] > 0
        bounded(origin.stats())


def test_vulkan_pool_exception_cleanup_and_cached_bits(vulkan_device):
    bits = np.array([0, 0x80000000, 1, 0x007fffff, 0x7fc12345], np.uint32)
    retained = []
    with pytest.raises(LookupError, match="scope sentinel"):
        with cx.memory_pool(vulkan_device) as pool:
            for repeat in range(4):
                host = bits if repeat % 2 else bits[::-1].copy()
                x = cx.tensor(host.view(np.float32), device=vulkan_device)
                y = x[::-1]
                np.testing.assert_array_equal(y.numpy().view(np.uint32), host[::-1])
                # Packed bool reuses padded capacities from previous dtype
                # leases; every byte must be freshly written.
                mask = cx.tensor([True, False, bool(repeat % 2), True, False], device=vulkan_device)
                actual = cx.concat([mask[::-1], mask], axis=0)
                np.testing.assert_array_equal(actual.numpy(), np.r_[mask.numpy()[::-1], mask.numpy()])
                if repeat == 3: retained.append(y)
                del x, y, mask, actual
            assert pool.stats()["hits"] > 0
            raise LookupError("scope sentinel")
    assert not pool.stats()["active"] and pool.stats()["cached_bytes"] == pool.stats()["cached_frames"] == 0
    np.testing.assert_array_equal(retained[0].numpy().view(np.uint32), bits[::-1])
    retained.clear(); gc.collect()
    assert pool.stats()["in_use_bytes"] == 0


def test_vulkan_pool_threads_independent_and_inputs_immutable(vulkan_device):
    data = np.arange(35, dtype=np.float32).reshape(5, 7)
    shared = cx.tensor(data, device=vulkan_device)
    def forward(i):
        with cx.memory_pool(vulkan_device, max_cached_bytes=8192, max_cached_blocks=16) as pool:
            for _ in range(3):
                result = (shared + i).layernorm(-1).gelu().softmax(-1)
            exported = result.numpy(); exported[...] = 123
            actual = result.numpy()
            assert not np.all(actual == 123)
            bounded(pool.stats())
            assert pool.stats()["frame_hits"] > 0
        return actual, pool.stats()
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(forward, range(8)))
    for i, (actual, stats) in enumerate(results):
        expected = (cx.tensor(data) + i).layernorm(-1).gelu().softmax(-1).numpy()
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
        assert not stats["active"] and stats["cached_frames"] == 0
    np.testing.assert_array_equal(shared.numpy(), data)


def test_vulkan_paired_benchmark_accepts_device_alias(vulkan_device):
    benchmark = runpy.run_path(str(Path(__file__).resolve().parents[2] / "benchmarks/vulkan_runtime.py"))
    alias = "vulkan:0" if vulkan_device == "vulkan" else vulkan_device
    report = benchmark["run"](device=alias, elements=5, iterations=1, rounds=2, warmup=0)
    assert report["device"] == vulkan_device and report["cpu_parity"]
    assert len(report["cases"]) == 6
    for case in report["cases"].values():
        for mode in ("direct", "pooled"):
            assert [sample["order"] for sample in case["samples"][mode]] == ([0, 1] if mode == "direct" else [1, 0])
            assert case["summary"][mode]["median_ns"] > 0
            assert case["profiles"][mode]["host_complete"]
