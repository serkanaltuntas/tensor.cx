"""Bounded reuse, numerical parity, isolation, exports and exception cleanup."""
from concurrent.futures import ThreadPoolExecutor
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import tensorcx as cx
from tensorcx import _core

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cuda():
    if not cx.is_available("cuda"):
        if os.environ.get("TENSORCX_REQUIRE_CUDA"):
            pytest.fail("CUDA required for memory pool acceptance")
        pytest.skip("CUDA is not available")


def bounded(stats):
    assert 0 <= stats["cached_bytes"] <= stats["max_cached_bytes"]
    assert 0 <= stats["cached_blocks"] <= stats["max_cached_blocks"]
    assert stats["in_use_bytes"] >= 0
    assert stats["peak_cached_bytes"] <= stats["max_cached_bytes"]


@pytest.mark.parametrize("dtype", [cx.float32, cx.int32, cx.bool])
@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (1,), (257,), (2, 3, 5)])
def test_pool_fill_reuse_and_parity(cuda, dtype, shape):
    with cx.memory_pool() as pool:
        for value in (1, 0, 1):
            tensor = cx.ones(shape, dtype=dtype, device="cuda") if value else cx.zeros(shape, dtype=dtype, device="cuda")
            np.testing.assert_array_equal(tensor.numpy(), np.full(shape, value, dtype=dtype))
            del tensor
            bounded(pool.stats())
        stats = pool.stats()
        assert stats["hits"] >= 2 if np.prod(shape, dtype=int) else stats["allocation_calls"] == 0
        assert stats["in_use_bytes"] == 0
    assert not pool.stats()["active"] and pool.stats()["cached_bytes"] == 0


def test_pool_profiles_real_allocations_and_default_path(cuda):
    with cx.profile() as normal:
        for _ in range(3):
            value = cx.ones((257,), device="cuda")
            del value
    assert sum(e["name"] == "cudaMalloc" for e in normal.report()["events"]) == 3
    assert sum(e["name"] == "cudaFree" for e in normal.report()["events"]) == 3
    assert not any(e["category"] == "memory_pool" for e in normal.report()["events"])
    with cx.profile() as profiled:
        with cx.memory_pool() as pool:
            for _ in range(3):
                value = cx.ones((257,), device="cuda")
                del value
    assert pool.stats()["hits"] == 2 and pool.stats()["allocation_calls"] == 1
    assert sum(e["name"] == "cudaMalloc" for e in profiled.report()["events"]) == 1
    assert sum(e["name"] == "cudaFree" for e in profiled.report()["events"]) == 1
    assert any(e["category"] == "memory_pool" for e in profiled.report()["events"])
    with cx.profile() as after:
        value = cx.ones((257,), device="cuda")
        del value
    assert sum(e["name"] == "cudaMalloc" for e in after.report()["events"]) == 1


@pytest.mark.parametrize("limit,blocks", [(0, 3), (3, 3), (4, 1), (12, 2), (1024, 2)])
def test_pool_caps_trim_and_live_views(cuda, limit, blocks):
    with cx.memory_pool(max_cached_bytes=limit, max_cached_blocks=blocks) as pool:
        tensors = [cx.ones((n,), device="cuda") for n in (1, 2, 3, 4)]
        while tensors:
            tensors.pop()
            bounded(pool.stats())
        value = cx.ones((3,), device="cuda")
        view = value.reshape((1, 3))
        del value
        assert pool.stats()["in_use_bytes"] == 12
        pool.trim()
        assert pool.stats()["cached_bytes"] == 0
        assert pool.stats()["in_use_bytes"] == 12
    np.testing.assert_array_equal(view.numpy(), [[1, 1, 1]])
    del view
    assert pool.stats()["in_use_bytes"] == pool.stats()["cached_bytes"] == 0


def test_pool_exact_size_cross_dtype_and_simultaneous_outputs(cuda):
    with cx.memory_pool() as pool:
        first = cx.ones((257,), dtype=cx.float32, device="cuda")
        del first
        second = cx.tensor(np.full(257, 16777217, dtype=np.int32), device="cuda")
        assert pool.stats()["hits"] == 1
        np.testing.assert_array_equal(second.numpy(), np.full(257, 16777217, np.int32))
        inputs = [cx.tensor([1., 2., 3.], device="cuda") for _ in range(2)]
        a, b = inputs[0] + inputs[1], inputs[0] * inputs[1]
        np.testing.assert_array_equal(a.numpy(), [2, 4, 6])
        np.testing.assert_array_equal(b.numpy(), [1, 4, 9])
        assert pool.stats()["in_use_bytes"] >= 257 * 4 + 4 * 3 * 4
    # Live tensors are not invalidated or adopted into a later pool.
    with cx.memory_pool() as later:
        del second, inputs, a, b
        assert later.stats()["in_use_bytes"] == later.stats()["cached_bytes"] == 0
    assert pool.stats()["in_use_bytes"] == 0


@pytest.mark.parametrize("copy", [False, True])
@pytest.mark.parametrize("version", [None, (1, 0)])
def test_pool_dlpack_export_and_borrow_never_recycle(cuda, copy, version):
    with cx.memory_pool() as pool:
        source = cx.ones((257,), device="cuda")
        capsule = source.__dlpack__(copy=copy, max_version=version)
        imported = cx.from_dlpack(capsule)
        del source, capsule
        np.testing.assert_array_equal(imported.numpy(), np.ones(257))
        del imported
        gc.collect()
        # A copy export protects the cloned allocation, while source may cache.
        stats = pool.stats()
        assert stats["free_calls"] >= 1 and stats["in_use_bytes"] == 0
        assert stats["cached_blocks"] == (1 if copy else 0)
        pool.trim()
    assert pool.stats()["cached_bytes"] == 0


def test_pool_discarded_export_is_permanently_excluded(cuda):
    with cx.memory_pool() as pool:
        value = cx.ones((257,), device="cuda")
        capsule = value.reshape((1, 257)).__dlpack__()
        del capsule
        del value
        gc.collect()
        assert pool.stats()["cached_blocks"] == 0 and pool.stats()["free_calls"] == 1


def test_pool_composite_math_masks_and_errors(cuda):
    a = np.arange(24, dtype=np.float32).reshape(4, 6) / 10
    expected = cx.softmax(cx.layernorm(cx.tensor(a) @ cx.ones((6, 3)), axis=-1), axis=-1).numpy()
    with cx.memory_pool() as pool:
        x = cx.tensor(a, device="cuda")
        for _ in range(3):
            result = cx.softmax(cx.layernorm(x @ cx.ones((6, 3), device="cuda"), axis=-1), axis=-1)
            np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-5)
            selected = x[x > 1]
            np.testing.assert_array_equal(selected.numpy(), a[a > 1])
            values, indices = cx.topk(x, 2, axis=-1)
            np.testing.assert_array_equal(values.numpy(), np.sort(a, axis=-1)[:, -2:][:, ::-1])
            assert indices.dtype == "int32"
            with pytest.raises(ValueError):
                cx.embedding(cx.tensor([99], dtype=cx.int32, device="cuda"), x)
            del result, selected, values, indices
        assert pool.stats()["hits"] > 0
        del x
    assert pool.stats()["in_use_bytes"] == pool.stats()["cached_bytes"] == 0


@pytest.mark.parametrize("field,bad", [("max_cached_bytes", v) for v in (-1, True, np.bool_(True), 1.5, "12", 2**100)] +
                         [("max_cached_blocks", v) for v in (0, -1, 4097, False, 1.5)])
def test_pool_limit_validation(field, bad):
    with pytest.raises(ValueError, match=field):
        cx.memory_pool(**{field: bad})


def test_pool_backend_routing_and_no_cpu_fallback():
    with pytest.raises(ValueError, match="not supported"):
        with cx.memory_pool("cpu"):
            pytest.fail("CPU pool silently accepted")
    with pytest.raises(ValueError, match="index 0"):
        cx.memory_pool("cuda:1")
    with pytest.raises(ValueError, match="not available"):
        with cx.memory_pool("missing"):
            pytest.fail("unknown pool backend accepted")


def test_pool_nested_exception_cleanup_and_single_use(cuda):
    before = cx.memory_pool()
    with pytest.raises(RuntimeError, match="after entry"):
        before.stats()
    with pytest.raises(RuntimeError, match="active owning"):
        before.trim()
    with pytest.raises(KeyError):
        with before:
            with pytest.raises(ValueError, match="nested"):
                with cx.memory_pool():
                    pytest.fail("nested entry succeeded")
            value = cx.ones((3,), device="cuda")
            del value
            raise KeyError("body failure")
    assert not before.stats()["active"] and before.stats()["cached_bytes"] == 0
    with pytest.raises(RuntimeError, match="single-use"):
        with before:
            pass
    with cx.memory_pool():
        pass
    abandoned = cx.memory_pool()
    abandoned.__enter__()
    del abandoned
    gc.collect()
    with cx.memory_pool():
        pass


def test_pool_threads_isolated_and_cross_thread_final_release(cuda):
    with cx.memory_pool() as pool:
        value = cx.ones((257,), device="cuda")
        slot = [value]
        del value
        def worker():
            with pytest.raises(RuntimeError, match="owning thread"):
                pool.__exit__(None, None, None)
            with cx.memory_pool() as own:
                slot.pop()
                assert own.stats()["cached_bytes"] == own.stats()["in_use_bytes"] == 0
                for _ in range(3):
                    tensor = cx.ones((13,), device="cuda")
                    del tensor
            return own.stats()
        with ThreadPoolExecutor(max_workers=1) as executor:
            separate = executor.submit(worker).result(timeout=30)
        assert separate["hits"] == 2
        assert pool.stats()["cached_bytes"] == 257 * 4 and pool.stats()["in_use_bytes"] == 0
    assert pool.stats()["cached_bytes"] == 0


@cx.experimental.kernel
def pooled_generated(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = (a[i] + b[i]) * b[i]


def test_pool_generated_private_output_and_suffix(cuda):
    from tensorcx._compiler.cpu import toolchain
    try:
        toolchain()
        _core._cuda_kernel_support()
    except RuntimeError:
        if os.environ.get("TENSORCX_REQUIRE_MLIR"):
            raise
        pytest.skip("optional native LLVM target unavailable")
    compiled = pooled_generated.compile(target="cuda", compiler="mlir", cache=False)
    x = cx.ones((257,), device="cuda")
    template = cx.tensor(np.full(257, -17, np.float32), device="cuda")
    with cx.memory_pool() as pool:
        for _ in range(3):
            result = compiled.launch(x, x, template, 255, thread_count=255, block_size=7)
            np.testing.assert_array_equal(result.numpy(), [2] * 255 + [-17] * 2)
            np.testing.assert_array_equal(template.numpy(), [-17] * 257)
            del result
        assert pool.stats()["hits"] >= 2


def benchmark():
    sys.path.insert(0, str(ROOT / "benchmarks"))
    try:
        spec = importlib.util.spec_from_file_location("bench_memory_pool", ROOT / "benchmarks/bench_memory_pool.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_pool_benchmark_safe_parity_and_cleanup(cuda):
    import argparse
    bench = benchmark()
    args = argparse.Namespace(pool="on", sizes=(17,), repeats=1, warmup=1,
                              max_cached_bytes=1024 * 1024, max_cached_blocks=64)
    report = bench.run(args)
    assert len(report["results"]) == 4
    assert str(ROOT) not in json.dumps(report)
    for row in report["results"]:
        assert len(row["samples_ms"]) == 1
        assert row["pool_before_close"]["hits"] > 0
        assert row["pool_after_close"]["in_use_bytes"] == row["pool_after_close"]["cached_bytes"] == 0


def test_pool_benchmark_failure_preserves_output(tmp_path, monkeypatch):
    bench = benchmark()
    target = tmp_path / "previous.json"
    target.write_text("previous")
    monkeypatch.setattr(sys, "argv", ["bench", "--pool", "on", "--output", str(target)])
    monkeypatch.setattr(bench, "run", lambda args: (_ for _ in ()).throw(AssertionError("parity failed")))
    with pytest.raises(AssertionError, match="parity failed"):
        bench.main()
    assert target.read_text() == "previous"


def test_pool_benchmark_setup_validation_cleanup_are_untimed(monkeypatch):
    import argparse
    bench = benchmark()
    clock = [0]
    class Result:
        def __del__(self): clock[0] += 100_000_000
    class FakePool:
        def __enter__(self): clock[0] += 100_000_000; return self
        def __exit__(self, *args): clock[0] += 100_000_000
        def stats(self): return {}
    class FakeProfile(FakePool):
        def report(self): return {"complete": True}
        def summary(self): return []
    def function(): clock[0] += 2_000_000; return Result()
    def validate(result): clock[0] += 100_000_000
    monkeypatch.setattr(bench.cx, "memory_pool", lambda *a, **kw: FakePool())
    monkeypatch.setattr(bench.cx, "profile", lambda *a, **kw: FakeProfile())
    monkeypatch.setattr(bench.time, "perf_counter_ns", lambda: clock[0])
    args = argparse.Namespace(pool="on", max_cached_bytes=100, max_cached_blocks=2, repeats=3, warmup=1)
    result = bench.measure(function, validate, args)
    assert result["samples_ms"] == [2, 2, 2]
