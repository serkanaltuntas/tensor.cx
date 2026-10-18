"""Generated CUDA private outputs: full-write contract and conservative fallback."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import tensorcx as cx
from tensorcx import _core
from tensorcx._compiler import cache, cuda
from tensorcx.experimental import IRBinaryOp, IRConstant, IRIf, IRLoad, IRName, KernelCompileError
from test_mlir_cuda_runtime import add, require_device


@cx.experimental.kernel
def expression(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = (a[i] + b[i]) * b[i]


@cx.experimental.kernel
def reordered(n, out, b, a):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = (a[i] + b[i]) * b[i]


@pytest.fixture(scope="module", params=[expression, reordered], ids=["expression", "reordered"])
def artifacts(request):
    require_device()
    dsl = request.param
    return dsl, dsl.compile(target="cpu", compiler="mlir", cache=False), dsl.compile(target="cuda", compiler="mlir", cache=False)


def launch(dsl, artifact, a, b, out, n):
    args = (n, out, b, a) if dsl.name == "reordered" else (a, b, out, n)
    return artifact.launch(*args, thread_count=n, block_size=7)


def profiled(call, *, copies, kernels):
    with cx.profile(device_timing=True) as p:
        output = call()
    report = p.report()
    assert report["complete"] and report["device_complete"]
    assert len(report["device_events"]) == kernels
    copies_seen = [e for e in report["events"] if e["backend"] == "cuda" and e["category"] == "copy_api"]
    waits = [e for e in report["events"] if e["backend"] == "cuda" and e["category"] == "synchronize"]
    assert len(copies_seen) == copies
    assert len(waits) == copies + kernels
    return output


@pytest.mark.parametrize("operation", ["add", "sub", "mul", "expr"])
def test_output_contract_manifest_version(operation):
    manifest = cuda.manifest("tttu", 2, 3, operation)
    assert manifest == (f"tensorcx.cuda.v2|linux-x86_64|llvm-21.1.8|sm52|ptx78|{operation}-f32-v1|"
                        "tttu|2|3|write-per-thread-no-output-read-v1")


@pytest.mark.parametrize("change", ["output_read", "load_offset", "store_offset", "conditional_store"])
def test_output_contract_invalid_ir_before_tools(monkeypatch, change):
    ir = add.parse_ir()
    guard = ir.body[1]
    store = guard.body[0]
    if change == "output_read":
        store = replace(store, value=replace(store.value, lhs=IRLoad("out", store.index)))
    elif change == "load_offset":
        store = replace(store, value=replace(store.value, lhs=IRLoad("a", IRBinaryOp("add", store.index, IRConstant(1)))))
    elif change == "store_offset":
        store = replace(store, index=IRBinaryOp("add", IRName("i"), IRConstant(1)))
    else:
        store = IRIf(guard.condition, (store,))
    bad = replace(ir, body=(ir.body[0], replace(guard, body=(store,))))
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid full-write IR reached toolchain/device")
    monkeypatch.setattr(cuda, "toolchain", forbidden)
    monkeypatch.setattr(_core, "_cuda_kernel_support", forbidden, raising=False)
    with pytest.raises(KernelCompileError):
        cuda.compile_kernel(bad, cache=False)


def test_output_contract_cache_identity_separates_legacy():
    common = dict(target="cuda", source="unchanged MLIR", tools={}, options={})
    old = cache.descriptor(**common, abi="tensorcx.cuda.v1|linux-x86_64|llvm-21.1.8|sm52|ptx78|add-f32-v1|tttu|2|3")
    new = cache.descriptor(**common, abi=cuda.manifest("tttu", 2, 3))
    assert cache.digest(json.dumps(old, sort_keys=True).encode()) != cache.digest(json.dumps(new, sort_keys=True).encode())


@pytest.mark.parametrize("n", [0, 1, 173, 257])
@pytest.mark.parametrize("alias", ["none", "left", "right", "both", "borrowed_left"])
def test_cuda_output_copy_alias_and_prefix(artifacts, n, alias):
    dsl, cpu, gpu = artifacts
    x = np.linspace(-2, 2, 257, dtype=np.float32)
    y = np.linspace(.25, .75, 257, dtype=np.float32)
    a = cx.tensor(x, device="cuda")
    b = a if alias == "both" else cx.tensor(y, device="cuda")
    out = {"left": a, "right": b, "both": a}.get(alias)
    if alias == "borrowed_left":
        out = cx.from_dlpack(a)
    if out is None:
        # Unwritten suffix must retain NaN payloads, signed zero and subnormals.
        bits = np.resize(np.array([0x7fc01234, 0x80000000, 1, 0xbf800000], np.uint32), 257)
        out = cx.tensor(bits.view(np.float32), device="cuda")
    originals = [t.numpy().view(np.uint32).copy() for t in (a, b, out)]
    expected = launch(dsl, cpu, a.cpu(), b.cpu(), out.cpu(), n).numpy()
    result = profiled(lambda: launch(dsl, gpu, a, b, out, n), copies=int(n < 257), kernels=int(n > 0))
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6, atol=1e-6, equal_nan=True)
    np.testing.assert_array_equal(result.numpy().view(np.uint32)[n:], originals[2][n:])
    for tensor, bits in zip((a, b, out), originals):
        np.testing.assert_array_equal(tensor.numpy().view(np.uint32), bits)
    assert result._impl is not out._impl


@pytest.mark.parametrize("shape", [(), (0,), (1,), (2, 3)])
def test_cuda_output_copy_full_shapes(artifacts, shape):
    dsl, _, gpu = artifacts
    a = cx.tensor(np.full(shape, 2, np.float32), device="cuda")
    n = int(np.prod(shape))
    result = profiled(lambda: launch(dsl, gpu, a, a, a, n), copies=0, kernels=int(n > 0))
    np.testing.assert_array_equal(result.numpy(), np.full(shape, 8, np.float32))
    np.testing.assert_array_equal(a.numpy(), np.full(shape, 2, np.float32))


@pytest.mark.parametrize("n", [0, 173, 257])
def test_cuda_output_copy_legacy_manifest(n):
    require_device()
    ptx = (Path(__file__).parents[1] / "cpp/fixtures/cuda_add_sm52.ptx").read_text()
    module = _core._load_cuda_kernel(ptx, "tensorcx_add_v1", "tttu", 2, 3)
    a = cx.tensor(np.arange(257, dtype=np.float32), device="cuda")
    result = profiled(lambda: cx.Tensor(_core._launch_cuda_kernel(module, [a._impl, a._impl, a._impl, n], n, 7)),
                      copies=1, kernels=int(n > 0))
    expected = np.arange(257, dtype=np.float32)
    expected[:n] *= 2
    np.testing.assert_array_equal(result.numpy(), expected)
    np.testing.assert_array_equal(a.numpy(), np.arange(257, dtype=np.float32))


def test_cuda_output_copy_pool_poison(artifacts):
    dsl, _, gpu = artifacts
    a = cx.tensor(np.linspace(1, 2, 257, dtype=np.float32), device="cuda")
    original = a.numpy()
    with cx.memory_pool() as pool:
        poison = cx.tensor(np.full(257, -111, np.float32), device="cuda")
        del poison
        hits = pool.stats()["hits"]
        result = profiled(lambda: launch(dsl, gpu, a, a, a, 257), copies=0, kernels=1)
        assert pool.stats()["hits"] > hits
        np.testing.assert_allclose(result.numpy(), (original + original) * original, rtol=1e-6)
        np.testing.assert_array_equal(a.numpy(), original)
        del result
    assert pool.stats()["in_use_bytes"] == pool.stats()["cached_bytes"] == 0


def test_cuda_output_copy_cache_reload(monkeypatch, tmp_path):
    require_device()
    monkeypatch.setenv("TENSORCX_DISABLE_CACHE", "0")
    monkeypatch.setenv("TENSORCX_CACHE_DIR", str(tmp_path / "cache"))
    ir = add.parse_ir()
    tools = cuda.toolchain()
    source = cuda.emit_cuda(ir)
    legacy_abi = "tensorcx.cuda.v1|linux-x86_64|llvm-21.1.8|sm52|ptx78|add-f32-v1|tttu|2|3"
    old = cache.descriptor("cuda", source, tools=tools,
                           options={"lower": cuda.LOWER_FLAGS, "serialize": cuda.SERIALIZE_FLAGS}, abi=legacy_abi)
    payload = (Path(__file__).parents[1] / "cpp/fixtures/cuda_add_sm52.ptx").read_bytes()
    with cache.artifact_cache(old) as entry:
        entry.store(payload)
    first = add.compile(target="cuda", compiler="mlir")
    second = add.compile(target="cuda", compiler="mlir")
    assert not first.cache_hit and second.cache_hit
    assert len(list((tmp_path / "cache").glob("*.bin"))) == 2
    a = cx.ones((257,), device="cuda")
    output = profiled(lambda: second.launch(a, a, a, 257), copies=0, kernels=1)
    np.testing.assert_array_equal(output.numpy(), [2] * 257)
    np.testing.assert_array_equal(a.numpy(), [1] * 257)
    with cache.artifact_cache(old) as entry:
        assert entry.read() == payload
