"""MLIR CUDA runtime: narrow public contract and native defensive checks."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gc
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import cortex_runtime as cx
from cortex_runtime import _core
from cortex_runtime._compiler import cuda
from cortex_runtime.experimental import IRBinaryOp, IRLoad, KernelCompileError


@cx.experimental.kernel
def add(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = a[i] + b[i]


@cx.experimental.kernel
def subtract(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = a[i] - b[i]


@cx.experimental.kernel
def multiply(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = a[i] * b[i]


@cx.experimental.kernel
def reordered(n, out, b, a):
    j = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if j < n:
        out[j] = a[j] + b[j]


@cx.experimental.kernel
def reordered_subtract(n, out, b, a):
    j = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if j < n:
        out[j] = a[j] - b[j]


@cx.experimental.kernel
def reordered_multiply(n, out, b, a):
    j = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if j < n:
        out[j] = a[j] * b[j]


def require_device():
    try:
        if not hasattr(_core, "_cuda_kernel_support"):
            raise RuntimeError("CUDA build unavailable")
        _core._cuda_kernel_support()
        cuda.toolchain()
    except RuntimeError as error:
        if os.environ.get("CORTEX_REQUIRE_MLIR_CUDA"):
            pytest.fail(str(error))
        pytest.skip(str(error))


@pytest.fixture(scope="module", params=[add, subtract, multiply], ids=["add", "sub", "mul"])
def kernel(request):
    return request.param


@pytest.fixture(scope="module")
def compiled(kernel):
    require_device()
    return kernel.compile(target="cuda", compiler="mlir")


@pytest.fixture(scope="module")
def cpu_compiled(compiled, kernel):
    return kernel.compile(target="cpu", compiler="mlir")


@pytest.mark.parametrize("shape", [(), (0,), (1,), (255,), (256,), (257,), (2, 3)])
@pytest.mark.parametrize("block", [1, 7, 256])
def test_parity(compiled, cpu_compiled, kernel, shape, block):
    size = int(np.prod(shape))
    a = cx.tensor(np.arange(size, dtype=np.float32).reshape(shape) * 0.25, device="cpu")
    b = cx.tensor(np.full(shape, 0.75, dtype=np.float32), device="cpu")
    out = cx.tensor(np.full(shape, -11, dtype=np.float32), device="cpu")
    ga, gb, go = a.to("cuda"), b.to("cuda"), out.to("cuda")
    result = compiled.launch(ga, gb, go, size, block_size=block)
    expected = kernel.reference(a, b, out, size, block_size=block)
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.numpy(), cpu_compiled.launch(a, b, out, size, block_size=block).numpy(),
                               rtol=1e-6, atol=1e-6)
    operation = {"add": np.add, "subtract": np.subtract, "multiply": np.multiply}[kernel.name]
    np.testing.assert_allclose(result.numpy(), operation(a.numpy(), b.numpy()), rtol=1e-6, atol=1e-6)
    if kernel.name != "subtract":
        static = ga + gb if kernel.name == "add" else ga * gb
        np.testing.assert_allclose(result.numpy(), static.numpy())
    np.testing.assert_array_equal(go.numpy(), out.numpy())
    assert result.shape == shape and result.device == "cuda" and result._impl is not go._impl


@pytest.mark.parametrize("threads", [0, 1, 255, 256, 257])
@pytest.mark.parametrize("alias", ["none", "left", "right", "both"])
def test_prefix_alias_repeat(compiled, kernel, threads, alias):
    a = cx.tensor(np.arange(257, dtype=np.float32), device="cuda")
    b = a if alias == "both" else cx.tensor(np.full(257, 0.75, dtype=np.float32), device="cuda")
    out = {"left": a, "right": b, "both": a}.get(alias)
    if out is None:
        out = cx.tensor(np.full(257, -9, dtype=np.float32), device="cuda")
    originals = [value.numpy().copy() for value in (a, b, out)]
    for _ in range(2):
        result = compiled.launch(a, b, out, threads, thread_count=threads, block_size=7)
        expected = kernel.reference(a.cpu(), b.cpu(), out.cpu(), threads, thread_count=threads, block_size=7)
        np.testing.assert_array_equal(result.numpy(), expected.numpy())
    for value, original in zip((a, b, out), originals):
        np.testing.assert_array_equal(value.numpy(), original)


def test_reordered_scalar_first(compiled, kernel):
    dsl = {"add": reordered, "subtract": reordered_subtract, "multiply": reordered_multiply}[kernel.name]
    native = dsl.compile(target="cuda", compiler="mlir")
    a = cx.tensor(np.array([1, -3, 0.25, 8], dtype=np.float32), device="cuda")
    b = cx.tensor(np.array([2, 5, -4, 0.5], dtype=np.float32), device="cuda")
    expected = dsl.reference(4, a.cpu(), b.cpu(), a.cpu()).numpy()
    np.testing.assert_array_equal(native.launch(4, a, b, a).numpy(), expected)


def test_concurrent_lifetime(compiled, kernel):
    native = kernel.compile(target="cuda", compiler="mlir")
    a = cx.ones((257,), device="cuda")
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(native.launch, a, a, a, 257) for _ in range(16)]
        del native
        gc.collect()
        for future in pending:
            np.testing.assert_array_equal(future.result().numpy(), {"add": 2, "subtract": 0, "multiply": 1}[kernel.name])
    np.testing.assert_array_equal(a.numpy(), 1)


@pytest.mark.parametrize("threads", [0, 4])
@pytest.mark.parametrize("case", ["count", "shape", "dtype", "guard", "block_zero", "block_limit", "scalar", "device"])
def test_invalid_public_launch(compiled, case, threads):
    a = cx.ones((4,), device="cuda")
    args = [a, a, a, threads]
    block = 256
    if case == "count": args.pop()
    if case == "shape": args[0] = cx.ones((2, 2), device="cuda")
    if case == "dtype": args[0] = cx.tensor(np.ones(4, dtype=np.int32), device="cuda")
    if case == "guard": args[-1] = threads + 1
    if case == "block_zero": block = 0
    if case == "block_limit": block = 1025
    if case == "scalar": args[-1] = -1
    if case == "device": args[0] = a.cpu()
    with pytest.raises((ValueError, TypeError)):
        compiled.launch(*args, thread_count=threads, block_size=block)
    np.testing.assert_array_equal(a.numpy(), 1)


@pytest.mark.parametrize("case", ["count", "kind", "guard", "block", "shape", "dtype"])
def test_native_revalidation(compiled, case):
    a = cx.ones((4,), device="cuda")
    values = [a._impl, a._impl, a._impl, 0]
    block = 256
    if case == "count": values.pop()
    if case == "kind": values[0] = 0
    if case == "guard": values[-1] = 1
    if case == "block": block = 1025
    if case == "shape": values[0] = cx.ones((2, 2), device="cuda")._impl
    if case == "dtype": values[0] = cx.tensor(np.ones(4, dtype=np.int32), device="cuda")._impl
    with pytest.raises((ValueError, TypeError)):
        _core._launch_cuda_kernel(compiled._module, values, 0, block)


def test_native_ptx_errors(compiled):
    ptx = (Path(__file__).parents[1] / "cpp/fixtures/cuda_add_sm52.ptx").read_text()
    for invalid in ("", ptx.replace("sm_52", "sm_90"), ptx.replace("ptx78", "ptx80"),
                    ptx.replace(".version 7.8", ".version 9.9"), ptx + "\x00",
                    ptx.replace("add.rn.f32", "invalid.opcode"),
                    ptx.replace(".entry cortex_add_v1", ".entry absent"),
                    ptx.replace(".u32 cortex_add_v1_param_3", ".u64 cortex_add_v1_param_3")):
        with pytest.raises((ValueError, RuntimeError)):
            _core._load_cuda_kernel(invalid, cuda.ENTRY, "tttu", 2, 3)


def test_missing_gpu_is_explicit(compiled):
    command = "from cortex_runtime import _core; _core._cuda_kernel_support()"
    result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, timeout=30)
    assert result.returncode != 0 and "RuntimeError" in result.stderr


def test_default_cuda_compile_not_silent_fallback():
    with pytest.raises(NotImplementedError):
        add.compile(target="cuda")


@pytest.mark.parametrize("change", ["div", "mixed", "index", "output_read", "extra_store"])
def test_unsupported_before_tools(monkeypatch, change):
    ir = add.parse_ir()
    statement = ir.body[1].body[0]
    value = statement.value
    if change == "div":
        value = replace(value, op=change)
    if change == "nested":
        value = replace(value, rhs=value)
    if change == "mixed":
        value = replace(value, rhs=ir.body[1].condition.rhs)
    if change == "index":
        value = replace(value, lhs=IRLoad("a", IRBinaryOp("add", value.lhs.index, value.lhs.index)))
    if change == "output_read":
        value = replace(value, lhs=IRLoad("out", value.lhs.index))
    body = (replace(statement, value=value),)
    if change == "extra_store":
        body = body + body
    ir = replace(ir, body=(ir.body[0], replace(ir.body[1], body=body)))
    monkeypatch.setattr(cuda, "toolchain", lambda: pytest.fail("invalid IR reached tools"))
    with pytest.raises(KernelCompileError):
        cuda.compile_kernel(ir)


@pytest.mark.parametrize("dsl,operation", [(add, "add"), (subtract, "sub"), (multiply, "mul")])
def test_gpu_emitter_and_fixture_without_device(tmp_path, dsl, operation):
    try:
        cuda.toolchain()
    except RuntimeError as error:
        if os.environ.get("CORTEX_REQUIRE_MLIR"):
            pytest.fail(str(error))
        pytest.skip(str(error))
    source, ptx = cuda.lower(dsl.parse_ir())
    assert "gpu.block_id" in source and "scf.for" not in source
    fixture = (Path(__file__).parents[1] / f"cpp/fixtures/cuda_{operation}_sm52.ptx").read_text()
    assert ptx == fixture


def test_malformed_serialized_assembly():
    valid = '#gpu.object<assembly = ".version 7.8\\0A.target sm_52\\0A.address_size 64\\0A.visible .entry cortex_add_v1(.param .u64 .ptr a,.param .u64 .ptr b,.param .u64 .ptr out,.param .u32 n)">'
    assert ".entry" in cuda.extract_ptx(valid, "tttu")
    for bad in ("", valid * 2, valid + 'assembly = "bad"',
                valid.replace("\\0A", "\\ZZ"),
                valid.replace("sm_52", "sm_90"), valid.replace(".address_size 64", ".address_size 32"),
                valid.replace(".u32 n", ".u64 n"), valid.replace(".ptr", ""),
                valid.replace("cortex_add_v1", "wrong"), valid.replace('n)">', 'n)\\00">')):
        with pytest.raises(ValueError):
            cuda.extract_ptx(bad, "tttu")


def test_compile_cleanup_on_failure(compiled, monkeypatch, tmp_path):
    monkeypatch.setattr(cuda.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(cuda, "_run", lambda _: (_ for _ in ()).throw(RuntimeError("tool failure")))
    with pytest.raises(RuntimeError, match="tool failure"):
        add.compile(target="cuda", compiler="mlir")
    assert list(tmp_path.iterdir()) == []


def test_compile_cleanup_on_success(compiled, monkeypatch, tmp_path):
    monkeypatch.setattr(cuda.tempfile, "tempdir", str(tmp_path))
    native = add.compile(target="cuda", compiler="mlir")
    assert list(tmp_path.iterdir()) == []
    a = cx.ones((1,), device="cuda")
    np.testing.assert_array_equal(native.launch(a, a, a, 1).numpy(), [2])


def test_cuda_disabled_build_error(monkeypatch):
    monkeypatch.delattr(_core, "_cuda_kernel_support", raising=False)
    with pytest.raises(RuntimeError, match="CUDA-enabled Cortex build"):
        add.compile(target="cuda", compiler="mlir")


def test_float_edges(compiled, cpu_compiled, kernel):
    tiny = np.finfo(np.float32).tiny
    values = np.array([0.0, -0.0, tiny / 2, -tiny / 2, 1e10, -3.5, np.inf, -np.inf, np.nan], dtype=np.float32)
    rhs = np.array([-0.0, 0.0, 0.5, 2.0, -1e10, 0.25, 2.0, -2.0, 1.0], dtype=np.float32)
    a, b, out = cx.tensor(values), cx.tensor(rhs), cx.zeros(values.shape)
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        expected = kernel.reference(a, b, out, values.size).numpy()
        cpu = cpu_compiled.launch(a, b, out, values.size).numpy()
        actual = compiled.launch(a.to("cuda"), b.to("cuda"), out.to("cuda"), values.size).numpy()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, cpu)
    np.testing.assert_array_equal(np.signbit(actual[actual == 0]), np.signbit(expected[expected == 0]))


@pytest.mark.parametrize("operation", ["add", "sub", "mul"])
def test_native_operation_manifest(compiled, operation):
    ptx = (Path(__file__).parents[1] / f"cpp/fixtures/cuda_{operation}_sm52.ptx").read_text()
    for other in {"add", "sub", "mul"} - {operation}:
        with pytest.raises(ValueError, match="manifest"):
            _core._load_cuda_kernel(ptx, cuda.entry_name(other), "tttu", 2, 3)
