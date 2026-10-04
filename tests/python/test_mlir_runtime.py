"""Optional compiled CPU runtime contract; required LLVM CI cannot green-skip."""
from concurrent.futures import ThreadPoolExecutor
import gc
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import tensorcx as cx
from tensorcx import _core
from tensorcx._compiler import cpu
from tensorcx.experimental import KernelCompileError


@cx.experimental.kernel
def add(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = a[i] + b[i]


@cx.experimental.kernel
def multiply(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = a[i] * b[i]


@cx.experimental.kernel
def subtract(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        value = a[i] - b[i]
        scaled = value * 0.5
        out[i] = scaled


@cx.experimental.kernel
def alias(a, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = a[i] + 1.0
        out[i] = a[i] + out[i]


def require_tools():
    try:
        if not _core._cpu_kernel_supported():
            raise RuntimeError("requires Linux x86_64")
        cpu.toolchain()
    except RuntimeError as error:
        if os.environ.get("TENSORCX_REQUIRE_MLIR"):
            pytest.fail(str(error))
        pytest.skip(str(error))


@pytest.fixture(scope="module")
def compiled():
    require_tools()
    return {kernel.name: kernel.compile(target="cpu", compiler="mlir")
            for kernel in (add, multiply, subtract, alias)}


@pytest.mark.parametrize("shape", [(), (0,), (1,), (255,), (256,), (257,), (2, 3)])
@pytest.mark.parametrize("kernel", [add, multiply, subtract])
def test_native_parity(compiled, shape, kernel):
    count = int(np.prod(shape))
    a = cx.tensor(np.arange(count, dtype=np.float32).reshape(shape) * 0.125, device="cpu")
    b = cx.tensor(np.full(shape, 0.75, dtype=np.float32), device="cpu")
    out = cx.tensor(np.full(shape, -7.0, dtype=np.float32), device="cpu")
    native = compiled[kernel.name].launch(a, b, out, count)
    reference = kernel.reference(a, b, out, count)
    np.testing.assert_allclose(native.numpy(), reference.numpy(), rtol=1e-6, atol=1e-6)
    if kernel in (add, multiply):
        primitive = a + b if kernel is add else a * b
        np.testing.assert_allclose(native.numpy(), primitive.numpy())
    np.testing.assert_array_equal(out.numpy(), np.full(shape, -7, dtype=np.float32))
    assert native.shape == shape and native._impl is not out._impl


@pytest.mark.parametrize("threads", [0, 1, 255, 256, 257])
def test_partial_repeated_and_alias(compiled, threads):
    a = cx.tensor(np.full((257,), 2.0, dtype=np.float32), device="cpu")
    out = cx.tensor(np.full((257,), -9.0, dtype=np.float32), device="cpu")
    for _ in range(2):
        result = compiled["add"].launch(a, a, out, threads, thread_count=threads, block_size=7)
        np.testing.assert_array_equal(result.numpy()[:threads], 4.0)
        np.testing.assert_array_equal(result.numpy()[threads:], -9.0)
    result = compiled["alias"].launch(a, a, threads, thread_count=threads)
    np.testing.assert_array_equal(result.numpy(), alias.reference(a, a, threads, thread_count=threads).numpy())
    np.testing.assert_array_equal(a.numpy(), 2.0)


def test_concurrent_lifetime(compiled):
    native = add.compile(target="cpu", compiler="mlir")
    a = cx.ones((257,), device="cpu")
    # Each bound call owns the compiled object until its native call completes.
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(native.launch, a, a, a, 257) for _ in range(16)]
        del native
        gc.collect()
        for future in pending:
            np.testing.assert_array_equal(future.result().numpy(), 2.0)


@pytest.mark.parametrize("case", ["count", "shape", "dtype", "guard", "block", "threads", "scalar"])
def test_public_invalid_arguments(compiled, case):
    a = cx.ones((4,), device="cpu")
    args = [a, a, a, 4]
    kwargs = {}
    if case == "count": args.pop()
    if case == "shape": args[0] = cx.ones((2, 2), device="cpu")
    if case == "dtype": args[0] = cx.ones((4,), dtype="int32", device="cpu")
    if case == "guard": args[-1] = 3
    if case == "block": kwargs["block_size"] = 0
    if case == "threads": kwargs["thread_count"] = 5
    if case == "scalar": args[-1] = -1
    with pytest.raises((ValueError, TypeError)):
        compiled["add"].launch(*args, **kwargs)


@pytest.mark.parametrize("threads", [0, 4])
@pytest.mark.parametrize("case", ["count", "shape", "dtype", "guard", "block", "kind"])
def test_native_revalidates(compiled, threads, case):
    a = cx.ones((4,), device="cpu")
    args = [a._impl, a._impl, a._impl, threads]
    block = 256
    if case == "count": args.pop()
    if case == "shape": args[0] = cx.ones((2, 2), device="cpu")._impl
    if case == "dtype": args[0] = cx.ones((4,), dtype="int32", device="cpu")._impl
    if case == "guard": args[-1] = threads + 1
    if case == "block": block = 0
    if case == "kind": args[0] = 4
    with pytest.raises((ValueError, TypeError)):
        _core._launch_cpu_kernel(compiled["add"]._module, args, threads, block)


@pytest.mark.parametrize("reported_length", [0, 1, 2**60])
def test_native_arguments_do_not_trust_sequence_length(compiled, reported_length):
    class MisreportedList(list):
        def __len__(self):
            return reported_length

    a = cx.ones((4,), device="cpu")
    out = cx.zeros((4,), device="cpu")
    values = MisreportedList([a._impl, a._impl, out._impl, 4])
    result = _core._launch_cpu_kernel(compiled["add"]._module, values, 4, 2)
    np.testing.assert_array_equal(cx.Tensor(result).numpy(), [2.0] * 4)
    np.testing.assert_array_equal(out.numpy(), [0.0] * 4)


def test_explicit_api_only():
    with pytest.raises(NotImplementedError):
        add.compile(target="cpu")
    for target in ("auto", "metal"):
        with pytest.raises(ValueError, match="requires target"):
            add.compile(target=target, compiler="mlir")
    with pytest.raises(ValueError, match="compiler"):
        add.compile(compiler="unknown")
    with pytest.raises(TypeError, match="compiler"):
        add.compile(compiler=None)


@pytest.mark.parametrize("required", [False, True])
def test_runtime_require_mode(monkeypatch, required):
    monkeypatch.setattr(cpu, "toolchain", lambda: (_ for _ in ()).throw(RuntimeError("missing")))
    if required:
        monkeypatch.setenv("TENSORCX_REQUIRE_MLIR", "1")
    else:
        monkeypatch.delenv("TENSORCX_REQUIRE_MLIR", raising=False)
    with pytest.raises(pytest.fail.Exception if required else pytest.skip.Exception):
        require_tools()


def test_import_never_discovers_llvm():
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import tensorcx; "
         "assert 'tensorcx._compiler.cpu' not in sys.modules; "
         "assert 'tensorcx._compiler.emitter' not in sys.modules; "
         "assert 'tensorcx._compiler.cuda' not in sys.modules"],
        capture_output=True, text=True, check=False,
        env={**os.environ, "TENSORCX_LLVM_BIN": "/nonexistent"})
    assert result.returncode == 0, result.stderr


def test_explicit_missing_tools_never_fall_back(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORCX_LLVM_BIN", str(tmp_path))
    monkeypatch.setattr(cpu.shutil, "which", lambda _: pytest.fail("unexpected PATH fallback"))
    with pytest.raises(RuntimeError, match="requires executable"):
        cpu.toolchain()


def test_version_mismatch(monkeypatch, tmp_path):
    executable = tmp_path / "mlir-opt"
    executable.write_text("")
    executable.chmod(0o755)
    monkeypatch.setenv("TENSORCX_LLVM_BIN", str(tmp_path))
    monkeypatch.setattr(cpu, "_run", lambda _: "LLVM version 21.1.7")
    with pytest.raises(RuntimeError, match="requires LLVM 21.1.8"):
        cpu.toolchain()


@pytest.mark.parametrize("failure", ["timeout", "exit", "oserror"])
def test_subprocess_failure(monkeypatch, failure):
    def run(*args, **kwargs):
        assert kwargs["timeout"] == 60
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args[0], 60)
        if failure == "oserror":
            raise OSError("missing")
        return subprocess.CompletedProcess(args[0], 2, "", "lowering failed")
    monkeypatch.setattr(cpu.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="MLIR tool failed"):
        cpu._run(["fake-tool"])


def test_failed_compile_cleans_scratch(compiled, monkeypatch, tmp_path):
    monkeypatch.setattr(cpu.tempfile, "tempdir", str(tmp_path))
    real_run = cpu._run
    def run(command):
        if "--convert-scf-to-cf" in command:
            raise RuntimeError("lowering failed")
        return real_run(command)
    monkeypatch.setattr(cpu, "_run", run)
    with pytest.raises(RuntimeError, match="lowering failed"):
        add.compile(target="cpu", compiler="mlir")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("body", [
    "out[i] = a[i + 1] + b[i]",
    "out[i] = a[i] + n",
    "if a[i] < b[i]:\n            out[i] = a[i]",
    "for j in range(n):\n            out[i] = a[i]",
    "i = i + 1\n        out[i] = a[i] + b[i]",
])
def test_unsupported_runtime_subset_before_tools(monkeypatch, body):
    # Alter immutable parsed IR via a real source function to keep parser rules.
    # Compiling these source strings via a file lets inspect retrieve the body.
    import tempfile
    import importlib.util
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "unsupported.py"
        path.write_text(
            "import tensorcx as cx\n@cx.experimental.kernel\ndef bad(a, b, out, n):\n"
            "    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()\n"
            "    if i < n:\n        " + body + "\n")
        spec = importlib.util.spec_from_file_location("unsupported", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setattr(cpu, "toolchain", lambda: pytest.fail("tools called before subset validation"))
        with pytest.raises(KernelCompileError):
            module.bad.compile(target="cpu", compiler="mlir")


def test_libc_symbol_name_is_local(compiled):
    from dataclasses import replace
    ir = replace(add.parse_ir(), name="abs")
    native = cpu.compile_kernel(ir)
    a = cx.ones((8,), device="cpu")
    np.testing.assert_array_equal(native.launch(a, a, a, 8).numpy(), 2.0)


def test_successful_compile_cleans_scratch(compiled, monkeypatch, tmp_path):
    monkeypatch.setattr(cpu.tempfile, "tempdir", str(tmp_path))
    native = add.compile(target="cpu", compiler="mlir")
    assert list(tmp_path.iterdir()) == []
    a = cx.ones((1,), device="cpu")
    np.testing.assert_array_equal(native.launch(a, a, a, 1).numpy(), [2.0])


def test_empty_explicit_directory(monkeypatch):
    monkeypatch.setenv("TENSORCX_LLVM_BIN", "")
    with pytest.raises(RuntimeError, match="must name"):
        cpu.toolchain()


@pytest.mark.parametrize("threads", [0, 3, 6])
def test_reshape_views_preserve_alias_remapping_and_lifetime(compiled, threads):
    values = np.arange(2, 8, dtype=np.float32)
    original = cx.tensor(values.reshape(2, 3), device="cpu")
    input_view = original.reshape((-1,))
    output_view = original.reshape((3, 2)).reshape((-1,))
    surviving_view = original.reshape((-1,))
    expected = values.copy()
    # The first store is visible through the aliased input on the second store.
    # Independent copies would incorrectly produce 2*x+1 instead of 2*(x+1).
    expected[:threads] = 2 * (values[:threads] + 1)
    result = compiled["alias"].launch(input_view, output_view, threads, thread_count=threads)
    np.testing.assert_array_equal(result.numpy(), expected)
    reference = alias.reference(input_view, output_view, threads, thread_count=threads)
    np.testing.assert_array_equal(reference.numpy(), expected)
    for view in (original, input_view, output_view, surviving_view):
        np.testing.assert_array_equal(view.numpy().reshape(-1), values)
    result_view = result.reshape((2, 3))
    del original, input_view, output_view, result, reference, view
    gc.collect()
    np.testing.assert_array_equal(result_view.numpy().reshape(-1), expected)
    np.testing.assert_array_equal(surviving_view.numpy(), values)
    another_view = surviving_view.reshape((3, 2)).reshape((-1,))
    np.testing.assert_array_equal(
        compiled["alias"].launch(surviving_view, another_view, threads, thread_count=threads).numpy(), expected,
    )
