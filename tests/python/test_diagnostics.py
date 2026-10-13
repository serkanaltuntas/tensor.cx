"""Declared capabilities must agree with execution; diagnostics stay bounded."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import tensorcx as cx
from tensorcx import backend
import importlib
from dataclasses import replace
diag = importlib.import_module("tensorcx.diagnostics")


def test_report_and_memory_are_json_snapshots():
    report = cx.diagnostics()
    assert report["status"] == "ok"
    assert report["backends"]["cpu"]["compiled"] is True
    assert report["backends"]["cpu"]["available"] is True
    assert report["backends"]["cpu"]["operations"] == cx.capabilities("cpu")
    assert report["llvm"]["status"] in ("not_checked", "missing")
    assert not any(report["llvm"]["ready_targets"].values())
    assert "cuda" in report["backends"] and "metal" in report["backends"]
    serialized = json.dumps(report)
    assert str(Path.home()) not in serialized
    assert "GPU-" not in serialized
    for device in cx.devices():
        memory = cx.memory_info(cx.device(device))
        assert set(memory) == {"scope", "total_bytes", "available_bytes"}
        if memory["total_bytes"] is not None:
            assert memory["total_bytes"] > 0
        if memory["available_bytes"] is not None:
            assert 0 <= memory["available_bytes"] <= memory["total_bytes"]
    report["backends"]["cpu"]["operations"]["zeros"].clear()
    assert cx.capabilities("cpu")["zeros"]


@pytest.mark.parametrize("query", [cx.capabilities, cx.memory_info])
def test_selection_errors(query):
    with pytest.raises(ValueError, match="not available"):
        query("missing")
    with pytest.raises(ValueError, match="index 0"):
        query("cpu:1")
    with pytest.raises(TypeError, match="device must"):
        query(1)


def _exercise(name, dtype, device):
    x = cx.tensor([[1, 2], [3, 4]], dtype=dtype, device=device)
    mask = cx.tensor([[True, False], [False, True]], device=device)
    unary = {"negative": lambda: -x, "indexing": lambda: x[:, :1],
             "copy": lambda: x.cpu().to(device), "reshape": lambda: x.reshape((4,)),
             "squeeze": lambda: cx.squeeze(cx.expand_dims(x, 0), 0),
             "expand_dims": lambda: cx.expand_dims(x, 0),
             "transpose": lambda: cx.transpose(x), "astype": lambda: cx.astype(x, "float32"),
             "from_dlpack": lambda: cx.from_dlpack(x), "__dlpack__": lambda: x.__dlpack__(),
             "sum": lambda: cx.sum(x), "max": lambda: cx.max(x),
             "mean": lambda: cx.mean(x), "min": lambda: cx.min(x),
             "argmax": lambda: cx.argmax(x), "topk": lambda: cx.topk(x, 1),
             "clip": lambda: cx.clip(x, 1, 3), "where": lambda: cx.where(mask, x, x),
             "masked_select": lambda: cx.masked_select(x, mask),
             "matmul": lambda: cx.matmul(x, x), "linear": lambda: cx.linear(x, x),
             "embedding": lambda: cx.embedding(cx.tensor([0, 1], dtype="int32", device=device), x),
             "attention": lambda: cx.attention(x, x, x),
             "scaled_dot_product_attention": lambda: cx.scaled_dot_product_attention(x, x, x),
             "stack": lambda: cx.stack([x, x]), "concat": lambda: cx.concat([x, x]),
             "split": lambda: cx.split(x, 2), "tensor": lambda: x,
             "empty": lambda: cx.empty((2,), dtype=dtype, device=device),
             "zeros": lambda: cx.zeros((2,), dtype=dtype, device=device),
             "ones": lambda: cx.ones((2,), dtype=dtype, device=device),
             "randn": lambda: cx.randn((2,), dtype=dtype, device=device),
             "add": lambda: x + x, "subtract": lambda: x - x,
             "multiply": lambda: x * x, "divide": lambda: x / x}
    if name in unary:
        return unary[name]()
    if name in ("softmax", "rmsnorm", "layernorm"):
        return getattr(cx, name)(x, axis=-1)
    if name in ("equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
                "logical_and", "logical_or", "logical_xor"):
        return getattr(cx, name)(x, x)
    return getattr(cx, name)(x)


_CASES = [(device, name, dtype) for device in cx.devices()
          for name, dtypes in cx.capabilities(device).items() for dtype in dtypes]


@pytest.mark.parametrize("device,name,dtype", _CASES)
def test_declared_operations_execute(device, name, dtype):
    _exercise(name, dtype, device)


def test_cuda_int32_gaps_are_reported_honestly():
    if not cx.is_available("cuda"):
        pytest.skip("CUDA unavailable")
    caps = cx.capabilities("cuda")
    x = cx.tensor([1, 2], dtype="int32", device="cuda")
    for name, action in (("zeros", lambda: cx.zeros((2,), dtype="int32", device="cuda")),
                         ("add", lambda: x + x), ("sum", lambda: cx.sum(x))):
        assert "int32" not in caps[name]
        with pytest.raises(ValueError, match="float32"):
            action()
    assert "int32" in caps["min"] and "bool" in caps["where"]


def test_native_cuda_info_preserves_foreign_context():
    if not cx.is_available("cuda"):
        pytest.skip("CUDA unavailable")
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    driver.cuCtxPushCurrent_v2.argtypes = [ctypes.c_void_p]
    driver.cuCtxPopCurrent_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    driver.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    driver.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
    before, foreign, popped, current = (ctypes.c_void_p() for _ in range(4))
    assert driver.cuCtxGetCurrent(ctypes.byref(before)) == 0
    assert driver.cuCtxCreate_v2(ctypes.byref(foreign), 0, 0) == 0
    try:
        info = cx._core._cuda_device_info()
        assert driver.cuCtxGetCurrent(ctypes.byref(current)) == 0
        assert current.value == foreign.value
        assert info["architecture"].startswith("sm_")
        assert info["runtime_version"] > 0 and info["driver_api_version"] > 0
        cx.memory_info("cuda")
        assert driver.cuCtxGetCurrent(ctypes.byref(current)) == 0
        assert current.value == foreign.value
    finally:
        assert driver.cuCtxPopCurrent_v2(ctypes.byref(popped)) == 0
        assert popped.value == foreign.value
        assert driver.cuCtxDestroy_v2(foreign) == 0
    assert driver.cuCtxGetCurrent(ctypes.byref(current)) == 0
    assert current.value == before.value


def test_tool_probes_are_opt_in_bounded_and_redacted(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORCX_LLVM_BIN", str(tmp_path))
    for name in ("mlir-opt", "mlir-translate", "clang"):
        file = tmp_path / name
        file.write_text("#!/bin/sh\necho 'LLVM version 21.1.8'\n")
        file.chmod(0o755)
    def run(command, **kwargs):
        assert kwargs["timeout"] == 3 and command[-1] == "--version"
        return subprocess.CompletedProcess(command, 0, "LLVM version 21.1.8\nInstalledDir: /private/secret", "")
    monkeypatch.setattr(diag.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected external execution"))
    assert cx.diagnostics()["llvm"]["status"] == "not_checked"
    monkeypatch.setattr(diag.subprocess, "run", run)
    result = cx.diagnostics(check_tools=True)
    assert result["llvm"]["status"] == "ready"
    assert "/private/secret" not in json.dumps(result)
    monkeypatch.setattr(diag.subprocess, "run", lambda command, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired(command, 3)))
    result = cx.diagnostics(check_tools=True)
    assert all(t["status"] == "timeout" for t in result["llvm"]["tools"].values())
    monkeypatch.setenv("TENSORCX_LLVM_BIN", "")
    assert cx.diagnostics(check_tools=True)["llvm"]["status"] == "missing"
    with pytest.raises(TypeError, match="bool"):
        cx.diagnostics(check_tools=1)


def test_cli_json_and_hidden_gpu(tmp_path):
    for hidden in (False, True):
        environment = os.environ.copy()
        if hidden:
            environment["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run([sys.executable, "-I", "-m", "tensorcx_doctor", "--json"],
                                cwd=tmp_path, env=environment, capture_output=True, text=True,
                                timeout=30, check=True)
        report = json.loads(result.stdout)
        assert report["status"] == "ok"
        if hidden:
            assert report["backends"]["cuda"]["available"] is False


def test_cli_explains_broken_import_without_leaking_paths(monkeypatch, capsys):
    import tensorcx_doctor
    def fail(_):
        raise ImportError(str(Path.home() / "credentials.env") + " libcuda.so.1 missing")
    monkeypatch.setattr(tensorcx_doctor.importlib, "import_module", fail)
    assert tensorcx_doctor.main(["--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "import_error"
    assert "libcuda" not in json.dumps(report) and str(Path.home()) not in json.dumps(report)


@pytest.mark.parametrize("field", ["is_available", "device_name", "diagnostic_info"])
def test_backend_query_failure_is_contained(field, monkeypatch, capsys):
    import tensorcx_doctor
    def fail():
        raise RuntimeError("/private/local/path device discovery failed")
    monkeypatch.setitem(backend._BACKENDS, "cpu", replace(backend._BACKENDS["cpu"], **{field: fail}))
    report = cx.diagnostics()
    assert report["backends"]["cpu"]["query_status"] == "failed"
    assert "llvm" in report and "metal" in report["backends"]
    assert "/private/local" not in json.dumps(report)
    assert tensorcx_doctor.main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out)["backends"]["cpu"]["query_status"] == "failed"


def test_cli_contains_collector_failure(monkeypatch, capsys):
    import tensorcx_doctor
    def fail(**kwargs):
        raise RuntimeError(str(Path.home()) + " native mismatch")
    monkeypatch.setattr(cx, "diagnostics", fail)
    assert tensorcx_doctor.main(["--json"]) == 1
    result = capsys.readouterr().out
    assert json.loads(result)["status"] == "diagnostics_error"
    assert str(Path.home()) not in result
