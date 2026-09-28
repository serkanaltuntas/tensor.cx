"""Research device-lowering evidence; no generated CUDA runtime capability."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments/mlir/probe_cuda.py"
spec = importlib.util.spec_from_file_location("cortex_cuda_probe", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def serialized(ptx):
    return 'assembly = "' + ptx.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\0A") + '"'


PTX = """.version 7.8
.target sm_52
.address_size 64
.visible .entry cortex_add(
.param .u64 a, .param .u64 b, .param .u64 out, .param .u32 n
)
"""


def test_extract_abi():
    assert probe.extract_ptx(serialized(PTX)) == PTX


@pytest.mark.parametrize("source", [
    "", serialized(PTX) * 2, serialized(PTX.replace("7.8", "8.0")),
    serialized(PTX.replace("sm_52", "sm_90")), serialized(PTX.replace("cortex_add", "other")),
    serialized(PTX.replace(".u32 n", ".u64 n")),
    serialized(PTX.replace(".address_size 64", ".address_size 32")),
    serialized(PTX).replace("\\0A", "\\ZZ"),
])
def test_reject_unexpected_assembly(source):
    with pytest.raises(ValueError):
        probe.extract_ptx(source)


def require_tools():
    try:
        probe.toolchain()
    except RuntimeError as error:
        if os.environ.get("CORTEX_REQUIRE_MLIR"):
            pytest.fail(str(error))
        pytest.skip(str(error))


def test_device_lowering_without_cuda(tmp_path):
    require_tools()
    result = probe.lower(tmp_path)
    ptx = result.read_text()
    assert "%ctaid.x" in ptx and "%ntid.x" in ptx and "%tid.x" in ptx
    assert "add.rn.f32" in ptx
    assert "setp.ge.u64" in ptx  # rounded-up lanes guarded before memory accesses
    assert ".extern" not in ptx  # no device math library required by this fixture


def test_driver_runtime_probe():
    if not os.environ.get("CORTEX_REQUIRE_MLIR_CUDA"):
        pytest.skip("set CORTEX_REQUIRE_MLIR_CUDA=1 on the validated sm_52/CUDA 12.4 host")
    # Explicit hardware mode is fail-closed: missing tools, driver or GPU fails.
    result = subprocess.run([sys.executable, str(SCRIPT)], text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"result": "PASS"' in result.stdout
    assert '"cases": 48' in result.stdout
    assert '"context_restored": true' in result.stdout
