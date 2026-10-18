"""Local installation, declared operation contracts and memory snapshots."""
from __future__ import annotations

import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

from . import _core, backend as _backend
from .device import Device, _normalize_device


def capabilities(device: str | Device = "cpu") -> dict[str, list[str]]:
    """Return declared input dtypes by operation for an available backend.

    This is metadata, not a kernel execution test. Every call returns a copy.
    """
    route = _backend.require_backend(_normalize_device(device))
    return {name: list(dtypes) for name, dtypes in sorted((route.operations or {}).items())}


def _sysconf(name):
    try:
        value = os.sysconf(name)
        return value if value > 0 else None
    except (AttributeError, OSError, ValueError):
        return None


def _host_memory():
    page = _sysconf("SC_PAGE_SIZE")
    total = _sysconf("SC_PHYS_PAGES")
    # Linux MemAvailable includes reclaimable caches. Free pages alone would
    # misleadingly label cached memory as unavailable.
    available = None
    if sys.platform == "linux":
        try:
            text = Path("/proc/meminfo").read_text()
            match = re.search(r"^MemAvailable:\s+(\d+)\s+kB$", text, re.MULTILINE)
            if match:
                available = int(match[1]) * 1024
        except OSError:
            pass
    return {"scope": "host_physical", "total_bytes": page * total if page and total else None,
            "available_bytes": available}


def memory_info(device: str | Device = "cpu") -> dict:
    """Host or device memory snapshot; never tensor-owned allocation accounting."""
    route = _backend.require_backend(_normalize_device(device))
    if route.diagnostic_info is None:
        return {"scope": "not_collected", "total_bytes": None, "available_bytes": None}
    return dict(route.diagnostic_info()["memory"])


def _llvm_info(check_tools, cuda_available=False):
    from ._compiler.cpu import LLVM_VERSION
    explicit = os.environ.get("TENSORCX_LLVM_BIN")
    tools = {}
    for name in ("mlir-opt", "mlir-translate", "clang"):
        candidate = str(Path(explicit) / name) if explicit is not None else shutil.which(name)
        found = bool(explicit != "" and candidate and Path(candidate).is_file()
                     and os.access(candidate, os.X_OK))
        record = {"status": "not_checked" if found else "missing", "version": None}
        if found and check_tools:
            try:
                result = subprocess.run([candidate, "--version"], capture_output=True,
                                        text=True, timeout=3, check=False)
                match = re.search(r"\bversion\s+(\d+\.\d+\.\d+)(?![\d.])", result.stdout)
                record["version"] = match[1] if match else None
                record["status"] = ("ready" if result.returncode == 0 and record["version"] == LLVM_VERSION
                                    else "incompatible")
            except subprocess.TimeoutExpired:
                record["status"] = "timeout"
            except (OSError, UnicodeError):
                record["status"] = "error"
        tools[name] = record
    states = {value["status"] for value in tools.values()}
    status = "ready" if states == {"ready"} else "missing" if "missing" in states else "not_checked" if not check_tools else "incompatible"
    cpu_target = bool(_core._cpu_kernel_supported())
    cuda_target = False
    if cuda_available and hasattr(_core, "_cuda_kernel_support"):
        try:
            _core._cuda_kernel_support()
            cuda_target = True
        except (RuntimeError, OSError):
            pass
    return {"required_version": LLVM_VERSION, "status": status, "tools": tools,
            "runtime_targets": {"cpu": cpu_target, "cuda": cuda_target},
            "ready_targets": {"cpu": cpu_target and status == "ready",
                              "cuda": cuda_target and status == "ready"},
            "hint": "Set TENSORCX_LLVM_BIN to an executable LLVM 21.1.8 toolchain; ordinary tensor operations do not require LLVM."}


def diagnostics(*, check_tools: bool = False) -> dict:
    """Return a JSON-compatible local report without paths or credentials.

    ``check_tools=True`` runs three external LLVM version probes (3 seconds
    each). No kernels, allocations, downloads or configuration writes are made;
    device discovery can initialize the accelerator context.
    """
    if not isinstance(check_tools, bool):
        raise TypeError("check_tools must be bool")
    build = dict(_core._build_info())
    backends = {}
    names = dict.fromkeys((*build, *_backend.backend_names(include_unavailable=True)))
    for name in names:
        family = name.partition(":")[0]
        info = {"compiled": build.get(family), "available": None,
                "operations": {}, "memory": {"scope": "not_collected", "total_bytes": None,
                                                "available_bytes": None}}
        try:
            info["available"] = _backend.is_available(name)
            if info["available"]:
                route = _backend.require_backend(name)
                info["operations"] = {op: list(dtypes) for op, dtypes in sorted((route.operations or {}).items())}
                info["name"] = route.device_name()
                if route.diagnostic_info:
                    info.update(route.diagnostic_info())
            else:
                info["hint"] = ("Backend was not compiled into this package. Install or build the matching variant."
                                if build.get(family) is False else
                                "No usable device discovered. Check device visibility and driver/runtime compatibility.")
        except (RuntimeError, OSError, ValueError):
            info["query_status"] = "failed"
            info["hint"] = "Device information query failed; check device/driver health."
        backends[name] = info
    return {"schema_version": 1, "status": "ok", "version": _core.version(),
            "python": platform.python_version(), "system": platform.system(),
            "machine": platform.machine(), "backends": backends,
            "llvm": _llvm_info(check_tools, backends.get("cuda", {}).get("available") is True)}
