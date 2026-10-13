"""Persistent Metal code artifacts; native pipelines remain device-owned."""
import platform
import shutil
import subprocess

from .. import experimental
from ..experimental import KernelCompileError
from .cache import artifact_cache, cache_enabled, descriptor


def _sdk_info(xcrun, *arguments):
    try:
        result = subprocess.run([xcrun, "-sdk", "macosx", *arguments],
                                capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise KernelCompileError("Metal compiler toolchain query failed") from error
    if result.returncode:
        raise KernelCompileError("Metal compiler toolchain query failed")
    return result.stdout.strip()


def compile_artifact(name, source, *, cache=True):
    if not cache_enabled(cache):
        return experimental._compile_msl_to_metallib(name, source), False
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        raise KernelCompileError("Metal compiler toolchain is unavailable: xcrun was not found")
    tools = {tool: _sdk_info(xcrun, "--find", tool) for tool in ("metal", "metallib")}
    description = descriptor("metal", source, tools=tools,
                             options=["-sdk", "macosx", "metal", "-c", "metallib"],
                             abi={"format": "metallib-v1", "entry": name,
                                  "sdk": _sdk_info(xcrun, "--show-sdk-version"),
                                  "os": platform.mac_ver()[0]})
    with artifact_cache(description) as entry:
        payload = entry.read()
        hit = payload is not None
        if not hit:
            payload = experimental._compile_msl_to_metallib(name, source)
            entry.store(payload)
    return payload, hit
