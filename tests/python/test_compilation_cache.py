"""Private cache integrity, process coordination and real compiler reuse."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
import tensorcx as cx
from tensorcx._compiler import cache, cpu, cuda


@cx.experimental.kernel
def cached_add(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = a[i] + b[i]


@pytest.fixture
def cache_dir(monkeypatch, tmp_path):
    path = tmp_path / "private-cache"
    monkeypatch.setenv("TENSORCX_DISABLE_CACHE", "0")
    monkeypatch.setenv("TENSORCX_CACHE_DIR", str(path))
    return path


def _key(description):
    return cache.digest(json.dumps(description, sort_keys=True, separators=(",", ":")).encode())


def test_round_trip_permissions_and_atomic_files(cache_dir):
    with cache.artifact_cache({"target": "cpu", "source": "one"}) as entry:
        assert entry.read() is None
        entry.store(b"compiled bytes")
        assert entry.read() == b"compiled bytes"
    assert cache_dir.stat().st_mode & 0o777 == 0o700
    assert all(file.stat().st_mode & 0o777 == 0o600 for file in cache_dir.iterdir())
    assert not list(cache_dir.glob(".tmp-*"))
    with cache.artifact_cache({"target": "cuda", "source": "one"}) as entry:
        assert entry.read() is None


@pytest.mark.parametrize("damage", [b"", b"bad", cache.MAGIC + b"{}\nbad",
                                    cache.MAGIC + b"not-json\n", cache.MAGIC + b"[]\n"])
def test_corrupted_entry_is_a_rebuildable_miss(cache_dir, damage):
    description = {"target": "cpu"}
    with cache.artifact_cache(description) as entry:
        entry.store(b"original")
    artifact = next(cache_dir.glob("*.bin"))
    artifact.write_bytes(damage)
    with cache.artifact_cache(description) as entry:
        assert entry.read() is None
        entry.store(b"replacement")
    with cache.artifact_cache(description) as entry:
        assert entry.read() == b"replacement"


def test_checksum_tampering_and_wrong_key(cache_dir):
    description = {"source": "kernel"}
    with cache.artifact_cache(description) as entry:
        entry.store(b"original")
    artifact = next(cache_dir.glob("*.bin"))
    artifact.write_bytes(artifact.read_bytes().replace(b"original", b"modified"))
    with cache.artifact_cache(description) as entry:
        assert entry.read() is None
        entry.store(b"fixed")
    text = artifact.read_bytes().replace(_key(description).encode(), b"0" * 64)
    artifact.write_bytes(text)
    with cache.artifact_cache(description) as entry:
        assert entry.read() is None


@pytest.mark.parametrize("name", ["artifact", "lock", "root"])
def test_symlinks_do_not_read_or_modify_other_files(cache_dir, tmp_path, name):
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"untouched")
    description = {"source": "x"}
    if name == "root":
        cache_dir.symlink_to(outside, target_is_directory=True)
        with pytest.raises(RuntimeError, match="unsafe"):
            with cache.artifact_cache(description):
                pytest.fail("unsafe root accepted")
    else:
        cache_dir.mkdir(mode=0o700)
        suffix = ".bin" if name == "artifact" else ".lock"
        (cache_dir / (_key(description) + suffix)).symlink_to(sentinel)
        if name == "lock":
            with pytest.raises(RuntimeError, match="unsafe"):
                with cache.artifact_cache(description):
                    pytest.fail("unsafe lock accepted")
        else:
            with cache.artifact_cache(description) as entry:
                assert entry.read() is None
                entry.store(b"new private artifact")
    assert sentinel.read_bytes() == b"untouched"


def test_unsafe_permissions_and_default_fallback(cache_dir, monkeypatch, tmp_path):
    cache_dir.mkdir(mode=0o700)
    cache_dir.chmod(0o777)
    with pytest.raises(RuntimeError, match="unsafe"):
        with cache.artifact_cache({}):
            pass
    monkeypatch.delenv("TENSORCX_CACHE_DIR")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    default = tmp_path / "tensorcx/compilation-v1"
    default.mkdir(parents=True, mode=0o700)
    default.chmod(0o777)
    with cache.artifact_cache({}) as entry:
        assert entry.root is None


def test_disabled_cache_and_invalid_override(cache_dir, monkeypatch):
    with cache.artifact_cache({}, enabled=False) as entry:
        entry.store(b"ignored")
    assert not cache_dir.exists()
    monkeypatch.setenv("TENSORCX_DISABLE_CACHE", "1")
    with cache.artifact_cache({}) as entry:
        assert entry.root is None
    assert not cache_dir.exists()
    monkeypatch.setenv("TENSORCX_DISABLE_CACHE", "0")
    monkeypatch.setenv("TENSORCX_CACHE_DIR", "")
    with pytest.raises(RuntimeError, match="must name"):
        with cache.artifact_cache({}):
            pass
    with pytest.raises(TypeError, match="cache must be bool"):
        cached_add.compile(cache=1)


def test_binary_fingerprint_and_options_invalidate(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_dependencies", lambda path: [])
    tool = tmp_path / "compiler"
    tool.write_bytes(b"first compiler")
    arguments = dict(tools={"compiler": tool}, options=["O2"], abi="v1")
    first = cache.descriptor("cpu", "source", **arguments)
    assert _key(first) != _key(cache.descriptor("cpu", "other", **arguments))
    assert _key(first) != _key(cache.descriptor("cuda", "source", **arguments))
    tool.write_bytes(b"other compiler")
    assert _key(first) != _key(cache.descriptor("cpu", "source", **arguments))
    assert _key(first) != _key({**first, "options": ["O3"]})
    assert _key(first) != _key({**first, "abi": "v2"})



def test_shared_library_and_environment_changes_invalidate(tmp_path, monkeypatch):
    tool, library = tmp_path / "compiler", tmp_path / "implementation.so"
    tool.write_bytes(b"unchanged executable")
    library.write_bytes(b"first implementation")
    monkeypatch.setattr(cache, "_dependencies", lambda path: [library])
    arguments = dict(tools={"compiler": tool}, options=[], abi="v1")
    first = cache.descriptor("cpu", "source", **arguments)
    library.write_bytes(b"other implementation")
    second = cache.descriptor("cpu", "source", **arguments)
    assert _key(first) != _key(second)
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path))
    assert _key(second) != _key(cache.descriptor("cpu", "source", **arguments))



def test_link_input_changes_invalidate(tmp_path, monkeypatch):
    linker = tmp_path / "linker"
    linker.write_bytes(b"first linker")
    monkeypatch.setattr(cache, "identity", lambda tools: {})
    arguments = dict(tools={}, options=[], abi="v1", extra_files=lambda: [linker])
    first = cache.descriptor("cpu", "source", **arguments)
    linker.write_bytes(b"other linker")
    assert _key(first) != _key(cache.descriptor("cpu", "source", **arguments))

def test_unknown_compiler_identity_never_reuses(cache_dir, monkeypatch):
    def unavailable(tools):
        raise OSError("dependency query unavailable")
    monkeypatch.setattr(cache, "identity", unavailable)
    arguments = dict(tools={}, options=[], abi="v1")
    with pytest.raises(RuntimeError, match="cannot identify"):
        cache.descriptor("cpu", "source", **arguments)
    monkeypatch.delenv("TENSORCX_CACHE_DIR")
    description = cache.descriptor("cpu", "source", **arguments)
    assert description is None
    with cache.artifact_cache(description) as entry:
        assert entry.root is None

def test_cross_process_single_publication(cache_dir):
    code = '''
from tensorcx._compiler.cache import artifact_cache
from pathlib import Path
import os,time
with artifact_cache({"source":"concurrent"}) as entry:
    if entry.read() is None:
        with (Path(os.environ["TENSORCX_CACHE_DIR"]).parent/"build-count").open("a") as f: f.write("one\\n")
        time.sleep(.1)
        entry.store(b"complete artifact")
    assert entry.read() == b"complete artifact"
'''
    workers = [subprocess.Popen([sys.executable, "-I", "-c", code], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True) for _ in range(4)]
    for worker in workers:
        stdout, stderr = worker.communicate(timeout=30)
        assert worker.returncode == 0, stdout + stderr
    assert (cache_dir.parent / "build-count").read_text().splitlines() == ["one"]


def test_lock_timeout_is_bounded(cache_dir, monkeypatch):
    import fcntl
    cache_dir.mkdir(mode=0o700)
    lock = cache_dir / (_key({}) + ".lock")
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    monkeypatch.setattr(cache, "LOCK_TIMEOUT", .02)
    try:
        with pytest.raises(RuntimeError, match="unsafe"):
            with cache.artifact_cache({}):
                pytest.fail("busy lock accepted")
    finally:
        os.close(fd)


def _require_target(target):
    if target == "metal":
        if not cx.is_available("metal"):
            pytest.skip("Metal unavailable")
        return
    if target == "cpu" and not cx._core._cpu_kernel_supported():
        pytest.skip("MLIR CPU target unavailable")
    if target == "cuda":
        if not cx.is_available("cuda"):
            pytest.skip("CUDA unavailable")
        cx._core._cuda_kernel_support()
    try:
        cpu.toolchain()
    except RuntimeError as error:
        if os.environ.get("TENSORCX_REQUIRE_MLIR"):
            pytest.fail(str(error))
        pytest.skip(str(error))


def _compile(target, **options):
    return cached_add.compile(target=target, compiler="auto" if target == "metal" else "mlir", **options)


def _launch(artifact):
    a = cx.ones((31,), device=artifact.target)
    b = cx.ones((31,), device=artifact.target)
    np.testing.assert_array_equal(artifact.launch(a, b, b, 31).numpy(), 2)


@pytest.mark.parametrize("target", ["cpu", "cuda", "metal"])
def test_real_cold_warm_rebuild_and_disabled(target, cache_dir):
    _require_target(target)
    first = _compile(target)
    assert first.cache_hit is False
    _launch(first)
    warm = _compile(target)
    assert warm.cache_hit is True
    _launch(warm)
    artifact = next(cache_dir.glob("*.bin"))
    artifact.write_bytes(b"incomplete")
    rebuilt = _compile(target)
    assert rebuilt.cache_hit is False
    _launch(rebuilt)
    assert _compile(target).cache_hit is True
    assert _compile(target, cache=False).cache_hit is False
    _launch(first)  # A loaded module survives replacement of its cache entry.


@pytest.mark.parametrize("target", ["cpu", "cuda", "metal"])
def test_cache_hit_skips_compiler_commands(target, cache_dir, monkeypatch):
    _require_target(target)
    _compile(target)
    if target == "metal":
        monkeypatch.setattr(cx.experimental, "_run_metal_tool", lambda *a: pytest.fail("compiler ran on hit"))
    else:
        original = cpu._run
        def only_version(command):
            assert (command[-1] == "--version" or command[-1].startswith("-print-")), "compiler ran on hit"
            return original(command)
        monkeypatch.setattr(cpu, "_run", only_version)
        monkeypatch.setattr(cuda, "_run", only_version)
    artifact = _compile(target)
    assert artifact.cache_hit is True
    _launch(artifact)


@pytest.mark.parametrize("target", ["cpu", "cuda", "metal"])
def test_real_cache_reused_after_process_restart(target, cache_dir, tmp_path):
    _require_target(target)
    source = tmp_path / "kernel.py"
    source.write_text('''import tensorcx as cx
@cx.experimental.kernel
def restarted(a,b,out,n):
    i=cx.experimental.program_id(0)*cx.experimental.block_size()+cx.experimental.thread_id()
    if i<n:
        out[i]=a[i]+b[i]
''')
    code = '''import importlib.util,sys,json,tensorcx as cx
spec=importlib.util.spec_from_file_location("kernel",sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
k=m.restarted.compile(target=sys.argv[2],compiler="auto" if sys.argv[2]=="metal" else "mlir")
a=cx.ones((7,),device=sys.argv[2]);r=k.launch(a,a,a,7)
assert r.numpy().tolist()==[2.]*7
print(json.dumps({"hit":k.cache_hit}))
'''
    for expected in (False, True):
        result = subprocess.run([sys.executable, "-I", "-c", code, str(source), target],
                                capture_output=True, text=True, timeout=60, check=True)
        assert json.loads(result.stdout)["hit"] is expected


def test_warm_cache_does_not_bypass_toolchain_or_signature(cache_dir, monkeypatch, tmp_path):
    _require_target("cpu")
    _compile("cpu")
    monkeypatch.setenv("TENSORCX_LLVM_BIN", str(tmp_path / "absent"))
    with pytest.raises(RuntimeError, match="requires executable"):
        _compile("cpu")
    bad = replace(cached_add.parse_ir(), body=())
    with pytest.raises(cx.experimental.KernelCompileError):
        cpu.compile_kernel(bad)


@pytest.mark.parametrize("target", ["cpu", "cuda", "metal"])
def test_concurrent_compiles_load_independent_valid_modules(target, cache_dir):
    _require_target(target)
    with ThreadPoolExecutor(max_workers=4) as pool:
        artifacts = list(pool.map(lambda _: _compile(target), range(4)))
    assert sum(not k.cache_hit for k in artifacts) == 1
    for artifact in artifacts:
        _launch(artifact)
