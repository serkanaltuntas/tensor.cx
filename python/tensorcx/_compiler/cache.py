"""Private local compiled-artifact storage. Loaded device modules are not cached.

Checksums detect accidental damage, not malicious changes by the same user.
Never use this cache as an interchange format for downloaded native code.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from functools import lru_cache
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import stat
import sys
import time
import uuid

MAGIC = b"TENSORCX-CACHE-1\n"
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
LOCK_TIMEOUT = 30


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=8192)
def _file_digest(path, stamp):
    # The stamp includes ctime: ordinary replacement/editing cannot reuse a hash.
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _fingerprint(path):
    path = Path(path).resolve(strict=True)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError("compiler identity requires regular files")
    return _file_digest(str(path), (info.st_dev, info.st_ino, info.st_size,
                                   info.st_mtime_ns, info.st_ctime_ns))


def _dependencies(path):
    if sys.platform == "linux":
        with Path(path).open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                raise RuntimeError("persistent cache requires native compiler executables")
        result = subprocess.run(["ldd", str(path)], capture_output=True, text=True,
                                timeout=60, check=False)
        output = result.stdout + result.stderr
        if "not found" in output:
            raise RuntimeError("compiler dependency could not be resolved")
        if result.returncode:
            if "not a dynamic executable" in output or "statically linked" in output:
                return []
            raise RuntimeError("compiler dependency query failed")
        paths = re.findall(r"(?:=>\s+|^\s*)(/[^\n]+?)\s+\(0x[0-9a-fA-F]+\)", output, re.M)
        if not paths:
            raise RuntimeError("compiler dependency query returned no libraries")
        return [Path(item) for item in paths]
    if sys.platform == "darwin":
        # Xcode's Metal implementation lives in its toolchain libraries/plugins.
        # System dylibs are tied to the macOS kernel/build identity below.
        root = Path(path).resolve(strict=True).parent.parent
        return sorted({file for folder in ("bin", "lib", "libexec")
                       for file in (root / folder).rglob("*") if file.is_file()})
    raise RuntimeError("compiler dependency identity is unsupported on this host")


def identity(tools):
    """Fingerprint compiler implementations, including resolved shared libraries."""
    result, dependencies, hashes = {}, {}, {}
    for name, path in sorted(tools.items()):
        resolved = Path(path).resolve(strict=True)
        group = resolved.parent if sys.platform == "darwin" else resolved
        if group not in dependencies:
            dependencies[group] = _dependencies(resolved)
        files = [resolved, *dependencies[group]]
        for file in files:
            if file not in hashes:
                hashes[file] = _fingerprint(file)
        result[name] = {"binary": hashes[resolved],
                        "dependencies": sorted({hashes[file] for file in dependencies[group]})}
    return result


def cache_enabled(requested=True):
    return (requested and os.name == "posix"
            and os.environ.get("TENSORCX_DISABLE_CACHE", "0").lower() not in ("1", "true", "on"))


def descriptor(target, source, *, tools, options, abi, extra_files=None):
    try:
        fingerprints = identity(tools)
        inputs = sorted({_fingerprint(path) for path in extra_files()}) if extra_files else []
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        if os.environ.get("TENSORCX_CACHE_DIR") is not None:
            raise RuntimeError("explicit compilation cache cannot identify compiler dependencies") from error
        return None  # Unknown implementation identity must never reuse an artifact.
    variables = ("PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT", "LIBRARY_PATH",
                 "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "COMPILER_PATH",
                 "GCC_EXEC_PREFIX", "CCC_OVERRIDE_OPTIONS", "DEVELOPER_DIR", "SDKROOT",
                 "MACOSX_DEPLOYMENT_TARGET", "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH",
                 "DYLD_INSERT_LIBRARIES")
    environment = digest(json.dumps({key: os.environ.get(key) for key in variables},
                                    sort_keys=True).encode())
    return {"schema": 1, "target": target, "source": source, "tools": fingerprints,
            "options": options, "abi": abi, "inputs": inputs, "system": platform.system(),
            "os_build": platform.version(), "environment": environment,
            "machine": platform.machine(), "libc": platform.libc_ver(),
            "byteorder": sys.byteorder}


class _Entry:
    def __init__(self, root=None, key=None, strict=False):
        self.root, self.key, self.strict = root, key, strict

    def read(self):
        if self.root is None:
            return None
        fd = None
        try:
            fd = os.open(self.key + ".bin", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=self.root)
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_mode & 0o077 or info.st_nlink != 1
                    or info.st_size > MAX_ARTIFACT_BYTES + 1024):
                return None
            with os.fdopen(fd, "rb") as stream:
                fd = None
                if stream.readline(64) != MAGIC:
                    return None
                header = json.loads(stream.readline(1024))
                payload = stream.read(MAX_ARTIFACT_BYTES + 1)
            if (header != {"key": self.key, "size": len(payload), "sha256": digest(payload)}
                    or not payload or len(payload) > MAX_ARTIFACT_BYTES):
                return None
            return payload
        except (OSError, ValueError, UnicodeError):
            return None
        finally:
            if fd is not None:
                os.close(fd)

    def store(self, payload):
        if self.root is None or not payload or len(payload) > MAX_ARTIFACT_BYTES:
            return
        name = ".tmp-" + uuid.uuid4().hex
        fd = None
        try:
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=self.root)
            header = {"key": self.key, "size": len(payload), "sha256": digest(payload)}
            with os.fdopen(fd, "wb") as stream:
                fd = None
                stream.write(MAGIC + json.dumps(header, sort_keys=True).encode() + b"\n" + payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.key + ".bin", src_dir_fd=self.root, dst_dir_fd=self.root)
            os.fsync(self.root)
        except OSError as error:
            if self.strict:
                raise RuntimeError("could not write the explicitly configured compilation cache") from error
        finally:
            if fd is not None:
                os.close(fd)
            try:
                os.unlink(name, dir_fd=self.root)
            except OSError:
                pass


def _open_root(path):
    # Parent symlinks (e.g. macOS /var) can be legitimate; the cache leaf cannot.
    path = path.parent.resolve() / path.name
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    root = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(root)
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise RuntimeError("compilation cache must be a private, user-owned directory")
        for parent in path.parents:
            info = parent.stat()
            sticky_root = info.st_uid == 0 and info.st_mode & stat.S_ISVTX
            if info.st_uid not in (0, os.getuid()) or (info.st_mode & 0o022 and not sticky_root):
                raise RuntimeError("compilation cache has an unsafe parent directory")
        return root
    except BaseException:
        os.close(root)
        raise


@contextmanager
def artifact_cache(description, *, enabled=True):
    if description is None or not cache_enabled(enabled):
        yield _Entry()
        return
    explicit = os.environ.get("TENSORCX_CACHE_DIR")
    if explicit == "":
        raise RuntimeError("TENSORCX_CACHE_DIR must name a private cache directory")
    if explicit is not None:
        path = Path(explicit).expanduser().absolute()
    else:
        base = Path.home() / ("Library/Caches" if sys.platform == "darwin" else ".cache")
        xdg = os.environ.get("XDG_CACHE_HOME")
        if xdg and Path(xdg).is_absolute():
            base = Path(xdg)
        path = base / "tensorcx" / "compilation-v1"
    key = digest(json.dumps(description, sort_keys=True, separators=(",", ":")).encode())
    root = lock = None
    locked = False
    import fcntl
    try:
        try:
            root = _open_root(path)
            lock = os.open(key + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
                           0o600, dir_fd=root)
            info = os.fstat(lock)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_mode & 0o077 or info.st_nlink != 1):
                raise RuntimeError("compilation cache lock is not private regular storage")
            deadline = time.monotonic() + LOCK_TIMEOUT
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("compilation cache lock timed out")
                    time.sleep(0.01)
        except (OSError, RuntimeError) as error:
            if explicit is not None:
                raise RuntimeError("explicit compilation cache is unavailable or unsafe") from error
        yield _Entry(root, key, explicit is not None) if locked else _Entry()
    finally:
        if locked:
            fcntl.flock(lock, fcntl.LOCK_UN)
        if lock is not None:
            os.close(lock)
        if root is not None:
            os.close(root)
