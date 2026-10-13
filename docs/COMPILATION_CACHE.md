# Persistent compilation cache

Experimental kernel compilation reuses local CPU shared-library, CUDA PTX and
Metal metallib artifacts across calls and process restarts. This implements the
persistent-cache portion of feature 8; operations, fusion and a production
graph compiler remain separate work.

```python
compiled = kernel.compile(target="cpu", compiler="mlir")
print(compiled.cache_hit)  # False on first compilation, True on reuse
cold = kernel.compile(target="cpu", compiler="mlir", cache=False)
```

`cache` defaults to `True` and must be bool. The controls also apply to CUDA/MLIR
and Metal/MSL. Toolchain/target validation still run on hits; missing or
mismatched LLVM/Xcode tools are not hidden by old artifacts. Reuse skips code
generation/linking, not native validation, CPU loading or CUDA Driver JIT.
Loaded modules, contexts, queues, pointers and tensor data are never persisted.
CPU libraries are copied to private scratch storage before loading, preserving
temporary-file cleanup and existing module lifetime rules.

## Location and controls

- Linux: absolute `$XDG_CACHE_HOME/tensorcx/compilation-v1` or
  `~/.cache/tensorcx/compilation-v1`.
- macOS: `~/Library/Caches/tensorcx/compilation-v1`; absolute XDG override is
  also supported.
- `TENSORCX_CACHE_DIR` selects private storage. Empty, unsafe or unavailable
  explicit directories fail clearly without falling back elsewhere.
- `TENSORCX_DISABLE_CACHE=1` (also `true` or `on`) and `cache=False` bypass
  persistence. Unavailable/unsafe default storage uses uncached compilation.
- Ordinary import/tensor operations create no cache. Non-POSIX hosts bypass it.

Directories are private/user-owned (0700), entry/lock files private (0600).
Unsafe writable ancestors, symlink roots/locks and non-regular storage are
rejected. Symlink/damaged artifacts are misses. Directory descriptors anchor
artifact I/O and atomic replacement. Per-key process locks coordinate cold
builders and release on termination. Waiting is limited to 30 seconds; busy
default storage falls back to uncached work, explicit storage reports an error.

## Compatibility, integrity and trust

Keys cover generated source, compiler implementations, target, stage options,
ABI/entry contract, compiler/loader environment hashes, host architecture, libc
and OS build. Linux fingerprints executables and their resolved shared libraries;
CPU also fingerprints its linker and startup/link inputs. Metal fingerprints the
selected Xcode/Metal toolchain bin/lib/libexec tree and SDK/macOS versions.
Nonstandard Metal toolchains with implementation libraries outside that tree are
not covered; use `cache=False` for those. Native Linux executables are required
for dependency discovery; unknown identity bypasses default caching and rejects
explicit caching. File hashes are memoized within a process using inode, size,
mtime and ctime; dependency resolution and tool validation still run on hits. Targets cannot collide. Metadata stores hashes/byte counts,
not tool paths or environment values. Each entry is one atomic checksummed
envelope with at most 32 MiB of artifact data. Damaged/truncated records are
recompiled; failed compiles are not published.

This is trusted local executable code, not a sandbox or interchange format.
Checksums detect damage; they do not authenticate against changes by the same
user. Do not restore downloaded/untrusted entries or share storage between
users. Kernel code/constants can be present in artifacts. Caches stay outside
Git and are never uploaded automatically. There is no automatic eviction;
crashes can leave private `.tmp-*` files which are never used as entries.
Reclaim storage by removing your dedicated cache while compilers are stopped.

## Verification and measurements

`test_compilation_cache.py` covers identity/options/ABI changes, permissions,
symlinks, corruption/checksums, bounded locking, cross-process publication,
real cold/warm/disabled compilation, restart, concurrency, module lifetime and
mandatory toolchain checks. Existing failure tests disable persistence so their
error paths remain exercised. CPU cache CI uses real LLVM; Metal/CUDA execution
retains the existing hardware gates.

```bash
uv run --no-sync python benchmarks/bench_compilation_cache.py --target cpu
uv run --no-sync python benchmarks/bench_compilation_cache.py --target cuda
```

The benchmark checks numerical parity and separates uncached, initial miss and
warm calls. Timings include toolchain checks/module loading; they are local
compile latency, not kernel throughput or universal speedup. `bench_cuda.py`
retains its uncached compilation measurements.

Local validation, 2026-10-13 (Linux x86_64, LLVM 21.1.8, CUDA 12.4, sm_52):
27 cache tests passed, with four expected Metal skips; the full CUDA build had
4465 passes/166 expected skips and the isolated CPU wheel had 2136 passes/965
expected skips. Fresh CPU/CUDA sdist-to-wheel probes passed. Two sequential
CUDA cache/launch tests each passed memcheck and racecheck with zero memory
errors/hazards. API-error reporting was disabled for the native validator's
intentional out-of-range `cuFuncGetParamInfo` probe; memory/race exit checks
remained enabled. Process restart and threaded compilation passed normally;
recursive/threaded compiler-process instrumentation was excluded after sanitizer
injection interfered with `ldd` subprocesses. Metal execution requires macOS CI.
