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

### Compile and launch profiling

From a Git source checkout with an installed runtime and the appropriate external
toolchain (LLVM 21.1.8 for CPU/CUDA, Xcode Metal tools for Metal):

```bash
mkdir -p build/compilation-profile
uv run --no-sync python benchmarks/bench_compilation_cache.py --target cpu --profile --output build/compilation-profile/cpu.json
uv run --no-sync python benchmarks/bench_compilation_cache.py --target cuda --profile --output build/compilation-profile/cuda.json
```

Select `metal` on a supported macOS host; no Metal timings were measured in the
local Linux run below. The tool accepts an editable or ordinary wheel runtime;
profiles normalize the loaded package's Python files into repository-style
labels. It does not alter the cache or runtime implementation.

The version-2 JSON report separates:

- **Initial miss:** one first cache-enabled public compile in this process with
  empty private storage. Includes first identity hashing and lazy adapter imports;
  this is a single observation, not a median or a cold-disk measurement.
- **Uncached versus warm:** alternating-order calls after that initial miss,
  with warmups excluded. Each measured call includes normal toolchain checks;
  CPU/CUDA also load a new native module or perform Driver JIT. Metal compilation
  produces a metallib; its native pipeline is created at launch.
- **Restart:** one hit in each fresh child process using the same persistent
  artifact. Interpreter startup, `tensorcx` import, device initialization and
  priming the parsed IR are outside the clock. First identity hashing and lazy
  compiler-adapter imports inside the public compile call remain included.
- **Resident launch:** synchronous public launch latency at 1K, 16K, 256K, 1M
  and 16M elements. Inputs and artifact exist before timing. A fresh NaN output
  is prepared before each clock to detect partial/no-op in-place launches; any
  output allocation/loading done inside public launch remains timed. This is
  not device-only kernel time.
- **Optional profiles:** separate `cProfile` calls for uncached, miss, warm and
  restarted hit. Function call counts, self/cumulative times and `subprocess.run`
  cumulative time help locate validation, hashing, dependency/linker queries
  and cache I/O. Cumulative times nest and **must not be added**. Subprocess time
  includes process startup/waiting and child work; it is not compiler CPU time.
  Python attribution does not expose native kernel/JIT internals. The profiler
  adds overhead, so these calls never enter the benchmark samples; see
  [Python's profiler guidance](https://docs.python.org/3.12/library/profile.html).

Every compile checks the expected cache-hit state and executes a nontrivial
seeded float32 expression against CPU outside the timed/profiled region. Every
launch checks CPU parity too. Result release, tensor preparation and validation
are outside the clocks. Child extension and script hashes must match the parent;
errors stop the run before a new report is written. A dedicated temporary cache
is removed and the caller's cache environment restored. Existing user artifacts
are untouched. OS page caches and driver caches are not flushed.

Use `--repeats`, `--warmup`, `--restart-repeats` and `--sizes` to control sampling
(defaults: 5, 1, 3 and the five sizes above). `--profile` is opt-in; `--output`
otherwise defaults to stdout. Reports contain samples/median/p10/p90, source
revision/dirty/diff and script/extension hashes, versions, and host observations.
Raw cProfile dumps, command arguments, absolute source/cache paths and GPU/PID
identifiers are not emitted. Reports still require publication review. Old
single-median JSON consumers must migrate to schema 2. `bench_cuda.py` retains
its independent uncached compilation measurements.

### Local measurement result — 2026-10-17

Two sequential runs per target on Linux x86_64, Intel i7-6700K, GTX 980 Ti
(sm_52), LLVM 21.1.8, CUDA 12.4 / driver 580.178.04, Python 3.12. All used
`--repeats 7 --restart-repeats 3 --profile` with one warmup and all five sizes.
Source reports retain measured working-tree/script/extension identities; the
benchmark file hash is identical across all four runs. These are reviewed public
JSON exports, not raw profiler dumps.

Milliseconds; uncached/warm are seven-sample medians, restart is a three-process
median, initial miss is a single call:

| Run | Uncached | Warm hit | Restart hit | Initial miss |
| --- | ---: | ---: | ---: | ---: |
| [cpu-1](../benchmarks/results/compilation-cache/cpu-1.json) | 202.68 | 226.94 | 1020.77 | 1250.83 |
| [cpu-2](../benchmarks/results/compilation-cache/cpu-2.json) | 203.80 | 227.02 | 1013.96 | 1189.24 |
| [cuda-1](../benchmarks/results/compilation-cache/cuda-1.json) | 124.04 | 86.02 | 838.14 | 921.73 |
| [cuda-2](../benchmarks/results/compilation-cache/cuda-2.json) | 117.70 | 84.76 | 834.52 | 883.71 |

CPU warm calls were slower than uncached calls for this tiny expression. CUDA
warm calls were faster, while first hits after a restart remained expensive on
both targets. These are shared-host observations: two other CUDA compute
processes were present in the before/after snapshots, alongside desktop GPU
activity and CPU load. No exclusive-use or clean optimization comparison is
claimed; snapshots do not capture all activity throughout a run.
No Metal measurement or cross-device speedup is claimed.

The separate profiles locate repeated toolchain/dependency/linker subprocess
queries in warm calls, and substantial first-process file hashing in restart
calls. CPU requires 15 `subprocess.run` calls per warm profile (including linker
and startup-object queries); CUDA requires six. These counts explain why a
small compile can cost less than validating/reloading a cached artifact. The
function tables include both parent and nested calls; inspect them without
summing cumulative values. Cache checks, security or runtime behavior were not
relaxed. Reducing validated identity-query overhead is follow-up optimization
work, alongside broader runtime tracing/MLIR/fusion.

The focused benchmark/cache regression set passed 44 tests with five expected
Metal skips on this Linux host. It covers clocks and cleanup, mismatched child
identity, installed-wheel profile labels, poisoned in-place outputs, expected
cache state and real CPU/CUDA reports. The CPU MLIR CI job now runs the new
benchmark tests with actual LLVM; Metal execution remains a macOS check.

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
