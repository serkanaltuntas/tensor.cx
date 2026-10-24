# NVIDIA OpenCL and PoCL CPU execution experiments

These two standalone Linux experiments complete the remaining
[Nightblade validation experiments](../../docs/DEVICE_BACKLOG.md#nightblade-validation-experiments-planned).
They use the OpenCL C API through the external ICD loader. They do not register
an OpenCL backend, extend `cx.devices()`, add a PyTorch device, or change the
production CPU/Metal/CUDA/Vulkan paths.

`opencl_probe.cpp` requires exactly one explicitly named device from the selected
implementation: an NVIDIA GPU (vendor ID `0x10de`), or a CPU from the Portable
Computing Language platform. Availability, compiler support and device/kernel
work-group limits are checked. The fixture builds as OpenCL C 1.2 without
relaxed math. There is no automatic device or implementation substitution.

`probe_opencl.py` builds the C++20 host, runs real allocation/upload/dispatch/
readback, and verifies every storage bit against tensor.cx CPU primitives,
guarded `Kernel.reference`, and NumPy. Each target runs **192 cases across two
complete context lifecycles**: sizes 0/1/63/64/65/255/256/257/1025; zero, half
and full active prefixes; add/multiply; separate output and aliasing the first
input. Rounded 64-work-item groups include an actual zero-bound launch.
Inactive suffixes, launch padding and the second input must remain unchanged;
the first input must remain unchanged when it is not the output.

All owned contexts, queues, programs, kernels and buffers are released through
RAII; all 490 releases must succeed. Completion is established before freeing
buffers, including on operation failures. If queue completion cannot be proved,
the process exits without unwinding possibly active resources. Failed builds,
wrong devices, incomplete reports, oracle differences and release errors fail
the command rather than producing a passing skip.

## Run

From the source checkout, use the project's uv environment with an installed
tensor.cx CPU path and NumPy, a C++20 compiler (default `g++-13`), OpenCL headers,
linkable `libOpenCL`, and the **ocl-icd** loader's explicit vendor-file support.
Supply a healthy implementation and the exact name enumerated by `clinfo`.
The drivers and SDK headers are external dependencies, not bundled in wheels.

```bash
uv run --no-sync python experiments/opencl/probe_opencl.py \
  --target nvidia-gpu --device-name 'NVIDIA GeForce GTX 980 Ti' \
  --icd /etc/OpenCL/vendors/nvidia.icd

uv run --no-sync python experiments/opencl/probe_opencl.py \
  --target pocl-cpu \
  --device-name 'cpu-haswell-Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz' \
  --icd /etc/OpenCL/vendors/pocl.icd
```

For a relocatable PoCL package extracted locally, point `--icd` at that package's
vendor file and add `--library-dir /path/to/extracted/usr/lib/x86_64-linux-gnu`.
The local validation used Ubuntu 26.04 packages extracted under ignored
`build/opencl-toolchain/root/`, including PoCL 6.0, LLVM/Clang 18.1.8 and hwloc
2.13.0. It did not install a system ICD or change NVIDIA's driver. The full
relocated package must include PoCL's private libraries, headers and builtin
bitcode; a vendor manifest alone is insufficient. This is a host-specific
validation setup, not a portable distribution recipe.

The controller clears inherited driver/compiler overrides and supplies only the
selected ICD. PoCL explicitly uses its `pthread` CPU device, at most four worker
threads and a per-run build cache. `--library-dir` may be repeated. `--output`
selects a new evidence directory; it must not already exist. By default outputs
are under ignored `build/opencl-probe/`. Raw readbacks/build logs remain local;
`result.json` contains source/executable hashes, platform/device/compiler
capabilities, counts and sanitized diagnostics. Inspect it before publication.

Pure CPU oracle/failure regression tests do not need OpenCL installed:

```bash
uv run --no-sync pytest tests/python/test_opencl_probe.py -q
```

## Reviewed acceptance — 2026-10-24

- **NVIDIA GPU:** GTX 980 Ti, driver 580.178.04; platform OpenCL 3.0 CUDA
  13.0.100, device OpenCL C 1.2, maximum group size 1024.
- **PoCL CPU:** i7-6700K `pthread` device, PoCL 6.0+debian with LLVM 18.1.8;
  OpenCL 3.0, OpenCL C 1.2, maximum group size 4096.
- Each target passed all 192 cases, two context cycles and 490/490 handle
  releases with zero release errors. All readbacks matched the three CPU oracles.

These establish two bounded execution experiments on the existing physical
hardware. They do not establish an integrated runtime backend, full operation
coverage, broad numerical/performance claims, additional hardware support,
GPU CI, or multi-device execution. The dyadic normal-valued fixtures do not
establish full float32 special-value/subnormal conformance. Raw local evidence
is deliberately ignored by Git.

Sources: [Khronos OpenCL API](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_API.html),
[NVIDIA OpenCL](https://developer.nvidia.com/opencl),
[PoCL ICD and runtime configuration](https://portablecl.org/docs/html/using.html).
