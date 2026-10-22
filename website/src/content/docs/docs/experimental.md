---
title: Experimental kernels
description: The custom kernel DSL and optional MLIR paths, with their supported subsets and toolchain boundaries.
---

Custom kernels live under `cx.experimental`. Ordinary tensor operations do not
require a compiler SDK at runtime. Explicit custom-kernel compilation does.

| Path | What is implemented | Required environment |
| --- | --- | --- |
| Metal / MSL | Restricted Python kernel DSL, float32 elementwise and bounded rowwise reductions | Apple Metal compiler tools and an available Metal device |
| CPU / MLIR | Guarded float32 arithmetic, division, sqrt/abs/exp/log/rsqrt/tanh/sigmoid/SiLU, minimum/maximum, comparisons and select; bounded local/nested expressions | Validated on Linux x86_64 with LLVM/MLIR 21.1.8 |
| CUDA / MLIR | Guarded float32 add/subtract/multiply/divide, sqrt/abs/exp/log/rsqrt/tanh/sigmoid/SiLU, minimum/maximum, comparisons and select; bounded local/nested expressions | Validated CPU toolchain plus CUDA toolkit 12.4, CUDA Driver API 13.0, and sm_52 hardware |
| Vulkan / MLIR → SPIR-V | Guarded float32 arithmetic, the same math intrinsics, minimum/maximum, comparisons and select; bounded expressions and private outputs | LLVM 21.1.8 mlir-opt/mlir-translate, glslang/SPIR-V Tools and an available Vulkan compute GPU; one GTX 980 Ti validated |

## Explicit compilation

The MLIR path is selected using `kernel.compile(target="cpu", compiler="mlir")`
or `kernel.compile(target="cuda", compiler="mlir")` or
`kernel.compile(target="vulkan", compiler="mlir")` on a supported experimental
kernel. External LLVM tools must be discoverable through the documented
toolchain setup. A missing toolchain or unsupported kernel fails explicitly.

Float32 `/` works in guarded CPU/CUDA bodies, including local and nested
expressions. Both operands must be float32; use `2.0` instead of `2`. The
tests compare CPU/CUDA division with the CPU reference for signed zero, NaN,
infinity and subnormals. Direct Metal kernels also accept float32 division, with
finite-input parity at the existing tolerances. See the
[division contract and example](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/MLIR_DIVISION.md)
for numerical limits and unchanged launch/alias rules.

Guarded MLIR kernels also support six float32 comparisons and
`cx.experimental.select(condition, on_true, on_false)`, including boolean locals
and nested values. Both values are validated and may be evaluated; selection
does not replace a memory-safety guard. The output remains float32. See the
[comparison/select contract and example](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/MLIR_SELECT.md)
for NaN behavior, complexity limits and Metal validation scope.

Guarded kernels accept `cx.experimental.sqrt(value)` for one float32 operand,
including locals and nested/selected expressions. CPU/CUDA numerical tests
cover signed zero, negative inputs, NaN, infinity and subnormals; CUDA produces
strict `sqrt.rn.f32` without an external math library. Direct MSL uses the
qualified Metal builtin and its existing finite-input tolerance. CUDA retains
its two-input/one-output signature. See the
[square-root contract and runnable example](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/MLIR_SQRT.md).

`cx.experimental.abs(value)` also accepts one float32 operand, including
locals and selected/nested values. CPU/CUDA tests verify non-NaN sign-bit
behavior (`-0` becomes `+0`, subnormal magnitudes are preserved) and NaN masks.
CUDA emits `abs.f32` without an external math library; direct Metal emits
qualified `fabs` with finite-input tolerance. The CUDA two-input signature
remains unchanged. See the
[absolute-value contract and example](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/MLIR_ABS.md).

`cx.experimental.minimum(lhs, rhs)` and `cx.experimental.maximum(lhs, rhs)`
accept two float32 values. Either NaN input yields NaN; mixed zero signs yield
`-0` for minimum and `+0` for maximum, regardless of order. Standalone operations
preserve subnormal magnitudes. CUDA lowering supports the validated sm_52
target; Metal uses bit-order helpers and disables fast-math for kernels that
contain these operations. CPU/CUDA and standalone Metal numerical checks compare
non-NaN bits; composed Metal expressions keep their finite-input tolerance. See
the [minimum/maximum contract and example](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/MLIR_MINMAX.md).

`cx.experimental.exp(value)` accepts one float32 value. CPU explicitly links
system libm; CUDA uses non-FTZ base-2 exponential lowering without libdevice.
This is bounded-error arithmetic: CPU uses `rtol=1e-6`, CUDA `rtol=1e-5`, with
only one/two smallest-subnormal units of absolute allowance. Direct Metal exp
uses strict compilation and `rtol=1e-6` for normal-output inputs in `[-80, 80]`;
gradual underflow and the near-overflow boundary are outside Metal acceptance.
NaN/infinity and zero behavior have explicit checks. See the
[exp contract and runnable example](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/MLIR_EXP.md).

`cx.experimental.log(value)` accepts one float32 value. Both zeros yield
negative infinity; negative nonzero values yield NaN; `log(1)` is positive zero.
CPU/CUDA acceptance includes positive subnormal inputs. Direct Metal uses
strict compilation and accepts normal inputs plus zero/domain/special cases;
nonzero subnormal arguments are outside its numerical promise. CPU/Metal use
`rtol=1e-6` with zero absolute allowance; CUDA uses `rtol=1e-6, atol=2e-7`,
covering base-2 approximation near one. CPU shares explicit libm linkage with
exp; CUDA emits non-FTZ logarithm lowering without libdevice. CPU kernel names
`expf`, `logf` and `tanhf` are reserved to prevent libm symbol collisions. See the
[log contract and runnable example](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/MLIR_LOG.md).

`cx.experimental.rsqrt(value)` computes float32 `1 / sqrt(value)`. Positive and
negative zero produce infinities with the corresponding sign, positive infinity
produces positive zero, and negative nonzero inputs produce NaN. Standalone
finite results use `rtol=1e-6, atol=0` on each target. CPU/CUDA acceptance includes
positive subnormal arguments; direct Metal uses strict compilation and excludes
nonzero subnormal arguments. CPU uses strict sqrt/division and CUDA uses a
non-FTZ reciprocal-root instruction without libdevice. See the
[rsqrt contract and runnable example](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/MLIR_RSQRT.md).

`cx.experimental.tanh(value)` accepts one float32 value. It preserves signed
zero, maps infinities to the corresponding `-1`/`+1`, propagates NaNs, and keeps
finite results in `[-1, 1]`. Standalone results use `rtol=1e-6, atol=0` on each
target. CPU/CUDA include subnormal inputs/results; direct Metal uses strict
compilation and excludes nonzero subnormal arguments. CPU uses explicit libm
linkage; CUDA uses a rational/exp implementation compatible with the validated
sm_52 target, without libdevice or a newer-GPU tanh instruction. See the
[tanh contract and example](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/MLIR_TANH.md).


`cx.experimental.sigmoid(value)` is a stable float32 logistic function. Both
zeros map to `0.5`, infinities map to `0`/`1`, NaNs propagate and results stay in
`[0, 1]`. Its `exp(-abs(x))` lowering avoids positive exponential overflow and
keeps small negative-tail results. CPU/CUDA acceptance includes gradual
underflow: CPU uses `rtol=1e-6` plus one smallest-subnormal unit, CUDA `rtol=1e-5`
plus two units. Direct Metal uses strict compilation; accuracy is accepted for
`x >= -80`, `x <= -128` and NaNs, with the intervening underflow range excluded.
CPU reuses the existing exp/libm policy; CUDA uses non-FTZ exp2 without libdevice.
See the [sigmoid contract and example](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/MLIR_SIGMOID.md).


`cx.experimental.silu(value)` computes float32 SiLU, preserving signed zeros,
mapping positive infinity to itself and negative infinity to negative zero.
It uses half exponentials to keep representable negative-tail results after
sigmoid itself would round to zero. CPU/CUDA include gradual underflow with
`rtol=1e-6`/`1e-5` and one/two smallest-subnormal units of absolute allowance.
Direct Metal uses strict compilation: normal-output ranges `[-80, -2*tiny]`
and `[2*tiny, +inf]`, signed zeros, `x <= -128` and NaNs are accepted, where
`tiny` is the smallest positive normal float32. Nonzero subnormal inputs/results
and the negative interval `(-128, -80)` are outside its accuracy promise.
Ordinary tensor `cx.silu` is an existing separate API. See the
[SiLU contract and example](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/MLIR_SILU.md).

Compiled CPU/CUDA/Metal/Vulkan code is cached in private local storage by default,
including across process restarts. Inspect `compiled.cache_hit` for reuse, or
use `kernel.compile(..., cache=False)` to bypass storage.
`TENSORCX_DISABLE_CACHE=1` disables persistence globally. Toolchain and target
checks still run on hits; this does not make kernels standalone packages.

## Vulkan kernels

```python
compiled = kernel.compile(target="vulkan", compiler="mlir")
result = compiled.launch(a, b, out, n, block_size=256)
```

Here `kernel` is a supported guarded kernel; `a`, `b` and the output template
are matching float32 Vulkan tensors, and `n` equals the requested thread count.
Zero, one or two inputs are supported. The compiler accepts up to 32 locals,
expression depth 16 and 64 operations, followed by one final store. Arbitrary
indices, output reads, loops, broadcasting and additional dtypes are rejected.
Results own private GPU storage, retain unwritten suffixes and preserve inputs
and the output template even when they alias. Operations remain synchronous.

The body is generated as typed MLIR and linked with the same integer RNE math
helpers used by ordinary Vulkan operations, preserving gradual underflow.
`target="vulkan:0"` is the first-device alias; indexed targets bind to their
actual device. Missing tools or devices produce errors without CPU substitution.
Tools are optional for ordinary tensor work. Set `TENSORCX_LLVM_BIN` for LLVM
21.1.8 and `TENSORCX_VULKAN_BIN` for shader tools outside `PATH`.

Cache hits revalidate SPIR-V and create a fresh native module/JIT; they do not
reuse loaded device handles. Use `benchmarks/bench_compilation_cache.py --target
vulkan` for paired compilation, launch and process-restart measurements. See the
[Vulkan compiler contract](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/VULKAN_COMPILER.md)
for numerical bounds, ownership and validation limits.
See the [cache controls and trust contract](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/COMPILATION_CACHE.md).

The kernel parser reads Python source. Define kernels in source files, not an
interactive command string. Supported indexing, guards, argument layout, and
launch geometry are part of the contract, so use the complete examples in the
corresponding integration guide:

- [Kernel DSL and Metal execution](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/KERNEL_DSL.md)
- [MLIR CPU installation and runnable example](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/MLIR_RUNTIME_INTEGRATION_DECISION.md)
- [MLIR CUDA installation and runnable example](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/MLIR_CUDA_INTEGRATION_DECISION.md)
- [CUDA local-expression contract](https://github.com/serkanaltuntas/tensor.cx/blob/main/docs/MLIR_CUDA_EXPRESSIONS_DECISION.md)

## Limits and trust

This is not general Python compilation or a full graph compiler. Ordinary
tensor expressions are not automatically fused. Broader generated operations,
targets, and reduction support require separate validation.
Ordinary tensor `cx.sqrt`/`cx.abs` and the explicit experimental kernel
intrinsics are implemented within their documented subsets. See [remaining compiler work](/docs/status/#remaining-work).

Generated native libraries and PTX are executable code, not a sandbox. Compile
and load only artifacts you trust; see the repository
[security policy](https://github.com/serkanaltuntas/tensor.cx/blob/main/SECURITY.md).
