---
title: Experimental kernels
description: The custom kernel DSL and optional MLIR paths, with their supported subsets and toolchain boundaries.
---

Custom kernels live under `cx.experimental`. Ordinary tensor operations do not
require a compiler SDK at runtime. Explicit custom-kernel compilation does.

| Path | What is implemented | Required environment |
| --- | --- | --- |
| Metal / MSL | Restricted Python kernel DSL, float32 elementwise and bounded rowwise reductions | Apple Metal compiler tools and an available Metal device |
| CPU / MLIR | Guarded float32 elementwise compile and launch | Validated on Linux x86_64 with LLVM/MLIR 21.1.8 |
| CUDA / MLIR | Guarded float32 add/subtract/multiply and bounded local/nested expressions | Validated CPU toolchain plus CUDA toolkit 12.4, CUDA Driver API 13.0, and sm_52 hardware |

## Explicit compilation

The MLIR path is selected using `kernel.compile(target="cpu", compiler="mlir")`
or `kernel.compile(target="cuda", compiler="mlir")` on a supported experimental
kernel. External LLVM tools must be discoverable through the documented
toolchain setup. A missing toolchain or unsupported kernel fails explicitly.

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

Generated native libraries and PTX are executable code, not a sandbox. Compile
and load only artifacts you trust; see the repository
[security policy](https://github.com/serkanaltuntas/tensor.cx/blob/main/SECURITY.md).
