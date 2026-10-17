# Native CUDA int32 primitives

Increment: 2026-10-17. Extends native CUDA primitive coverage under the
[product backlog](ROADMAP.md#product-feature-backlog), without changing the
historical Phase 9/10 acceptance records or generated-kernel dtype support.
The public API is defined in [TENSOR_API.md](TENSOR_API.md).

## Contract

| Operation | Int32 behavior on CPU, Metal and CUDA |
| --- | --- |
| `zeros`, `ones` | Native fill, including scalar and empty shapes |
| `+`, `-`, `*` | Same-dtype/device tensors with binary broadcasting; independent contiguous result |
| Integer scalars on either side | One-input primitive; scalar must be in int32 range, without float32 narrowing |
| Unary `-` | Wraps, including `-INT32_MIN == INT32_MIN` |
| `sum` | Int32 result and modulo-2^32 accumulation, without dtype promotion |
| `max` | Signed int32 comparison, including all-negative inputs and `INT32_MIN` |

Arithmetic wraps in two's-complement. Scalar `16777217` and `INT32_MAX` are
passed exactly; they do not round through float32. Fractional, non-finite and
out-of-range scalar/fill values are rejected before allocation or launch, even
for empty shapes. Public arithmetic rejects floating scalars for int32 inputs.
Native `OpDesc` fill supports every representable int32 value; the public
factory API continues to expose zeros/ones, not a new `full` function.

Sum/max accept integer axes, `None`, axis sequences, negative axes and
`keepdims`. `axis=()` returns an independent copy. Empty selected dimensions
sum to zero; max rejects an empty selected dimension even if the output would
also be empty. Empty kept dimensions yield empty outputs. Scalar axes 0/-1,
metadata overflow checks and the absence of an arbitrary rank cap retain the
existing [reduction contract](TENSOR_API.md#reduction-dimensions).

Division, mean, matmul, activations and normalizations still require float32.
There is no implicit promotion or device transfer. Experimental MLIR/Metal DSL
kernels retain their own float32 arithmetic contract; this increment adds no
int32 generated kernels, FP16/BF16, device-index selection or asynchronous API.

## Usage

```python
import tensorcx as cx

device = "cuda"  # Use "cpu" for the mandatory reference path.
x = cx.tensor([2147483647, -2147483648, 16777217], dtype=cx.int32, device=device)
print((x + 1).numpy())  # [-2147483648 -2147483647    16777218]
print(x.sum().numpy())  # 16777216
print(x.max().numpy())  # 2147483647
```

## Implementation and verification

The existing `CudaBackend::execute` and `OpDesc` dispatch validate dtype/scalar
constraints before allocating. Native launch helpers select float32 or int32;
the float32 kernels retain explicit rounding and their existing reduction
paths. Int32 arithmetic uses corresponding unsigned 32-bit loads/stores and
accumulation to avoid signed-overflow undefined behavior. Max uses signed
values. Both single-axis and mapped multi-axis reductions use one lane per
output slice; no performance gain is claimed.

Inputs remain in their original device buffers. Broadcasting/reductions upload
only O(rank) indexing metadata; there is no CPU tensor fallback or expanded
operand. Output publication occurs only after a successful synchronous launch.
Primary-context ownership, restoration of a foreign current context, RAII
allocation cleanup and the compiled-kernel ABI are unchanged.

`tests/python/test_int32_primitives.py` checks exact NumPy/CPU parity, boundary
and random bit patterns, scalar precision/order, signed max, shape/axis/empty
cases, broadcasting, high-rank alias/lifetime behavior, rejection of float-only
operations and a tail beyond the 65535-block grid cap. Existing arithmetic,
broadcast, reduction, parity and diagnostics tests now exercise CUDA int32 too.
`tests/cpp/int32_contract.h` additionally bypasses Python validation to check
native scalar/fill errors and unchanged output slots on failure. The exact
commit CUDA push gate includes this suite and explicit test anchors.

Local validation on the recorded CUDA environment:

| Check | Result |
| --- | --- |
| New int32 suite, CPU + CUDA | 484 passed, no skips |
| Affected arithmetic/reduction/CUDA/diagnostic suites | 1375 passed, no skips |
| Full suite with required LLVM/CUDA | 6180 passed, 171 expected platform/optional skips |
| Native contracts | 4/4 passed |
| Native ASan/UBSan contracts | 4/4 passed; `protect_shadow_gap=0`, leak detection disabled |
| New CUDA suite under memcheck / racecheck | 242 passed each; zero errors/hazards |

The example above was also executed on CPU and CUDA. Independent QA checked
extra random-bit inputs and foreign-current-context preservation through
successful operations, rejected scalars and buffer destruction. These checks
do not measure performance or establish Metal execution. Native ASan/UBSan
does not replace GPU memory/race instrumentation or a leak check.

Real CUDA validation remains limited to Linux x86_64, CUDA 12.4 and sm_52.
Additional GPUs/toolchains require actual execution evidence; compilation-only
CI and successful CPU/Metal tests do not establish that compatibility.
