# Tensor API extension

User-requested increment, 2026-10-04: basic arithmetic and Python/NumPy real
scalars, contiguous reshape, and reduction `keepdims`, followed by explicit
`astype` conversion, tensor broadcasting, and shape operations (`transpose`,
`squeeze`, `expand_dims`), followed by all-axis/multi-axis reductions and basic indexing/concat/stack/split, then boolean comparisons and masks. These extend the original API;
they do not change the historical Phase 9/10 acceptance records.

## Arithmetic

`x + y`, `x - y`, `x * y`, `x / y`, and `-x` return new tensors. Two tensor
operands must have the same dtype and device. Shapes broadcast by aligning axes
from the right: each pair must be equal or one dimension must be 1. Missing
leading dimensions act as 1; rank-0 tensors broadcast to any compatible shape.
For example, `(2, 3) + (3,)` yields `(2, 3)` and `(2, 1, 3) + (4, 1)` yields
`(2, 4, 3)`. Dimension 0 paired with 1 stays 0; 0 paired with 2 is incompatible.
Incompatible shapes and result metadata overflow raise `ValueError`, including
empty results. There is no implicit dtype promotion or device transfer.

Broadcasting reads original contiguous buffers with per-axis index mapping;
it does not allocate expanded operands. The result is a new contiguous tensor.
Equal-shape arithmetic retains its direct indexing path. This applies only to
ordinary binary arithmetic. Matmul has separate batch broadcasting rules below;
experimental generated kernels retain their exact-shape contract.

CPU and Metal support float32 for all operations; int32 supports add, subtract,
multiply and negate with defined two's-complement wraparound. Division requires
float32, including empty inputs. CUDA arithmetic remains float32-only; CUDA
int32 copies, explicit casts and reshape are supported.

A Python or NumPy integer/float scalar may be on either side: `x + 2`, `2 - x`,
`x * 0.5`, `2 / x`. Scalars preserve the tensor's dtype, shape and device. Int32
requires an integer scalar within int32 bounds; floats (even `1.0`) and
out-of-range integers are rejected. Float32 scalars are rounded to float32 before
the operation; narrowing can lose precision or overflow to infinity. Booleans,
complex numbers and arrays are not arithmetic scalars. Floating division uses
IEEE zero/infinity/NaN behavior; it does not raise Python `ZeroDivisionError`.

Each scalar operation is a single-input native primitive using
`OpDesc.scalar_value` and `scalar_left`; it does not materialize a repeated
scalar tensor or transfer input data through CPU. CPU references and static
Metal/CUDA kernels use the shared `Backend::execute` path.

## Explicit dtype conversion

`x.astype(dtype, copy=True)` and `cx.astype(x, dtype, copy=True)` accept
`cx.float32`/`"float32"`, `cx.int32`/`"int32"` and `cx.bool`/`"bool"` on CPU, Metal and CUDA. They
preserve shape and device, including rank-0 and empty tensors. `copy` is a
keyword-only Python/NumPy boolean. With the default, even a same-dtype cast owns
a new buffer. `copy=False` returns the input for an unchanged dtype; a changed
dtype still allocates. Reshape views and inputs are never mutated.

- Int32 to float32 rounds to the nearest representable value (ties to even),
  so integers beyond float32 precision may change. In particular, `INT32_MAX`
  rounds up to `2147483648.0`, which cannot be cast back to int32.
- Float32 to int32 truncates toward zero. NaN, infinity, and values outside
  `[-2147483648, 2147483648)` raise `ValueError`. Each value is checked
  before conversion; any invalid value fails the operation without publishing
  a result or changing caller tensors.
- Numeric to bool maps zero (including signed zero) to false and every nonzero
  value, NaN and infinity to true. Bool to numeric yields 0 or 1.
- Same-dtype copies preserve the stored bits, including signed zero and NaN.

Conversion executes natively on the selected device. Checked GPU float-to-int
conversion reads back only a validation flag, not the tensor. CUDA int32
arithmetic remains unsupported; explicitly cast to float32 before arithmetic.

```python
values = cx.tensor([[1, 2, 3], [4, 5, 6]])
bias = cx.tensor([0.25, 0.5, 0.75])
result = values.astype(cx.float32) + bias
centered = result - result.mean(axis=-1, keepdims=True)
print(centered.numpy())  # [[-1.25, 0., 1.25], [-1.25, 0., 1.25]]
```

## Contiguous reshape

`cx.reshape(x, shape)` and `x.reshape(shape)` return a new metadata view over
the same native buffer. The original shape is unchanged. The dtype, device,
element count and buffer lifetime are preserved without a copy or GPU launch.
Deleting the source tensor does not invalidate the view. Experimental Metal
kernel output writes are visible through other views sharing that buffer.
MLIR CPU/CUDA launches retain their existing output-copy semantics and preserve
caller tensors; the CPU interpreter also preserves callers while recognizing
shared-storage arguments. Ordinary arithmetic is out of place; NumPy export copies.

The shape is an integer or iterable of integer dimensions; `()` means scalar.
One dimension may be `-1` and is inferred from the element count. Other negative
dimensions, booleans, non-integral dimensions, mismatched sizes and metadata
overflow are rejected. For empty tensors, `(-1, 3)` resolves to `(0, 3)`;
`(0, -1)` is ambiguous and rejected. Existing contiguous shape/stride overflow
checks also apply to empty tensors. Arbitrary strided views remain separate future work; basic indexing below
materializes contiguous copies.

## Transpose and singleton dimensions

`cx.transpose(x, axes=None)`, `x.transpose(axes=None)`, and `x.T` permute axes
into a **new contiguous buffer on the same device**. With `None` (or `.T`),
all axes are reversed, including tensors of rank greater than two. Otherwise,
`axes` is a full permutation; negative indices are normalized against input
rank. Repeated, missing and out-of-range axes are rejected. An integer is
accepted for rank-one tensors. Even identity and scalar transposes copy; they
do not return strided views. The result works with existing matmul, reshape,
reductions and generated kernels that require contiguous inputs.

`cx.squeeze(x, axis=None)` / `x.squeeze(axis=None)` remove every size-one axis
by default. An integer or iterable selects only those axes to remove; selecting
a non-singleton axis is an error. Negative axes refer to input rank. On a
scalar, integer axis `0` or `-1` is a no-op; a non-empty axis sequence is invalid.

`cx.expand_dims(x, axis)` / `x.expand_dims(axis)` insert size-one axes at an
integer position or several positions. Axes, including negative indices, refer
to the **final output rank**. For example, expanding `(2, 3)` at `(0, -1)` gives
`(1, 2, 3, 1)`. Repeated and out-of-range axes are rejected. Both singleton
operations return metadata views sharing the original storage and its lifetime,
using the same ownership rules as reshape. An empty axis tuple returns an
unchanged-shape view. Axis values follow the integer protocol; booleans,
strings, non-integral values and values outside signed 64-bit bounds are rejected.

All three operations support float32/int32/bool on CPU, Metal and CUDA, including
scalars and empty tensors. Transpose preserves stored bits (including NaN
payloads and signed zero), with native CPU/MSL/CUDA execution and only O(rank)
metadata transferred to a GPU. It adds no arbitrary rank cap. The existing
Metal element-count limit and all shape/product/stride overflow checks still
apply, including checks on the permuted shape of an empty tensor. Singleton
views require no data copy or GPU launch.

```python
x = cx.tensor([[1, 2, 3], [4, 5, 6]])
print(x.T.numpy())  # [[1, 4], [2, 5], [3, 6]]
print(x.expand_dims((0, -1)).shape)  # (1, 2, 3, 1)
print(x.expand_dims((0, -1)).squeeze().shape)  # (2, 3)
```

## Basic indexing and joining

`x[key]` accepts integers, slices (`start:stop:step`, including negative steps),
`...`, `None` for new axes, and tuples combining them. Negative integers count
from the end; slices clip their bounds like Python/NumPy. Selecting all axes
with integers returns a rank-0 **Tensor**. Even `x[...]` and `x[()]` copy.
Boolean Tensor masks have the separate contract below. Advanced integer
array/list indexing, masks combined with other indices, and indexed assignment
are not supported. Invalid integer indices raise `IndexError`; invalid slice bounds or
zero steps raise `ValueError`.

- `cx.concat(tensors, axis=0)` joins a non-empty iterable along an existing axis.
  Ranks and all non-axis dimensions must match; scalars cannot be concatenated.
- `cx.stack(tensors, axis=0)` joins equal-shaped tensors along a new axis.
  Scalar inputs are supported; the axis is relative to the final output rank.
- `cx.split(x, indices_or_sections, axis=0)` / `x.split(...)` return a tuple.
  A positive integer requests equal sections and must divide the axis exactly.
  An iterable gives NumPy-style cut indices, not chunk sizes: negative indices
  count from the end, out-of-range cuts clip, repeated or descending cuts can
  produce empty pieces. Empty axes can be divided into positive section counts.

All operations preserve dtype/device and stored bits on CPU, Metal and CUDA,
including float32/int32/bool, scalar stack and empty outputs. Inputs to a join must
share dtype/device; no casting, broadcasting or device transfers occur. Axes
accept integers (including negative axes); booleans are rejected. Every result
owns independent contiguous storage, including each split piece and a
single-input concat. Results can feed reshape, reductions and matmul normally.

The shared native `kSlice` validates normalized start/step/length metadata and
copies with signed strides. Variadic `kConcat` validates all inputs before
allocating once and copying each input into its output region. GPU paths execute
on-device, transferring only slice metadata. `stack` composes shared-storage
expand_dims with native concat; `split` composes native slices. General strided
views and advanced indexing remain outside this increment.

```python
x = cx.tensor([[1, 2, 3], [4, 5, 6]])
print(x[:, ::-1].numpy())  # [[3, 2, 1], [6, 5, 4]]
print(cx.concat([x, x], axis=0).shape)  # (4, 3)
print(cx.stack([x, x], axis=1).shape)  # (2, 2, 3)
print([part.shape for part in x.split([1], axis=1)])  # [(2, 1), (2, 2)]
```

## Reduction dimensions

`cx.sum/max/mean(x, axis=None, keepdims=False)` and corresponding tensor methods
accept one integer, an iterable of integer axes, or `None` (the default) to
reduce every axis. Negative indices refer to input rank. Repeated axes,
out-of-range axes, booleans, strings and non-integral values are rejected.
Axis values must fit signed 64 bits. An explicit empty selection `axis=()`
reduces nothing and returns an independent same-shape copy preserving stored
bits; dtype restrictions still apply. On a scalar, `None`, integer `0` and `-1`
use the existing scalar reduction; `()` copies, and non-empty sequences fail.

`keepdims=True` replaces every selected axis with size 1, using a metadata view
of the independent result. Otherwise those axes are removed. The flag must be
a Python or NumPy boolean. Results retain the input dtype and device. Sum/max
support float32/int32 on CPU/Metal; CUDA sum/max and all means require
float32. Int32 sums wrap in two's-complement, with no implicit promotion.

Selected axes are traversed in their original input order, independent of the
order in the axis argument. Multiple-axis mean sums each group once in float32
then divides once by the total number of selected elements. It is not a chain
of intermediate means. NaNs propagate; sum/mean accumulation can overflow or
lose precision, and NumPy may use a different accumulation order. The existing
single-axis path remains unchanged. Max keeps the first equal value in the
multi-axis traversal, including signed zeros.

If any selected dimension is zero, sum fills zeros, mean fills NaNs, and max
raises `ValueError`, even when the output would also be empty. A zero in a kept
dimension instead yields an empty output. Output shape/stride overflow is
always checked. An empty output does not multiply an unvisited nonzero
reduction space, so valid empty shapes with very large dimensions remain usable.
Existing Metal element-count limits apply; there is no additional rank cap.

The optional native `OpDesc.reduction_axes` and shared `make_reduction_plan`
map output/reduced coordinates directly into the original contiguous buffer.
CPU, MSL and CUDA paths allocate an independent output and do not materialize
a transposed intermediate or download the input. GPU indexing metadata is
O(rank). Metal retains its existing host initialization of zero/NaN identities
for empty reductions. Normalization operations still accept only one axis;
generated-kernel semantics are unchanged.

```python
import tensorcx as cx

x = cx.tensor([[1.0, 2.0], [3.0, 4.0]])
print(((x - 1) / 2).numpy())  # [[0.  0.5], [1.  1.5]]
print(x.reshape((4,)).shape)  # (4,)
print(x.sum(axis=-1, keepdims=True).numpy())  # [[3.], [7.]]
print(x.sum().numpy())  # 10.0
print(x.mean(axis=(0, 1), keepdims=True).numpy())  # [[2.5]]
```

## Verification and remaining work

`tests/python/test_tensor_api.py`, `tests/python/test_cast_broadcast.py`,
`tests/python/test_shape_ops.py`, `tests/python/test_multi_axis_reductions.py`
`tests/python/test_indexing_joining.py` and `tests/python/test_boolean_masks.py`
check the public contract on every available
backend. CUDA acceptance requires a GPU before collection and includes these
files without skips; native contracts cover arithmetic, cast, broadcast and
transpose/multi-axis reduction, signed slice mapping, variadic concat validation
and numerical edge cases. Metal must
also pass compilation and device parity on a Metal host.

First increment acceptance on 2026-10-04 (before casts/broadcasting): full suite
with required CUDA and MLIR modes
passed 1397 tests, with 160 expected platform/Metal skips. The five-file strict
CUDA acceptance set passed 895 tests without skips; focused API/MLIR checks
passed 449. Native and ASan/UBSan contracts passed 4/4 each. Website checking,
production build and four browser tests passed. Two independent code reviews
and test/acceptance QA closed their findings. These local results do not claim
Metal device execution or additional NVIDIA architecture support.

Cast/broadcast local acceptance on 2026-10-04: required CUDA/MLIR full suite
passed 1573 tests with 160 expected platform/Metal skips. The new API file
passed 176 tests without skips on CPU/CUDA. Native and ASan/UBSan contracts
passed 4/4 each. Fresh CPU and CUDA sdist/wheel installations passed with LLVM
present, absent, and GPU hidden; the installed wheels passed 88 CPU and 176
CPU/CUDA cast/broadcast tests respectively. Website checking, production build,
four browser tests and the documented operations example passed. These results
establish local CPU/CUDA acceptance; Metal compilation/device execution remains
a separate macOS CI gate.

Shape-operation local acceptance on 2026-10-04: required CUDA/MLIR full suite
passed 1737 tests with 160 expected platform/Metal skips. The new shape API file
passed 164 CPU/CUDA tests without skips. Native and ASan/UBSan contracts passed
4/4 each. Fresh CPU/CUDA sdist-to-wheel installations passed with LLVM present,
absent and the GPU hidden; installed wheels passed 82 CPU and 164 CPU/CUDA
shape-operation tests. Website checking, production build, four browser tests
and the operations example passed. These results establish local CPU/CUDA
acceptance; Metal compilation and device execution require a separate macOS gate.

Multi-axis reduction local acceptance on 2026-10-04: required CUDA/MLIR full
suite passed 2059 tests with 160 expected platform/Metal skips. The new API
file passed 322 CPU/CUDA tests without skips. Native and ASan/UBSan contracts
passed 4/4 each. Fresh CPU/CUDA sdist-to-wheel installations passed with LLVM
present, absent and the GPU hidden; installed wheels passed 161 CPU and 322
CPU/CUDA multi-axis tests. Website checking, production build, four browser
tests and the operations example passed. These are local CPU/CUDA results;
Metal compilation and device execution require the separate macOS gate.

Indexing/joining local acceptance on 2026-10-04: 340 new CPU/CUDA tests passed,
including negative/large indices, empty and scalar tensors, storage independence,
bit preservation and no host export. The full LLVM-enabled suite passed 2398
with 161 expected platform/opt-in skips. Native and ASan/UBSan contracts passed
4/4; all 340 indexing tests also passed CUDA memcheck with zero errors. Fresh
CPU/CUDA sdist-to-wheel installs passed with LLVM present/absent and the GPU
hidden; installed wheels passed 170 CPU / 340 CPU+CUDA indexing tests. Website
check/build, four preview tests, five Workers-emulation tests, deploy dry run and
the operations snippet passed. Metal compilation/device execution remains the
separate macOS gate; these results do not cover other NVIDIA architectures.

The subsequent [CI run for `871bd64`](https://github.com/serkanaltuntas/tensor.cx/actions/runs/37213435073)
passed all 11 jobs, including Metal with MPSGraph enabled and disabled. The
MPSGraph-disabled job ran all 340 indexing tests on CPU/Metal without skips;
its full suite passed 1688 tests with 871 expected CUDA/LLVM/platform skips,
and both native CPU/Metal contracts passed. The same commit's mandatory local
CUDA push gate passed 1897 tests without skips, all four native contracts,
memcheck/racecheck and MLP CPU parity. Isolated remote GPU CI remains pending.

This increment does not complete the broader product backlog. The existing
[CUDA completion ledger](CUDA_PRODUCT_COMPLETION.md) still tracks additional
GPU/toolchain evidence and an isolated remote GPU runner. Further proposed API
work follows the ordered [product feature backlog](ROADMAP.md#product-feature-backlog).
Product proposals also include portable
releases, diagnostics, interoperability, compiler caching,
profiling, more dtypes, and asynchronous execution. These need their own scopes
and acceptance evidence; [PyTorch integration](PYTORCH_PORTABILITY_ROADMAP.md)
remains a separate staged track.

## Comparisons and boolean masks

`cx.bool` / `"bool"` is a one-byte dtype on CPU, Metal and CUDA. Boolean input
infers it; mixed boolean/numeric constructor input requires an explicit dtype.
`empty`, `zeros`, `ones`, copies, casts and shape/index/join operations support it.
Numeric arithmetic, numeric reductions and normalizations reject bool tensors.

- Six operators (`==`, `!=`, `<`, `<=`, `>`, `>=`) and `cx.equal`, `not_equal`,
  `less`, `less_equal`, `greater`, `greater_equal` return bool tensors. Operands
  share dtype/device and follow binary broadcasting. Float32 comparisons use
  IEEE rules (NaN is unequal even to itself); int32 comparisons are exact.
- Python/NumPy scalars follow the tensor dtype: real nonboolean scalars for
  float32, in-range integers for int32, and booleans for bool. No implicit
  tensor casts or transfers occur. Scalar constants become rank-0 tensors on
  the operand device; input tensors stay on that device.
- `&`, `|`, `^`, `~` and `cx.logical_and`, `logical_or`, `logical_xor`,
  `logical_not` operate on bool tensors with broadcasting. Parenthesize each
  comparison, for example `(x > 0) & (x < 10)`.
- `cx.where(condition, x, y)` / `x.where(condition, y)` broadcasts all three
  inputs. The condition is a bool Tensor; branches have the same dtype/device
  and may be matching scalars. If both branches are scalars, their inferred
  dtypes must match. This selects stored values, preserving signed zero and
  NaN payloads; it does not lazily evaluate branches.
- `cx.any(x, axis=None, keepdims=False)` / `all` and Tensor methods accept bool
  inputs only. All/single/multiple/negative axes, empty axis selections and
  `keepdims` follow the numeric reduction contract. Empty reductions yield
  false for any and true for all. Every output owns a new buffer.
- `x[mask]` and `cx.masked_select(x, mask)` require a bool Tensor mask on the
  same device whose shape matches the input's leading dimensions, without
  broadcasting. Result shape is `(selected_count, *unmasked_trailing_shape)`.
  Thus a full-shape mask flattens selected elements; a row mask keeps remaining
  axes. Scalar bool Tensor masks add a leading size-0/1 axis. Selection is
  stable in row-major order and copies into independent contiguous storage.
  Mixed mask/basic-index tuples, integer-array indexing and assignment remain
  unsupported; Python bool indices are rejected.
- `bool(x)` requires exactly one element (of any supported dtype), including
  high-rank singleton shapes. Empty/multielement tensors raise `ValueError`;
  use `any`/`all` to state the intended reduction. Truth conversion reads one
  value back to the host.

The native operations use the shared `Backend::execute` contract. GPU
comparisons, logical operations, where and reductions do not export tensors
through CPU/NumPy. Mask selection uses a device prefix sum and gather; only
its final selected count is read back to size the output. The current scan
uses O(n log n) work and O(n) scratch space with synchronous launches; it is a
correctness-first implementation, not a tuned compaction benchmark claim.
Generated-kernel dtype/operator subsets remain unchanged.

Verification: `tests/python/test_boolean_masks.py` covers NumPy/CPU parity,
IEEE values, integer precision, scalars, broadcasting, empty/high-rank tensors,
ownership, invalid inputs and the no-host-fallback contract. Native
`predicate_contract.h` exercises CPU/Metal/CUDA byte storage, all new primitive
families and invalid descriptors without replacing outputs on failure; it is
also included in the CUDA push gate's GPU sanitizer runs.


Boolean/mask local acceptance on 2026-10-04: 560 new CPU/CUDA tests passed;
the full required-CUDA/LLVM suite passed 2959 with 160 expected platform/Metal
skips. Native and ASan/UBSan contracts passed 4/4; the 560 new tests passed
GPU memcheck/racecheck with zero errors or hazards. Fresh CPU/CUDA sdist-to-wheel
installations passed with LLVM present/absent and GPU hidden, and installed
wheels passed 280 CPU / 560 CPU+CUDA tests. Website checking/build, four preview
tests, five Workers tests, deploy dry run and the operations snippet passed.
Two code reviews and independent test/acceptance QA closed their findings.
Metal compilation and device execution remain the separate macOS CI gate;
these local results do not cover additional NVIDIA architectures.


## Batched matrix multiplication

`cx.matmul(lhs, rhs, backend="auto")` and `lhs @ rhs` now accept float32 tensors
with rank >= 1 on CPU, Metal and CUDA. Dtype and device must match; scalar
operands, implicit dtype promotion and implicit device transfer are rejected.

The last two dimensions represent `(M, K)` and `(K, N)`; contracting dimensions
must match exactly. Leading batch dimensions broadcast from the right using the
same equal-or-one rule as elementwise operations, including zero versus one.
For example, `(2, 1, 3, 4) @ (1, 5, 4, 6)` returns `(2, 5, 3, 6)`. No expanded
copies of the operands are allocated.

Rank-one inputs follow NumPy semantics. A left vector becomes `(1, K)` and a
right vector becomes `(K, 1)` for execution; the corresponding temporary output
axes are then omitted. Vector dot products return a rank-0 Tensor; matrix/vector
returns `(..., M)`, vector/matrix returns `(..., N)`. Matrix dimensions never
broadcast against the contracting dimension.

Every result owns new contiguous storage. Empty batch/M/N dimensions produce
empty tensors. A zero K produces positive zeros, including the dot product of
two empty vectors. Invalid batch/K dimensions, dtypes and metadata overflow are
still rejected for empty outputs. Inputs and supplied native result slots stay
unchanged on failure. Metadata-only reshapes may be used as inputs.

CPU executes each output against its broadcasted batch offsets. CUDA extends
the existing 16x16 shared-memory tiled kernel over a bounded, grid-stride batch
index; one launch handles the full product. Custom Metal kernels use 64-bit
operand offsets and retain the existing 2^32-1 output-element limit. GPU paths
upload only O(batch rank) metadata and do not export input tensors to NumPy/CPU.
The shared backend-neutral MatmulPlan owns promotion, broadcasting and overflow
checks. This adds no batched generated-kernel/MLIR or integer matmul support.

Backend choices stay explicit: CPU `auto`/`cpu`/`reference`, CUDA `auto`/`custom`,
and Metal `auto`/`custom`/`optimized` when MPSGraph is enabled. The optimized
path uses [MPSGraph broadcasting](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph/matrixmultiplication(primary:secondary:name:)).
It removes common singleton batch axes and promotes vectors in graph metadata,
without copying the inputs. The graph is limited to 16 dimensions (including
matrix axes), matching [MPSNDArray's dimensional limit](https://developer.apple.com/videos/play/wwdc2020/10677/);
explicit optimized requests above that limit fail clearly. `auto` selects custom
when either original input rank exceeds 16. Custom kernels have no rank cap.
Zero-K products use the custom zero-producing path, and empty products do not
submit work, including explicit optimized requests. Accumulation order can vary;
compare float32 results at the existing matmul tolerance (rtol/atol 1e-4).

`tests/python/test_batched_matmul.py` compares NumPy and CPU across every
available backend preference, vector/matrix combinations, broadcast axes,
partial tiles, empty/high-rank inputs, grid-stride batches, special values,
ownership and no-host-fallback behavior. The shared native
`tests/cpp/batched_matmul_contract.h` checks CPU/Metal/CUDA parity and malformed
input descriptors without replacing result slots on errors. The new Python
file and native checks are included in the strict CUDA push gate.

Batched-matmul local acceptance on 2026-10-04: all 200 new CPU/CUDA cases
passed; the full required-CUDA/LLVM suite passed 3171 with 160 expected platform
and capability skips. Native and ASan/UBSan contracts passed 4/4; the new cases
passed GPU memcheck/racecheck with zero errors or hazards. Fresh CPU/CUDA
sdist-to-wheel installations passed with LLVM present/absent and GPU hidden;
installed wheels passed 120 CPU / 200 CPU+CUDA cases. Website checking/build,
four preview tests, five Workers tests, deploy dry run and the operations
snippet passed. Two code reviews and separate test/acceptance QA closed their
findings. Real Metal execution remains the separate macOS CI gate; these local
results do not cover additional NVIDIA architectures or make performance claims.


## Math and selection

The top-level APIs below also have Tensor methods; Python `abs(x)` calls
`x.abs()`. All results own contiguous storage on the input device. Operations
use native CPU/Metal/CUDA execution without exporting input tensors to the host.
There is no implicit dtype promotion or device transfer. Generated kernels and
MLIR operation subsets are unchanged.

| API | Input/output dtype | Semantics |
| --- | --- | --- |
| `log(x)` | float32 → float32 | Natural logarithm; ±0 → -inf, negatives → NaN |
| `sqrt(x)` | float32 → float32 | Negative nonzero → NaN; preserves -0 |
| `abs(x)` | float32/int32 → same | Clears float sign bits; int32 minimum wraps to itself |
| `min(x, axis=None, keepdims=False)` | float32/int32 → same | All/single/multiple axes, like sum/max; NaNs propagate |
| `argmax(x, axis=None, keepdims=False)` | float32/int32 → int32 | Flattened index for None, or one integer axis; first maximum/first NaN wins |
| `clip(x, lower=None, upper=None)` | float32/int32 → same | Broadcast scalar/Tensor bounds; at least one bound required |
| `topk(x, k, axis=-1, largest=True, sorted=True)` | float32/int32 → (same, int32) | Values and axis indices, with the selected axis replaced by k |

Bool inputs are rejected. `log`/`sqrt` reject int32, even for empty inputs.
Use explicit `astype` when a conversion is intended. Int32 scalar clip bounds
must be integral and in range; float32 bounds are narrowed from real scalars.
Tensor bounds must share dtype/device. Clip broadcasts all three inputs; a
reversed lower/upper pair yields the upper bound. Any participating NaN
propagates. Clipping equal numeric values selects the boundary's stored bits.
Absent bounds are implemented with the dtype's minimum/maximum (infinities for
float32). Input values and exported NumPy copies remain independent.

Min supports negative/sequence axes, keepdims and `axis=()` as a copy. Min and
argmax reject an empty reduced dimension, even if another dimension makes the
output empty. Scalar argmax accepts None/0/-1 and returns index zero. Argmax
rejects axis sequences. Equal extrema retain the first input value/index in
canonical input-axis order; +0 and -0 compare equal. NaNs are not ignored.
The [NumPy argmax contract](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html)
is the reference for axis and first-index behavior.

Topk requires rank >= 1, an integer `k` in `[0, axis_size]` and one valid axis.
Zero k returns two empty tensors. Negative axes and non-last axes are supported.
With `largest=True`, NaNs rank above +inf; with `False`, NaNs follow all numeric
values. Equal values (including signed zeros and multiple NaNs) use smaller
original indices first. `sorted=True` orders by value in the requested direction;
`sorted=False` returns the same selected entries in original axis-index order.
Both flags require booleans. Argmax's reduced index space and topk's axis size
must be at most INT32_MAX, including empty-output requests. Broader index/dtype
support belongs to the separate dtype workstream.

Core MathPlan validates operation arity, dtype, broadcasting, axes, shapes and
index ranges. Backend converters validate storage and device metadata. Topk is
a two-output BackendExecution primitive; failure preserves both supplied output
slots. The CPU uses partial sort, O(n log k) selection with O(n) temporary indices
per group (O(n log n) when k=n). GPU topk uses one thread per group and O(n*k)
selection, plus O(k²) index ordering for `sorted=False`; it allocates only output
and O(rank) metadata. This is a correctness-first synchronous implementation,
not a tuned topk or throughput claim. Float comparison/selection on Metal uses
IEEE word ordering to preserve subnormals and signed-zero ties. Log/sqrt
normalize subnormal magnitudes before arithmetic to avoid device input flushing.
Metal retains its 2^32-1 output-element limit.

`tests/python/test_math_ops.py` covers NumPy/CPU parity, all seven APIs, dtype and
argument validation, broadcasting, empty/scalar/high-rank tensors, ties,
NaN/infinity/subnormal values, ownership and no-host-fallback behavior.
`tests/cpp/math_contract.h` exercises native CPU/Metal/CUDA parity and malformed
metadata/arity, including preservation of both topk outputs on failure. Both
are part of the strict CUDA push gate. Additional hardware and portable package
publication still require their separate acceptance work.

Math local acceptance on 2026-10-04: 446 new CPU/CUDA cases passed; the full
required-CUDA/LLVM suite passed 3617 with 160 expected platform/capability skips.
Native and ASan/UBSan contracts passed 4/4. The new cases passed GPU memcheck and
racecheck with zero errors/hazards. CPU/CUDA sdist-to-wheel installations passed
with LLVM present/absent and GPU hidden; fresh installed wheels passed 223 CPU /
446 CPU+CUDA math tests. Website checking/build, four preview tests, five Workers
tests, deploy dry run and the operations snippet passed. Two code reviews and
separate test/acceptance QA closed their findings. Metal device execution is
verified separately through macOS CI; local results do not cover other NVIDIA
architectures, portable wheel publication or a throughput claim.

## Inference operations

All inference values and parameters are float32 on the same device. CPU, Metal
and CUDA use native execution through the shared backend ABI; no operation
implicitly casts, transfers devices, or exports tensor data to Python/NumPy.
Results own independent contiguous buffers. Tensor methods mirror the functions;
the attention method is `Tensor.attention` (the long-form name is a module function).

| Function | Contract |
| --- | --- |
| `linear(input, weight, bias=None)` | `input (..., I) @ weight (O, I).T`, plus optional bias `(O,)`; output `(..., O)` |
| `rmsnorm(input, axis, eps=1e-5, *, weight=None)` | Existing RMSNorm followed by optional per-axis multiplication |
| `layernorm(input, axis, eps=1e-5, *, weight=None, bias=None)` | Existing LayerNorm followed by optional per-axis multiplication and addition |
| `embedding(indices, weight)` | Gather from float32 `(vocabulary, width)` using int32 indices of any rank; output `indices.shape + (width,)` |
| `scaled_dot_product_attention(query, key, value, attn_mask=None, *, is_causal=False, scale=None)` | Stable masked attention; `attention` is an alias |

Normalization still requires one explicit axis, including negative axes. Each
provided affine parameter must have exactly the vector shape `(axis_size,)`;
scalar input instead requires scalar parameters and axis 0 or -1. LayerNorm
allows bias without weight. The existing unweighted calls are unchanged.
`eps` must be finite, nonnegative and representable as float32. Zero epsilon
retains the existing NaN behavior for zero denominators. Empty tensors do not
bypass argument validation.

Embedding accepts only nonnegative indices below the vocabulary size. Repeated
indices and scalar/empty index tensors are supported. Invalid indices are
rejected even when width is zero. The gather preserves float32 bit patterns,
including NaN payloads and signed zero. GPU index validation reads back a single
error flag before gathering; it does not transfer the index or weight tensors
for host computation. There is no padding index, max-norm update or gradient.

Attention uses query `(..., L, E)`, key `(..., S, E)` and value `(..., S, Ev)`;
all have rank at least two, `E` is positive, and leading dimensions broadcast
from the right across all three inputs. The result is `(..., L, Ev)`.
An ordinary leading axis can represent heads; grouped-query head replication
is not implemented. Query/key sequence lengths may differ. Empty batches,
query/key sequences and value widths are supported; a zero key sequence
produces zeros in a nonempty output.

The score is `(query @ key.T) * scale`. The default scale is `1/sqrt(E)`;
explicit scale must be a real, finite, float32-representable scalar. Zero,
negative and subnormal scales are accepted; nonzero values that underflow to
zero are rejected. A bool mask permits entries marked True. A float32 mask
adds a score bias; negative infinity excludes an entry. Masks must broadcast
to the score shape without adding output dimensions. A causal mask permits
`key_position <= query_position` (upper-left alignment for unequal lengths)
and may be combined with an explicit mask.

Excluded entries ignore their score, including a NaN score. Unmasked NaNs
propagate. A row containing positive-infinity scores produces NaNs under the
usual max-subtraction formula. Fully excluded or all-negative-infinity rows
have zero probabilities. Their output is zero for finite values; nonfinite
values can still propagate NaNs through the final matrix multiplication.
This exception applies to attention's masked softmax, not standalone softmax.

Linear and affine normalization compose existing native primitives. Attention
composes two native matmuls with a stable masked-softmax primitive and allocates
score/probability buffers proportional to the full broadcasted `L*S` shape.
It is synchronous and prioritizes correctness; it is not FlashAttention and
makes no throughput claim. Dropout, autograd, KV caching and training are outside
this API. The generated-kernel/MLIR subset is unchanged.

`tests/python/test_inference_ops.py` contains NumPy references, CPU/backend
comparisons, empty and exceptional-value cases, parameter validation and a
complete embedding→linear→attention→affine-normalization→linear workflow.
`tests/cpp/inference_contract.h` checks the native ABI and output preservation
on invalid descriptors. Actual platform acceptance is recorded only after the
corresponding execution checks pass.


Local inference validation (2026-10-04): the full CPU/CUDA Python suite passed
4017 tests with 160 expected platform/capability skips; the new inference suite
passed all 400 CPU/CUDA cases. Native and ASan/UBSan suites each passed 4/4.
The 400 inference cases also passed CUDA memcheck and racecheck with no reported
errors/hazards. Fresh CPU/CUDA sdist→wheel installs passed the distribution
checks; isolated installed-wheel inference tests passed 199 CPU and 400
CPU/CUDA cases. These results apply to the documented validation host/toolchain;
they do not establish additional NVIDIA architecture support. Metal acceptance
must use the real-device CI jobs; Linux validation does not substitute for it.
