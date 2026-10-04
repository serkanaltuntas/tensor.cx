# Tensor API extension

User-requested increment, 2026-10-04: basic arithmetic and Python/NumPy real
scalars, contiguous reshape, and reduction `keepdims`, followed by explicit
`astype` conversion, tensor broadcasting, and shape operations (`transpose`,
`squeeze`, `expand_dims`). These extend the original API;
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
ordinary binary arithmetic, not matmul or experimental generated kernels.

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
`cx.float32`/`"float32"` and `cx.int32`/`"int32"` on CPU, Metal and CUDA. They
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
checks also apply to empty tensors. Arbitrary strides and slicing remain
separate future work.

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

All three operations support float32/int32 on CPU, Metal and CUDA, including
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

## Reduction dimensions

`cx.sum/max/mean(x, axis, keepdims=False)` and corresponding tensor methods
retain existing results by default. `keepdims=True` replaces the reduced axis
with size 1 using a metadata view of the result. Negative axes work. A rank-0
input remains rank-0 for axis 0/-1. `keepdims` must be a Python or NumPy boolean.
The existing empty-axis and dtype contracts are unchanged. Axis is still one
explicit integer; all-axis and multiple-axis reductions remain future work.

```python
import tensorcx as cx

x = cx.tensor([[1.0, 2.0], [3.0, 4.0]])
print(((x - 1) / 2).numpy())  # [[0.  0.5], [1.  1.5]]
print(x.reshape((4,)).shape)  # (4,)
print(x.sum(axis=-1, keepdims=True).numpy())  # [[3.], [7.]]
```

## Verification and remaining work

`tests/python/test_tensor_api.py`, `tests/python/test_cast_broadcast.py` and
`tests/python/test_shape_ops.py` check the public contract on every available
backend. CUDA acceptance requires a GPU before collection and includes these
files without skips; native contracts cover arithmetic, cast, broadcast and
transpose validation and numerical edge cases. Metal must
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

This increment does not complete the broader product backlog. The existing
[CUDA completion ledger](CUDA_PRODUCT_COMPLETION.md) still tracks additional
GPU/toolchain evidence and an isolated remote GPU runner. Further proposed API
work includes slicing, concatenation and multi-axis reductions.
Product proposals also include portable
releases, diagnostics, interoperability, inference primitives, compiler caching,
profiling, more dtypes, and asynchronous execution. These need their own scopes
and acceptance evidence; [PyTorch integration](PYTORCH_PORTABILITY_ROADMAP.md)
remains a separate staged track.
