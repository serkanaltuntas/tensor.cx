# Tensor API extension

User-requested increment, 2026-10-04: basic arithmetic and Python/NumPy real
scalars, contiguous reshape, and reduction `keepdims`. This extends the original
exact-shape API; it does not change the historical Phase 9/10 acceptance records.

## Arithmetic

`x + y`, `x - y`, `x * y`, `x / y`, and `-x` return new tensors. Two tensor
operands must have exactly the same shape, dtype and device, including rank-0
tensors. There is no implicit tensor broadcasting or dtype promotion.

CPU and Metal support float32 for all operations; int32 supports add, subtract,
multiply and negate with defined two's-complement wraparound. Division requires
float32, including empty inputs. CUDA arithmetic remains float32-only; CUDA
int32 copies and reshape are supported.

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
checks also apply to empty tensors. Transpose, arbitrary strides and slicing
are separate future work.

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

`tests/python/test_tensor_api.py` checks the public contract on every available
backend. CUDA acceptance requires a GPU before collection and includes this
file without skips; native contracts cover arithmetic validation and numerical
edge cases. Metal must also pass compilation and device parity on a Metal host.

Local acceptance on 2026-10-04: full suite with required CUDA and MLIR modes
passed 1397 tests, with 160 expected platform/Metal skips. The five-file strict
CUDA acceptance set passed 895 tests without skips; focused API/MLIR checks
passed 449. Native and ASan/UBSan contracts passed 4/4 each. Website checking,
production build and four browser tests passed. Two independent code reviews
and test/acceptance QA closed their findings. These local results do not claim
Metal device execution or additional NVIDIA architecture support.

This increment does not complete the broader product backlog. The existing
[CUDA completion ledger](CUDA_PRODUCT_COMPLETION.md) still tracks additional
GPU/toolchain evidence and an isolated remote GPU runner. Further proposed API
work includes dtype conversion, broadcasting, shape/indexing operations beyond
reshape, and multi-axis reductions. Product proposals also include portable
releases, diagnostics, interoperability, inference primitives, compiler caching,
profiling, more dtypes, and asynchronous execution. These need their own scopes
and acceptance evidence; [PyTorch integration](PYTORCH_PORTABILITY_ROADMAP.md)
remains a separate staged track.
