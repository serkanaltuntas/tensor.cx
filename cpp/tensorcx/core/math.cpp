#include "tensorcx/core/math.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace tensorcx {
bool is_math_operation(OpKind kind) {
  switch (kind) {
    case OpKind::kLog: case OpKind::kSqrt: case OpKind::kAbs: case OpKind::kMin:
    case OpKind::kArgmax: case OpKind::kClip: case OpKind::kTopK: return true;
    default: return false;
  }
}
MathPlan make_math_plan(const OpDesc& op, std::span<const Tensor> inputs) {
  if (!is_math_operation(op.kind) || inputs.size() != (op.kind == OpKind::kClip ? 3U : 1U))
    throw std::invalid_argument("invalid math operation or input count");
  const auto& x = inputs[0];
  for (const auto& t : inputs) {
    if (t.dtype != DType::kFloat32 && t.dtype != DType::kInt32)
      throw std::invalid_argument("math operations require float32 or int32 tensors");
    if (t.dtype != x.dtype) throw std::invalid_argument("math input dtypes must match");
    (void)numel(t.shape); (void)contiguous_strides(t.shape);
  }
  MathPlan p; p.shape = x.shape; p.dtype = x.dtype; p.rank = x.shape.size();
  switch (op.kind) {
    case OpKind::kLog: p.code=0; break; case OpKind::kSqrt: p.code=1; break;
    case OpKind::kAbs: p.code=2; break; case OpKind::kMin: p.code=3; break;
    case OpKind::kArgmax: p.code=4; break; case OpKind::kClip: p.code=5; break;
    case OpKind::kTopK: p.code=6; break; default: break;
  }
  if (p.code < 2 && x.dtype != DType::kFloat32)
    throw std::invalid_argument("log/sqrt only support float32 tensors");
  if (op.kind == OpKind::kClip) {
    for (const auto& t : inputs) p.shape = make_broadcast_plan(p.shape, t.shape).output_shape;
    p.metadata = p.shape; p.rank = p.shape.size();
    for (const auto& t : inputs) {
      auto strides = make_broadcast_plan(t.shape, p.shape).lhs_strides;
      p.metadata.insert(p.metadata.end(), strides.begin(), strides.end());
    }
  } else if (op.kind == OpKind::kMin || op.kind == OpKind::kArgmax) {
    auto axes = op.reduction_axes.value_or(x.shape.empty() ? Shape{} : Shape{op.axis});
    if (x.shape.empty() && !op.reduction_axes && op.axis != 0 && op.axis != -1)
      throw std::invalid_argument("reduction axis out of range");
    auto rp = make_reduction_plan(x.shape, axes);
    p.shape = rp.output_shape; p.metadata = rp.index_metadata; p.reduce = rp.reduction_size;
    if (!p.reduce) throw std::invalid_argument("min/argmax require a non-empty reduction");
    if (op.kind == OpKind::kArgmax) {
      // Validate the index range even when kept dimensions make output empty.
      Dim extent = 1;
      for (Dim axis : axes) {
        if (axis < 0) axis += x.shape.size();
        if (x.shape[axis] > std::numeric_limits<std::int32_t>::max() / extent)
          throw std::invalid_argument("argmax index range exceeds int32");
        extent *= x.shape[axis];
      }
      p.dtype = DType::kInt32;
    }
  } else if (op.kind == OpKind::kTopK) {
    if (x.shape.empty()) throw std::invalid_argument("topk requires rank >= 1");
    Dim axis = op.axis; if (axis < 0) axis += x.shape.size();
    if (axis < 0 || axis >= static_cast<Dim>(x.shape.size())) throw std::invalid_argument("topk axis out of range");
    p.reduce = x.shape[axis]; p.k = op.k;
    if (p.reduce > std::numeric_limits<std::int32_t>::max()) throw std::invalid_argument("topk index range exceeds int32");
    if (p.k < 0 || p.k > p.reduce) throw std::invalid_argument("topk k must be between zero and the axis size");
    p.shape[axis] = p.k;
    // Only traversed products are needed. Empty tensors retain validated shapes.
    if (numel(p.shape)) p.inner = contiguous_strides(x.shape)[axis];
  }
  const auto count = numel(p.shape); (void)contiguous_strides(p.shape);
  p.groups = op.kind == OpKind::kTopK ? (p.k ? count / p.k : 0) : count;
  return p;
}
}
