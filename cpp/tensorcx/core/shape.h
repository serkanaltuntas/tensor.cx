#pragma once

#include <cstdint>
#include <vector>

namespace tensorcx {

using Dim = std::int64_t;
using Shape = std::vector<Dim>;

std::int64_t numel(const Shape& shape);
Shape contiguous_strides(const Shape& shape);

struct BroadcastPlan {
  Shape output_shape;
  Shape lhs_strides;
  Shape rhs_strides;
};

// Right-aligned NumPy-style broadcasting of contiguous inputs. Expanded and
// padded dimensions have zero strides; incompatible/overflowing shapes throw.
BroadcastPlan make_broadcast_plan(const Shape& lhs, const Shape& rhs);

struct MatmulPlan {
  Shape output_shape;
  Shape batch_shape;
  Shape lhs_batch_strides;
  Shape rhs_batch_strides;
  Dim m, k, n;
};

// NumPy-style matmul: rank-one operands are promoted to matrices and their
// temporary axes removed from the result; only leading batch axes broadcast.
MatmulPlan make_matmul_plan(const Shape& lhs, const Shape& rhs);
Dim matmul_batch_offset(Dim batch, const Shape& shape, const Shape& strides);

struct TransposePlan {
  Shape output_shape;
  Shape input_strides;
  Dim offset{0};
};

// Full permutation, including negative axes; checks output metadata even for
// empty tensors. The caller materializes a new contiguous output buffer.
TransposePlan make_transpose_plan(const Shape& input, const Shape& axes);

// Normalized basic slice: signed element strides and starting element offset.
// It retains input rank; integer-axis removal/new axes are metadata reshapes.
TransposePlan make_slice_plan(const Shape& input, const Shape& starts,
                              const Shape& steps, const Shape& lengths);

struct ConcatPlan {
  Shape output_shape;
  std::size_t axis;
  Dim inner;
};
ConcatPlan make_concat_plan(const std::vector<Shape>& inputs, Dim axis);

struct ReductionPlan {
  Shape output_shape;
  // Kept dimensions first, then reduced dimensions, each in input axis order.
  // Metadata has extent/element-stride pairs in that same order.
  Shape index_metadata;
  Dim reduction_size{1};
};

// Validates selected axes and output metadata. With no output elements, a
// nonempty reduction uses size 1 as an unvisited sentinel to avoid overflow
// in an iteration space that will never be traversed. A zero extent stays 0.
ReductionPlan make_reduction_plan(const Shape& input, const Shape& axes);

}  // namespace tensorcx
