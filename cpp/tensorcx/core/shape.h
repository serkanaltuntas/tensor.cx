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

struct TransposePlan {
  Shape output_shape;
  Shape input_strides;
};

// Full permutation, including negative axes; checks output metadata even for
// empty tensors. The caller materializes a new contiguous output buffer.
TransposePlan make_transpose_plan(const Shape& input, const Shape& axes);

}  // namespace tensorcx
