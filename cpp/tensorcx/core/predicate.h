#pragma once

#include <span>
#include <vector>
#include "tensorcx/core/operation.h"
#include "tensorcx/core/tensor.h"

namespace tensorcx {

bool is_comparison(OpKind kind);
bool is_logical(OpKind kind);
bool is_predicate_elementwise(OpKind kind);

struct PredicatePlan {
  Shape output_shape;
  std::vector<Shape> input_strides;
  DType output_dtype;
};

// Type/shape validation shared by CPU and accelerators. Concrete buffer and
// device validation remains at each backend's existing from_core_tensor gate.
PredicatePlan make_predicate_plan(const OpDesc& op, std::span<const Tensor> inputs);
struct MaskedSelectPlan { Shape tail_shape; Dim block_size; };
MaskedSelectPlan make_masked_select_plan(const Tensor& input, const Tensor& mask);
Dim broadcast_offset(Dim flat, const Shape& shape, const Shape& strides);

}  // namespace tensorcx
