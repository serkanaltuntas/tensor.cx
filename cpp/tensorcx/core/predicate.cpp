#include "tensorcx/core/predicate.h"
#include <algorithm>
#include <stdexcept>

namespace tensorcx {

bool is_comparison(OpKind kind) {
  switch (kind) {
    case OpKind::kEqual: case OpKind::kNotEqual: case OpKind::kLess:
    case OpKind::kLessEqual: case OpKind::kGreater: case OpKind::kGreaterEqual:
      return true;
    default: return false;
  }
}
bool is_logical(OpKind kind) {
  return kind == OpKind::kLogicalAnd || kind == OpKind::kLogicalOr ||
         kind == OpKind::kLogicalXor || kind == OpKind::kLogicalNot;
}
bool is_predicate_elementwise(OpKind kind) {
  return is_comparison(kind) || is_logical(kind) || kind == OpKind::kWhere;
}

PredicatePlan make_predicate_plan(const OpDesc& op, std::span<const Tensor> inputs) {
  const auto expected = op.kind == OpKind::kWhere ? 3U : op.kind == OpKind::kLogicalNot ? 1U : 2U;
  if (!is_predicate_elementwise(op.kind) || inputs.size() != expected)
    throw std::invalid_argument("invalid predicate operation or input count");
  const bool where = op.kind == OpKind::kWhere;
  if (where) {
    if (inputs[0].dtype != DType::kBool) throw std::invalid_argument("where condition must be bool");
    if (inputs[1].dtype != inputs[2].dtype) throw std::invalid_argument("where branch dtypes must match");
  } else {
    for (const auto& input : inputs) {
      if (input.dtype != inputs[0].dtype) throw std::invalid_argument("comparison dtypes must match");
      if (is_logical(op.kind) && input.dtype != DType::kBool)
        throw std::invalid_argument("logical operations require bool tensors");
    }
  }
  PredicatePlan plan{{}, {}, where ? inputs[1].dtype : DType::kBool};
  for (const auto& input : inputs) {
    (void)dtype_size(input.dtype);
    plan.output_shape = make_broadcast_plan(plan.output_shape, input.shape).output_shape;
  }
  for (const auto& input : inputs)
    plan.input_strides.push_back(make_broadcast_plan(input.shape, plan.output_shape).lhs_strides);
  return plan;
}

MaskedSelectPlan make_masked_select_plan(const Tensor& input, const Tensor& mask) {
  if (mask.dtype != DType::kBool) throw std::invalid_argument("mask must be bool");
  (void)dtype_size(input.dtype);
  (void)numel(input.shape); (void)contiguous_strides(input.shape);
  (void)numel(mask.shape); (void)contiguous_strides(mask.shape);
  if (mask.shape.size() > input.shape.size()) throw std::invalid_argument("mask rank exceeds input rank");
  for (std::size_t i = 0; i < mask.shape.size(); ++i)
    if (mask.shape[i] != input.shape[i]) throw std::invalid_argument("mask must match leading input dimensions");
  Shape tail(input.shape.begin() + mask.shape.size(), input.shape.end());
  // Empty masks never address a block. A later zero extent can also make a
  // huge trailing product empty before any overflowing multiplication matters.
  const Dim block = numel(mask.shape) == 0 || std::find(tail.begin(), tail.end(), 0) != tail.end()
      ? 0 : numel(tail);
  return {tail, block};
}

Dim broadcast_offset(Dim flat, const Shape& shape, const Shape& strides) {
  Dim offset = 0;
  for (std::size_t axis = shape.size(); axis-- > 0;) {
    offset += (flat % shape[axis]) * strides[axis];
    flat /= shape[axis];
  }
  return offset;
}

}  // namespace tensorcx
