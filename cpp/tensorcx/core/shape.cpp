#include "tensorcx/core/shape.h"

#include <limits>
#include <algorithm>
#include <stdexcept>

namespace tensorcx {
namespace {

void validate_dim(Dim dim) {
  if (dim < 0) {
    throw std::invalid_argument("shape dimensions must be non-negative");
  }
}

Dim checked_multiply(Dim lhs, Dim rhs, const char* error_message) {
  if (lhs != 0 && rhs > std::numeric_limits<Dim>::max() / lhs) {
    throw std::invalid_argument(error_message);
  }
  return lhs * rhs;
}

}  // namespace

std::int64_t numel(const Shape& shape) {
  if (shape.empty()) {
    return 1;
  }

  std::int64_t total = 1;
  for (Dim dim : shape) {
    validate_dim(dim);
    total = checked_multiply(total, dim, "shape size overflow");
  }
  return total;
}

Shape contiguous_strides(const Shape& shape) {
  Shape strides(shape.size(), 1);
  Dim stride = 1;
  for (auto index = shape.size(); index > 0; --index) {
    validate_dim(shape[index - 1]);
    strides[index - 1] = stride;
    stride = checked_multiply(stride, shape[index - 1], "shape stride overflow");
  }
  return strides;
}

TransposePlan make_slice_plan(const Shape& input, const Shape& starts,
                              const Shape& steps, const Shape& lengths) {
  (void)numel(input);
  const auto strides = contiguous_strides(input);
  if (starts.size() != input.size() || steps.size() != input.size() ||
      lengths.size() != input.size())
    throw std::invalid_argument("slice metadata must match input rank");
  const auto count = numel(lengths);
  (void)contiguous_strides(lengths);
  TransposePlan plan{lengths, Shape(input.size(), 0)};
  for (std::size_t axis = 0; axis < input.size(); ++axis) {
    const auto start = starts[axis], step = steps[axis], length = lengths[axis];
    if (step == 0) throw std::invalid_argument("slice step cannot be zero");
    if (length == 0) continue;
    if (start < 0 || start >= input[axis])
      throw std::invalid_argument("slice start is out of bounds");
    if (length > 1) {
      // Division checks endpoints without overflowing step*(length-1), even
      // for INT64_MIN. Such a negative step cannot select a second element.
      if ((step > 0 && step > (input[axis] - 1 - start) / (length - 1)) ||
          (step < 0 && step < -(start / (length - 1))))
        throw std::invalid_argument("slice endpoint is out of bounds");
      if (count) plan.input_strides[axis] = step * strides[axis];
    }
    if (count) plan.offset += start * strides[axis];
  }
  return plan;
}

ConcatPlan make_concat_plan(const std::vector<Shape>& inputs, Dim axis) {
  if (inputs.empty()) throw std::invalid_argument("concat requires at least one tensor");
  const auto rank = static_cast<Dim>(inputs[0].size());
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) throw std::invalid_argument("concat axis is out of range");
  Shape output = inputs[0];
  output[axis] = 0;
  for (const auto& shape : inputs) {
    (void)numel(shape);
    (void)contiguous_strides(shape);
    if (shape.size() != output.size()) throw std::invalid_argument("concat ranks must match");
    for (std::size_t i = 0; i < shape.size(); ++i) {
      if (i != static_cast<std::size_t>(axis) && shape[i] != output[i])
        throw std::invalid_argument("concat non-axis dimensions must match");
    }
    if (shape[axis] > std::numeric_limits<Dim>::max() - output[axis])
      throw std::invalid_argument("concat dimension overflow");
    output[axis] += shape[axis];
  }
  (void)numel(output);
  const auto strides = contiguous_strides(output);
  return {output, static_cast<std::size_t>(axis), strides[axis]};
}

ReductionPlan make_reduction_plan(const Shape& input, const Shape& axes) {
  (void)numel(input);
  const auto strides = contiguous_strides(input);
  const auto rank = static_cast<Dim>(input.size());
  std::vector<bool> selected(input.size(), false);
  for (auto axis : axes) {
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) throw std::invalid_argument("reduction axis is out of range");
    if (selected[axis]) throw std::invalid_argument("reduction axes must not repeat");
    selected[axis] = true;
  }
  ReductionPlan plan;
  Shape reduced;
  for (std::size_t axis = 0; axis < input.size(); ++axis) {
    if (selected[axis]) {
      reduced.push_back(input[axis]);
    } else {
      plan.output_shape.push_back(input[axis]);
      plan.index_metadata.insert(plan.index_metadata.end(), {input[axis], strides[axis]});
    }
  }
  for (std::size_t axis = 0; axis < input.size(); ++axis) {
    if (selected[axis])
      plan.index_metadata.insert(plan.index_metadata.end(), {input[axis], strides[axis]});
  }
  const auto count = numel(plan.output_shape);
  (void)contiguous_strides(plan.output_shape);
  if (std::find(reduced.begin(), reduced.end(), 0) != reduced.end()) {
    plan.reduction_size = 0;
  } else if (count != 0) {
    plan.reduction_size = numel(reduced);
  }
  return plan;
}

TransposePlan make_transpose_plan(const Shape& input, const Shape& axes) {
  (void)numel(input);
  const auto strides = contiguous_strides(input);
  if (axes.size() != input.size()) {
    throw std::invalid_argument("transpose axes must be a full permutation");
  }
  const auto rank = static_cast<Dim>(input.size());
  std::vector<bool> seen(input.size(), false);
  TransposePlan plan{Shape(input.size()), Shape(input.size())};
  for (std::size_t i = 0; i < axes.size(); ++i) {
    auto axis = axes[i];
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) {
      throw std::invalid_argument("transpose axis is out of range");
    }
    if (seen[axis]) {
      throw std::invalid_argument("transpose axes must not repeat");
    }
    seen[axis] = true;
    plan.output_shape[i] = input[axis];
    plan.input_strides[i] = strides[axis];
  }
  (void)numel(plan.output_shape);
  (void)contiguous_strides(plan.output_shape);
  return plan;
}

BroadcastPlan make_broadcast_plan(const Shape& lhs, const Shape& rhs) {
  (void)numel(lhs);
  (void)numel(rhs);
  const auto lhs_contiguous = contiguous_strides(lhs);
  const auto rhs_contiguous = contiguous_strides(rhs);
  const auto rank = std::max(lhs.size(), rhs.size());
  BroadcastPlan plan{Shape(rank), Shape(rank, 0), Shape(rank, 0)};
  const auto lhs_padding = rank - lhs.size();
  const auto rhs_padding = rank - rhs.size();
  for (std::size_t axis = 0; axis < rank; ++axis) {
    const auto left = axis < lhs_padding ? Dim{1} : lhs[axis - lhs_padding];
    const auto right = axis < rhs_padding ? Dim{1} : rhs[axis - rhs_padding];
    if (left != right && left != 1 && right != 1) {
      throw std::invalid_argument("shape mismatch: tensor shapes are not broadcastable");
    }
    // max(left, right) would incorrectly map 0 with 1 to 1.
    plan.output_shape[axis] = left == 1 ? right : left;
    if (axis >= lhs_padding && left != 1) {
      plan.lhs_strides[axis] = lhs_contiguous[axis - lhs_padding];
    }
    if (axis >= rhs_padding && right != 1) {
      plan.rhs_strides[axis] = rhs_contiguous[axis - rhs_padding];
    }
  }
  (void)numel(plan.output_shape);
  (void)contiguous_strides(plan.output_shape);
  return plan;
}

}  // namespace tensorcx
