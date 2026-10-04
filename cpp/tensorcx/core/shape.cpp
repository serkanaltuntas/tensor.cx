#include "tensorcx/core/shape.h"

#include <limits>
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

}  // namespace tensorcx
