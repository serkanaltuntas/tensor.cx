#include "cortex/core/shape.h"

#include <stdexcept>

namespace cortex {

std::int64_t numel(const Shape& shape) {
  if (shape.empty()) {
    return 1;
  }

  std::int64_t total = 1;
  for (Dim dim : shape) {
    if (dim < 0) {
      throw std::invalid_argument("shape dimensions must be non-negative");
    }
    total *= dim;
  }
  return total;
}

Shape contiguous_strides(const Shape& shape) {
  Shape strides(shape.size(), 1);
  Dim stride = 1;
  for (auto index = shape.size(); index > 0; --index) {
    strides[index - 1] = stride;
    stride *= shape[index - 1];
  }
  return strides;
}

}  // namespace cortex
