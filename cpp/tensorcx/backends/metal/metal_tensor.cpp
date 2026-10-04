#include "tensorcx/backends/metal/metal_tensor.h"

#include <limits>
#include <stdexcept>
#include <utility>

#include "tensorcx/backends/metal/metal_buffer.h"
#include "tensorcx/core/shape.h"

namespace tensorcx::metal {

MetalTensor::MetalTensor(DType dtype, Shape shape, std::shared_ptr<MetalBuffer> buffer)
    : dtype_(dtype),
      shape_(std::move(shape)),
      size_(numel(shape_)),
      strides_(contiguous_strides(shape_)),
      buffer_(std::move(buffer)) {
  if (!buffer_) {
    throw std::invalid_argument("Metal tensor requires a buffer");
  }
  if (buffer_->dtype() != dtype_) {
    throw std::invalid_argument("Metal buffer dtype mismatch");
  }
  const auto element_bytes = dtype_size(dtype_);
  if (static_cast<std::uint64_t>(size_) >
      std::numeric_limits<std::size_t>::max() / element_bytes) {
    throw std::invalid_argument("Metal tensor byte size overflow");
  }
  if (buffer_->nbytes() != static_cast<std::size_t>(size_) * element_bytes) {
    throw std::invalid_argument("Metal buffer size does not match tensor shape");
  }
}

std::size_t MetalTensor::nbytes() const { return buffer_->nbytes(); }

}  // namespace tensorcx::metal
