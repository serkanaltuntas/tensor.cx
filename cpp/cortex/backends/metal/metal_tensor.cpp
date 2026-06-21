#include "cortex/backends/metal/metal_tensor.h"

#include <utility>

#include "cortex/backends/metal/metal_buffer.h"
#include "cortex/core/shape.h"

namespace cortex::metal {

MetalTensor::MetalTensor(DType dtype, Shape shape, std::shared_ptr<MetalBuffer> buffer)
    : dtype_(dtype),
      shape_(std::move(shape)),
      strides_(contiguous_strides(shape_)),
      size_(numel(shape_)),
      buffer_(std::move(buffer)) {}

std::size_t MetalTensor::nbytes() const { return buffer_->nbytes(); }

}  // namespace cortex::metal
