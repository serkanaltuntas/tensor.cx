#include "cortex/backends/metal/metal_buffer.h"

#include <cstring>
#include <stdexcept>

#include "cortex/backends/metal/metal_context.h"
#include "cortex/core/dtype.h"

namespace cortex::metal {

MetalBuffer::MetalBuffer(DType dtype, std::size_t elements)
    : dtype_(dtype), nbytes_(elements * dtype_size(dtype)) {
  if (nbytes_ == 0) {
    return;
  }

  buffer_ = NS::TransferPtr(default_context().device().newBuffer(
      nbytes_, MTL::ResourceStorageModeShared));
  if (!buffer_) {
    throw std::runtime_error("failed to allocate Metal buffer");
  }
}

void MetalBuffer::copy_from_host(const void* src, std::size_t nbytes) {
  if (nbytes != nbytes_) {
    throw std::invalid_argument("host-to-Metal copy size mismatch");
  }
  if (nbytes_ == 0) {
    return;
  }
  std::memcpy(buffer_->contents(), src, nbytes_);
}

void MetalBuffer::copy_to_host(void* dst, std::size_t nbytes) const {
  if (nbytes != nbytes_) {
    throw std::invalid_argument("Metal-to-host copy size mismatch");
  }
  if (nbytes_ == 0) {
    return;
  }
  std::memcpy(dst, buffer_->contents(), nbytes_);
}

}  // namespace cortex::metal
