#include "cortex/backends/metal/metal_buffer.h"

#include <cstring>
#include <limits>

#include "cortex/backends/metal/metal_context.h"
#include "cortex/core/dtype.h"

namespace cortex::metal {

Expected<std::shared_ptr<MetalBuffer>> MetalBuffer::create(DType dtype, std::size_t elements) {
  // The byte size below is computed as elements * dtype_size. An unchecked
  // wraparound would allocate a tiny buffer that later copies would overrun,
  // so reject the request before the multiplication can overflow size_t.
  const std::size_t element_size = dtype_size(dtype);
  if (element_size != 0 &&
      elements > std::numeric_limits<std::size_t>::max() / element_size) {
    return Status(StatusCode::kInvalidArgument, "Metal buffer size overflows size_t");
  }

  auto& context = default_context();
  if (!context.ready()) {
    return context.status();
  }

  auto buffer = std::shared_ptr<MetalBuffer>(new MetalBuffer(dtype, elements));
  if (!buffer->valid()) {
    return Status(StatusCode::kInternal, "failed to allocate Metal buffer");
  }
  return buffer;
}

MetalBuffer::MetalBuffer(DType dtype, std::size_t elements)
    : dtype_(dtype), nbytes_(elements * dtype_size(dtype)) {
  if (nbytes_ == 0) {
    return;
  }

  auto& context = default_context();
  if (!context.ready()) {
    return;
  }

  buffer_ = NS::TransferPtr(default_context().device().newBuffer(
      nbytes_, MTL::ResourceStorageModeShared));
}

Status MetalBuffer::copy_from_host(const void* src, std::size_t nbytes) {
  if (nbytes != nbytes_) {
    return Status(StatusCode::kInvalidArgument, "host-to-Metal copy size mismatch");
  }
  if (nbytes_ == 0) {
    return Status::Ok();
  }
  if (!buffer_) {
    return Status(StatusCode::kInternal, "Metal buffer is not allocated");
  }
  std::memcpy(buffer_->contents(), src, nbytes_);
  return Status::Ok();
}

Status MetalBuffer::copy_to_host(void* dst, std::size_t nbytes) const {
  if (nbytes != nbytes_) {
    return Status(StatusCode::kInvalidArgument, "Metal-to-host copy size mismatch");
  }
  if (nbytes_ == 0) {
    return Status::Ok();
  }
  if (!buffer_) {
    return Status(StatusCode::kInternal, "Metal buffer is not allocated");
  }
  std::memcpy(dst, buffer_->contents(), nbytes_);
  return Status::Ok();
}

}  // namespace cortex::metal
