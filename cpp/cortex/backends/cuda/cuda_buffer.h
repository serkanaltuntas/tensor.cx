#pragma once

#include "cortex/backends/cuda/cuda_context.h"
#include "cortex/core/buffer.h"
#include "cortex/core/dtype.h"
#include "cortex/core/shape.h"

namespace cortex::cuda {
struct DeviceDeleter {
  std::shared_ptr<PrimaryContext> owner;
  void operator()(void* pointer) const noexcept;
};

// Concrete buffer and pointer access stay inside the CUDA backend.
class CudaBuffer final : public Buffer {
 public:
  static Expected<std::shared_ptr<CudaBuffer>> create(DType dtype, const Shape& shape);
  std::size_t nbytes() const override { return bytes_; }
  DType dtype() const { return dtype_; }
  void* data() const { return data_.get(); }
  const std::shared_ptr<PrimaryContext>& context() const { return data_.get_deleter().owner; }
 private:
  DType dtype_{DType::kFloat32};
  std::size_t bytes_{0};
  std::unique_ptr<void, DeviceDeleter> data_;
};
}  // namespace cortex::cuda
