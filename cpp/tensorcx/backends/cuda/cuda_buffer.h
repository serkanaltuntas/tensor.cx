#pragma once

#include "tensorcx/backends/cuda/cuda_context.h"
#include "tensorcx/core/buffer.h"
#include "tensorcx/core/dtype.h"
#include "tensorcx/core/shape.h"

namespace tensorcx::cuda {
struct DeviceDeleter {
  std::shared_ptr<PrimaryContext> owner;
  void operator()(void* pointer) const noexcept;
};

// Concrete buffer and pointer access stay inside the CUDA backend.
class CudaBuffer final : public Buffer {
 public:
  static Expected<std::shared_ptr<CudaBuffer>> create(DType dtype, const Shape& shape);
  static Expected<std::shared_ptr<CudaBuffer>> borrow(DType dtype, const Shape& shape,
                                                     void* data, std::shared_ptr<void> owner);
  std::size_t nbytes() const override { return bytes_; }
  StorageRelation storage_relation(const Buffer& other) const override;
  DType dtype() const { return dtype_; }
  void* data() const { return external_owner_ ? external_data_ : data_.get(); }
  const std::shared_ptr<PrimaryContext>& context() const { return data_.get_deleter().owner; }
 private:
  DType dtype_{DType::kFloat32};
  std::size_t bytes_{0};
  std::unique_ptr<void, DeviceDeleter> data_;
  void* external_data_{nullptr};
  std::shared_ptr<void> external_owner_;
};
}  // namespace tensorcx::cuda
