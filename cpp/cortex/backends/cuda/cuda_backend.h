#pragma once

#include <memory>
#include <string>

#include "cortex/backends/cpu/cpu_tensor.h"
#include "cortex/core/backend.h"
#include "cortex/core/expected.h"

namespace cortex::cuda {

// CUDA handles and allocation details remain private to the backend.
class CudaBuffer;

class CudaTensor {
 public:
  CudaTensor(DType dtype, Shape shape, std::shared_ptr<CudaBuffer> buffer);
  DType dtype() const { return dtype_; }
  const Shape& shape() const { return shape_; }
  const Shape& strides() const { return strides_; }
  const std::shared_ptr<CudaBuffer>& buffer() const { return buffer_; }
  std::size_t nbytes() const;

 private:
  DType dtype_;
  Shape shape_;
  Shape strides_;
  std::shared_ptr<CudaBuffer> buffer_;
};

class CudaBackend final : public Backend {
 public:
  std::string name() const override { return "cuda"; }
  Status execute(const BackendExecution& execution) override;
};

bool available();
Expected<std::string> device_name();
Expected<CudaTensor> from_cpu(const cpu::CpuTensor& tensor);
Expected<cpu::CpuTensor> to_cpu(const CudaTensor& tensor);
Tensor to_core_tensor(const CudaTensor& tensor);
Expected<CudaTensor> from_core_tensor(const Tensor& tensor);

}  // namespace cortex::cuda
