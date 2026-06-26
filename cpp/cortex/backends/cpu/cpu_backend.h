#pragma once

#include "cortex/backends/cpu/cpu_tensor.h"
#include "cortex/core/backend.h"
#include "cortex/core/operation.h"

namespace cortex::cpu {

class CpuBackend final : public Backend {
 public:
  std::string name() const override;
  Status execute(const BackendExecution& execution) override;
};

CpuTensor empty(Shape shape, DType dtype);
CpuTensor fill(Shape shape, DType dtype, double value);
CpuTensor execute_unary(const OpDesc& op, const CpuTensor& input);
CpuTensor execute_binary(const OpDesc& op, const CpuTensor& lhs, const CpuTensor& rhs);
CpuTensor reduce(const OpDesc& op, const CpuTensor& input);
CpuTensor matmul(const CpuTensor& lhs, const CpuTensor& rhs);

Tensor to_core_tensor(const CpuTensor& tensor);
CpuTensor from_core_tensor(const Tensor& tensor);

}  // namespace cortex::cpu
