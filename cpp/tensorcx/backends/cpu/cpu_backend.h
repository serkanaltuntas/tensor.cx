#pragma once

#include "tensorcx/backends/cpu/cpu_tensor.h"
#include "tensorcx/core/backend.h"
#include "tensorcx/core/operation.h"

namespace tensorcx::cpu {

class CpuBackend final : public Backend {
 public:
  std::string name() const override;
  Status execute(const BackendExecution& execution) override;
};

Status execute_math(const BackendExecution& execution);
CpuTensor empty(Shape shape, DType dtype);
CpuTensor fill(Shape shape, DType dtype, double value);
CpuTensor predicate(const OpDesc& op, std::span<const CpuTensor> inputs);
CpuTensor reduce_boolean(const OpDesc& op, const CpuTensor& input);
CpuTensor masked_select(const CpuTensor& input, const CpuTensor& mask);
CpuTensor execute_unary(const OpDesc& op, const CpuTensor& input);
CpuTensor execute_binary(const OpDesc& op, const CpuTensor& lhs, const CpuTensor& rhs);
CpuTensor reduce(const OpDesc& op, const CpuTensor& input);
CpuTensor matmul(const CpuTensor& lhs, const CpuTensor& rhs);

Tensor to_core_tensor(const CpuTensor& tensor);
CpuTensor from_core_tensor(const Tensor& tensor);
Status contract_smoke_test();

}  // namespace tensorcx::cpu
