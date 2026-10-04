#pragma once

#include <string>
#include <vector>

#include "tensorcx/backends/cpu/cpu_tensor.h"
#include "tensorcx/backends/metal/metal_tensor.h"
#include "tensorcx/core/backend.h"
#include "tensorcx/core/expected.h"
#include "tensorcx/core/operation.h"
#include "tensorcx/core/status.h"
#include "tensorcx/core/tensor.h"

namespace tensorcx::metal {

class MetalBackend final : public Backend {
 public:
  std::string name() const override;
  Status execute(const BackendExecution& execution) override;
};

bool available();
std::vector<std::string> devices();

Expected<MetalTensor> from_cpu(const cpu::CpuTensor& tensor);
Expected<cpu::CpuTensor> to_cpu(const MetalTensor& tensor);
Expected<MetalTensor> fill(const OpDesc& op, Shape shape, DType dtype, double value);
Expected<MetalTensor> execute_unary(const OpDesc& op, const MetalTensor& input);
Expected<MetalTensor> execute_binary(const OpDesc& op, const MetalTensor& lhs, const MetalTensor& rhs);
Expected<MetalTensor> reduce(const OpDesc& op, const MetalTensor& input);
Tensor to_core_tensor(const MetalTensor& tensor);
MetalTensor from_core_tensor(const Tensor& tensor);
Status contract_smoke_test();

}  // namespace tensorcx::metal
