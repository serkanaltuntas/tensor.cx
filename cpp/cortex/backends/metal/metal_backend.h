#pragma once

#include <string>
#include <vector>

#include "cortex/backends/cpu/cpu_tensor.h"
#include "cortex/backends/metal/metal_tensor.h"
#include "cortex/core/backend.h"
#include "cortex/core/expected.h"
#include "cortex/core/operation.h"
#include "cortex/core/status.h"
#include "cortex/core/tensor.h"

namespace cortex::metal {

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

}  // namespace cortex::metal
