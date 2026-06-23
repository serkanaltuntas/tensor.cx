#pragma once

#include <string>
#include <vector>

#include "cortex/backends/cpu/cpu_tensor.h"
#include "cortex/backends/metal/metal_tensor.h"
#include "cortex/core/expected.h"
#include "cortex/core/operation.h"

namespace cortex::metal {

bool available();
std::vector<std::string> devices();

Expected<MetalTensor> from_cpu(const cpu::CpuTensor& tensor);
Expected<cpu::CpuTensor> to_cpu(const MetalTensor& tensor);
Expected<MetalTensor> fill(const OpDesc& op, Shape shape, DType dtype, double value);
Expected<MetalTensor> execute_binary(const OpDesc& op, const MetalTensor& lhs, const MetalTensor& rhs);
Expected<MetalTensor> reduce(const OpDesc& op, const MetalTensor& input);

}  // namespace cortex::metal
