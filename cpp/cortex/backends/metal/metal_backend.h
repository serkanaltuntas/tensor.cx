#pragma once

#include <string>
#include <vector>

#include "cortex/backends/cpu/cpu_tensor.h"
#include "cortex/backends/metal/metal_tensor.h"

namespace cortex::metal {

bool available();
std::vector<std::string> devices();

MetalTensor from_cpu(const cpu::CpuTensor& tensor);
cpu::CpuTensor to_cpu(const MetalTensor& tensor);

}  // namespace cortex::metal
