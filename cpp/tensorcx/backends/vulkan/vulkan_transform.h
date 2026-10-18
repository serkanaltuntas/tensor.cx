#pragma once

#include "tensorcx/core/backend.h"

namespace tensorcx::vulkan {

// Contiguous device copies/conversions. The common primitive ABI is validated
// by VulkanBackend before entering this executor.
Status execute_transform(const BackendExecution& execution);

}  // namespace tensorcx::vulkan
