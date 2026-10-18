#pragma once

#include "tensorcx/backends/vulkan/vulkan_tensor.h"
#include "tensorcx/core/backend.h"

namespace tensorcx::vulkan {

class VulkanBackend final : public Backend {
 public:
  std::string name() const override { return "vulkan"; }
  Status execute(const BackendExecution& execution) override;
};

}  // namespace tensorcx::vulkan
