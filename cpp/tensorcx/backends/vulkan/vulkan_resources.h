#pragma once

#include "tensorcx/backends/vulkan/vulkan_context.h"

namespace tensorcx::vulkan {
struct DescriptorPool {
  std::shared_ptr<Context> owner;
  VkDescriptorPool handle = VK_NULL_HANDLE;
  ~DescriptorPool();
};

// Destruction/reset requires Context::mutex: the command pool is shared.
// An unproven completion quarantines this object with every referenced buffer.
struct Frame {
  std::shared_ptr<Context> owner;
  std::vector<std::shared_ptr<Allocation>> resources;
  VkCommandBuffer command = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  std::shared_ptr<DescriptorPool> descriptors;
  VkDescriptorSet set = VK_NULL_HANDLE;
  VkQueryPool queries = VK_NULL_HANDLE;
  ~Frame();
};
}  // namespace tensorcx::vulkan
