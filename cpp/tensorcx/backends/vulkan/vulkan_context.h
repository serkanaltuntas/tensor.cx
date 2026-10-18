#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "tensorcx/backends/vulkan/vulkan_api.h"
#include "tensorcx/core/expected.h"

namespace tensorcx::vulkan {

struct Instance;
struct Context;
struct DescriptorPool;

struct Allocation {
  std::shared_ptr<Context> owner;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize capacity = 0;
  bool coherent = false;
  bool allocation_reserved = false;
  void* mapped = nullptr;
  ~Allocation();
};

struct DeviceInfo {
  std::string name;
  std::uint32_t api_version = 0;
  std::uint32_t driver_version = 0;
  std::uint32_t vendor_id = 0;
  std::uint32_t max_storage_buffer_range = 0;
  std::uint64_t device_local_bytes = 0;
};

struct PushConstants {
  std::uint32_t count, operation, scalar_bits, scalar_left;
  std::uint32_t groups_x, rank, integer_dtype, reserved = 0;
};

// GPU completion on device loss follows the Vulkan device-lost lifetime rule.
// Other failed waits do not authorize freeing submitted resources.
bool completion_allows_release(VkResult result);

struct Context : std::enable_shared_from_this<Context> {
  std::shared_ptr<Instance> instance;
  int index = 0;
  DeviceInfo info;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  VkPhysicalDeviceMemoryProperties memory_properties{};
  VkPhysicalDeviceLimits limits{};
  VkCommandPool command_pool = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  std::mutex mutex;
  std::atomic<bool> poisoned{false};
  std::atomic<std::uint32_t> allocation_count{0};
#define TENSORCX_VK_DECLARE(name) PFN_vk##name name = nullptr;
  TENSORCX_VK_DEVICE_FUNCTIONS(TENSORCX_VK_DECLARE)
#undef TENSORCX_VK_DECLARE

  ~Context();
  std::shared_ptr<Allocation> allocate(std::size_t bytes, bool host_visible = false);
  void upload(const std::shared_ptr<Allocation>& destination, const void* data, std::size_t bytes);
  void download(const std::shared_ptr<Allocation>& source, void* data, std::size_t bytes,
                std::size_t offset = 0);
  void compute(const std::array<std::shared_ptr<Allocation>, 4>& buffers, PushConstants push);

 private:
  // Queue, pool and descriptor access is externally synchronized even with the
  // Python GIL released. Callers retain every referenced allocation until idle.
  void submit(const std::vector<std::shared_ptr<Allocation>>& resources,
              const std::function<void(VkCommandBuffer)>& encode,
              std::shared_ptr<DescriptorPool> descriptors = nullptr);
};

std::size_t device_count();
bool available(int index = 0);
Expected<std::shared_ptr<Context>> get_context(int index);
Expected<DeviceInfo> device_info(int index);
std::uint64_t validation_errors();
std::string discovery_error();

}  // namespace tensorcx::vulkan
