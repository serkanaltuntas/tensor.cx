#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <span>
#include <string>
#include <vector>

#include "tensorcx/backends/vulkan/vulkan_api.h"
#include "tensorcx/core/expected.h"
#include "tensorcx/core/buffer.h"

namespace tensorcx::vulkan {

struct Instance;
struct Context;
struct DescriptorPool;
struct Frame;
struct KernelPipeline;
class MemoryPool;
struct SharedStorage;

struct ExternalSubmission {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkSemaphore wait = VK_NULL_HANDLE, signal = VK_NULL_HANDLE;
  bool acquire = false;
};
class ExternalAccess : public StorageAccess {
 public:
  virtual std::size_t logical_bytes() const = 0;
  virtual ExternalSubmission prepare() = 0;
  virtual void complete() = 0;
  virtual void cancel() noexcept = 0;
  virtual void poison() noexcept = 0;
};
class VulkanError : public std::runtime_error {
 public:
  VulkanError(VkResult result, const char* operation);
  VkResult result;
};

enum class AllocationKind : std::size_t { kTensor, kInternal, kStaging };

#define TENSORCX_VK_COUNTERS(X) \
  X(physical_blocks) X(physical_bytes) X(peak_physical_bytes) \
  X(cached_blocks) X(cached_bytes) X(allocation_calls) X(free_calls) X(allocation_failures) \
  X(uploads) X(downloads) X(dispatches) X(submissions) \
  X(command_buffer_creations) X(fence_creations) X(descriptor_pool_creations) \
  X(descriptor_set_allocations) X(query_pool_creations) X(frame_reuses)

struct ContextStats {
#define TENSORCX_VK_COUNTER(name) std::uint64_t name = 0;
  TENSORCX_VK_COUNTERS(TENSORCX_VK_COUNTER)
#undef TENSORCX_VK_COUNTER
  std::array<std::uint64_t, 3> live_blocks{}, live_bytes{};
};

struct Allocation {
  std::shared_ptr<Context> owner;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize capacity = 0;
  bool coherent = false;
  bool allocation_reserved = false;
  bool leased = false, host_visible = false;
  AllocationKind kind = AllocationKind::kInternal;
  std::size_t requested_bytes = 0, memory_bytes = 0;
  void* mapped = nullptr;
  // External access owns the physical handles; this Allocation is only a view.
  // CUDA aliases may keep that owner alive after the Vulkan view disappears.
  std::shared_ptr<ExternalAccess> external;
  ~Allocation();
};

struct DeviceInfo {
  std::string name;
  std::uint32_t api_version = 0;
  std::uint32_t driver_version = 0;
  std::uint32_t vendor_id = 0;
  std::uint32_t max_storage_buffer_range = 0;
  std::uint64_t device_local_bytes = 0;
  std::uint32_t timestamp_bits = 0;
  float timestamp_period = 0;
};

struct PushConstants {
  std::uint32_t count, operation, scalar_bits, scalar_left;
  std::uint32_t groups_x, rank, integer_dtype, reserved = 0;
};

// GPU completion on device loss follows the Vulkan device-lost lifetime rule.
// Other failed waits do not authorize freeing submitted resources.
bool completion_allows_release(VkResult result);

struct TimestampInterval { std::optional<float> milliseconds; const char* error; };
TimestampInterval timestamp_interval(std::uint64_t begin, std::uint64_t end,
    std::uint32_t bits, float period, std::uint64_t host_elapsed_ns);

struct Context : std::enable_shared_from_this<Context> {
  std::shared_ptr<Instance> instance;
  int index = 0;
  DeviceInfo info;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  std::uint32_t queue_family = 0;
  std::uint64_t process = 0;
  std::array<std::uint8_t, VK_UUID_SIZE> device_uuid{};
  VkExternalMemoryFeatureFlags external_memory = 0;
  VkExternalSemaphoreFeatureFlags external_semaphore = 0;
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
  TENSORCX_VK_QUERY_FUNCTIONS(TENSORCX_VK_DECLARE)
  TENSORCX_VK_EXTERNAL_DEVICE_FUNCTIONS(TENSORCX_VK_DECLARE)
#undef TENSORCX_VK_DECLARE

  ~Context();
  bool current_process() const;
  void require_process() const;
  void synchronize_external(const std::shared_ptr<Allocation>& allocation);
  std::shared_ptr<Allocation> allocate(std::size_t bytes, bool host_visible = false,
                                     AllocationKind kind = AllocationKind::kInternal);
  void upload(const std::shared_ptr<Allocation>& destination, const void* data, std::size_t bytes);
  void download(const std::shared_ptr<Allocation>& source, void* data, std::size_t bytes,
                std::size_t offset = 0);
  void copy(const std::shared_ptr<Allocation>& destination,
            const std::shared_ptr<Allocation>& source, std::size_t bytes);
  void compute(const std::array<std::shared_ptr<Allocation>, 4>& buffers, PushConstants push,
               std::shared_ptr<KernelPipeline> generated = {}, std::uint32_t block = 64);
  ContextStats statistics() const;
  void count(std::uint64_t ContextStats::* field, std::uint64_t value = 1);
  void cache_change(const Allocation& allocation, bool insert);

 private:
  friend class MemoryPool;
  friend struct Allocation;
  friend struct SharedStorage;
  std::unique_ptr<Allocation> allocate_raw(std::size_t capacity, bool host_visible);
  void lease_change(Allocation& allocation, bool insert);
  mutable std::mutex stats_mutex_;
  ContextStats stats_;
  // Queue, pool and descriptor access is externally synchronized even with the
  // Python GIL released. Callers retain every referenced allocation until idle.
  void submit(const std::vector<std::shared_ptr<Allocation>>& resources,
              const std::function<void(Frame&)>& encode, const char* device_name);
};

std::size_t device_count();
bool available(int index = 0);
Expected<std::shared_ptr<Context>> get_context(int index);
Expected<DeviceInfo> device_info(int index);
std::uint64_t validation_errors();
std::string discovery_error();

}  // namespace tensorcx::vulkan
