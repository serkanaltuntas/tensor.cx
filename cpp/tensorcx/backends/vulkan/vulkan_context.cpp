#include "tensorcx/backends/vulkan/vulkan_context.h"

#include <algorithm>
#include <cstring>
#include <chrono>
#include <cmath>
#include <dlfcn.h>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>

#include "elementwise_spirv.h"
#include "tensorcx/backends/vulkan/vulkan_memory_pool.h"
#include "tensorcx/core/profiling.h"

namespace tensorcx::vulkan {

VulkanError::VulkanError(VkResult code, const char* operation)
    : std::runtime_error(std::string("Vulkan ") + operation + " failed (VkResult " + std::to_string(code) + ")"), result(code) {}
DescriptorPool::~DescriptorPool() {
  if (handle) owner->DestroyDescriptorPool(owner->device, handle, nullptr);
}
Frame::~Frame() {
  if (queries) owner->DestroyQueryPool(owner->device, queries, nullptr);
  if (fence) owner->DestroyFence(owner->device, fence, nullptr);
  if (command) owner->FreeCommandBuffers(owner->device, owner->command_pool, 1, &command);
}

namespace {

std::atomic<std::uint64_t> error_count{0};

void check(VkResult result, const char* operation) {
  if (result != VK_SUCCESS) {
    throw VulkanError(result, operation);
  }
}

VKAPI_ATTR VkBool32 VKAPI_CALL debug_message(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT*, void*) {
  // Messages may contain raw handles and local paths. Expose a count, not logs
  // or identifiers, in the public diagnostics and acceptance artifacts.
  if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ++error_count;
  return VK_FALSE;
}

void memory_barrier(Context& context, VkCommandBuffer command) {
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
                          VK_ACCESS_HOST_READ_BIT;
  constexpr VkPipelineStageFlags stages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT |
                                          VK_PIPELINE_STAGE_HOST_BIT;
  context.CmdPipelineBarrier(command, stages, stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

// A failed idle call (other than device loss) gives no lifetime guarantee. Keep
// that poisoned context and its pending resources alive until process exit.
// This exceptional path favors bounded retention over freeing in-flight memory.
void quarantine(std::unique_ptr<Frame> frame) {
  static_cast<void>(frame.release());
}

}  // namespace

struct Instance {
  struct Physical {
    VkPhysicalDevice handle;
    VkPhysicalDeviceProperties properties;
    std::uint32_t family;
    std::uint32_t timestamp_bits;
  };
  void* loader = nullptr;
  VkInstance handle = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
  std::vector<Physical> physicals;
  PFN_vkGetInstanceProcAddr get_proc = nullptr;
#define TENSORCX_VK_DECLARE(name) PFN_vk##name name = nullptr;
  TENSORCX_VK_INSTANCE_FUNCTIONS(TENSORCX_VK_DECLARE)
#undef TENSORCX_VK_DECLARE
  ~Instance() {
    if (messenger && DestroyDebugUtilsMessengerEXT) DestroyDebugUtilsMessengerEXT(handle, messenger, nullptr);
    if (handle && DestroyInstance) DestroyInstance(handle, nullptr);
    if (loader) dlclose(loader);
  }

  void initialize() {
    loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!loader) throw std::runtime_error("Vulkan loader libvulkan.so.1 is unavailable");
    get_proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(loader, "vkGetInstanceProcAddr"));
    if (!get_proc) throw std::runtime_error("Vulkan loader has no vkGetInstanceProcAddr");
    auto create = reinterpret_cast<PFN_vkCreateInstance>(get_proc(nullptr, "vkCreateInstance"));
    if (!create) throw std::runtime_error("Vulkan loader has no vkCreateInstance");
    const char* layer = "VK_LAYER_KHRONOS_validation";
    const char* extension = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    const char* validation_env = std::getenv("TENSORCX_VULKAN_VALIDATION");
    const bool validation = validation_env && std::string_view(validation_env) == "1";
    VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "tensor.cx";
    app.apiVersion = VK_API_VERSION_1_1;
    VkDebugUtilsMessengerCreateInfoEXT debug{};
  debug.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debug.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debug.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debug.pfnUserCallback = debug_message;
    VkInstanceCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pApplicationInfo = &app;
    if (validation) {
      info.enabledLayerCount = 1; info.ppEnabledLayerNames = &layer;
      info.enabledExtensionCount = 1; info.ppEnabledExtensionNames = &extension;
      info.pNext = &debug;
    }
    check(create(&info, nullptr, &handle), "vkCreateInstance");
#define TENSORCX_VK_LOAD(name) \
    name = reinterpret_cast<PFN_vk##name>(get_proc(handle, "vk" #name));
    TENSORCX_VK_INSTANCE_FUNCTIONS(TENSORCX_VK_LOAD)
#undef TENSORCX_VK_LOAD
    if (!DestroyInstance || !EnumeratePhysicalDevices || !GetPhysicalDeviceProperties ||
        !GetPhysicalDeviceMemoryProperties || !GetPhysicalDeviceQueueFamilyProperties ||
        !CreateDevice || !GetDeviceProcAddr) {
      throw std::runtime_error("Vulkan instance entry points are incomplete");
    }
    if (validation) {
      if (!CreateDebugUtilsMessengerEXT || !DestroyDebugUtilsMessengerEXT)
        throw std::runtime_error("Vulkan validation debug utilities are unavailable");
      check(CreateDebugUtilsMessengerEXT(handle, &debug, nullptr, &messenger),
            "vkCreateDebugUtilsMessengerEXT");
    }
    std::uint32_t count = 0;
    check(EnumeratePhysicalDevices(handle, &count, nullptr), "vkEnumeratePhysicalDevices");
    std::vector<VkPhysicalDevice> devices(count);
    VkResult result = EnumeratePhysicalDevices(handle, &count, devices.data());
    check(result, "vkEnumeratePhysicalDevices");
    devices.resize(count);
    for (auto physical : devices) {
      VkPhysicalDeviceProperties properties;
      GetPhysicalDeviceProperties(physical, &properties);
      if (properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
          properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU &&
          properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) continue;
      if (properties.apiVersion < VK_API_VERSION_1_1 ||
          properties.limits.maxComputeWorkGroupInvocations < 64 ||
          properties.limits.maxComputeWorkGroupSize[0] < 64 ||
          properties.limits.maxPerStageDescriptorStorageBuffers < 4 ||
          properties.limits.maxDescriptorSetStorageBuffers < 4 ||
          properties.limits.maxPushConstantsSize < sizeof(PushConstants)) continue;
      std::uint32_t families = 0;
      GetPhysicalDeviceQueueFamilyProperties(physical, &families, nullptr);
      std::vector<VkQueueFamilyProperties> queues(families);
      GetPhysicalDeviceQueueFamilyProperties(physical, &families, queues.data());
      for (std::uint32_t i = 0; i < families; ++i) {
        if (queues[i].queueCount && (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
          physicals.push_back({physical, properties, i, queues[i].timestampValidBits});
          break;
        }
      }
    }
  }
};

namespace {
struct Discovery {
  std::shared_ptr<Instance> instance;
  std::string error;
  std::mutex mutex;
  std::vector<std::shared_ptr<Context>> contexts;
  Discovery() {
    try {
      auto candidate = std::make_shared<Instance>();
      candidate->initialize();
      instance = std::move(candidate);
      contexts.resize(instance->physicals.size());
      if (contexts.empty()) error = "no eligible Vulkan 1.1 compute GPU (software CPU devices excluded)";
    } catch (const std::exception& e) { error = e.what(); }
  }
};
Discovery& discovery() { static Discovery value; return value; }

void initialize_context(Context& context, const Instance::Physical& physical) {
  auto& instance = *context.instance;
  context.physical = physical.handle;
  context.limits = physical.properties.limits;
  context.info.timestamp_bits = physical.timestamp_bits;
  context.info.timestamp_period = context.limits.timestampPeriod;
  context.info.name = physical.properties.deviceName;
  context.info.api_version = physical.properties.apiVersion;
  context.info.driver_version = physical.properties.driverVersion;
  context.info.vendor_id = physical.properties.vendorID;
  context.info.max_storage_buffer_range = context.limits.maxStorageBufferRange;
  instance.GetPhysicalDeviceMemoryProperties(physical.handle, &context.memory_properties);
  for (std::uint32_t i = 0; i < context.memory_properties.memoryHeapCount; ++i) {
    const auto& heap = context.memory_properties.memoryHeaps[i];
    if (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) context.info.device_local_bytes += heap.size;
  }
  float priority = 1.0f;
  VkDeviceQueueCreateInfo queue{};
  queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue.queueFamilyIndex = physical.family; queue.queueCount = 1; queue.pQueuePriorities = &priority;
  VkDeviceCreateInfo device{};
  device.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device.queueCreateInfoCount = 1; device.pQueueCreateInfos = &queue;
  check(instance.CreateDevice(physical.handle, &device, nullptr, &context.device), "vkCreateDevice");
#define TENSORCX_VK_LOAD(name) \
  context.name = reinterpret_cast<PFN_vk##name>(instance.GetDeviceProcAddr(context.device, "vk" #name));
  TENSORCX_VK_DEVICE_FUNCTIONS(TENSORCX_VK_LOAD)
  TENSORCX_VK_QUERY_FUNCTIONS(TENSORCX_VK_LOAD)
#undef TENSORCX_VK_LOAD
#define TENSORCX_VK_REQUIRE(name) \
  if (!context.name) throw std::runtime_error("Vulkan device entry point missing: vk" #name);
  TENSORCX_VK_DEVICE_FUNCTIONS(TENSORCX_VK_REQUIRE)
#undef TENSORCX_VK_REQUIRE
  context.GetDeviceQueue(context.device, physical.family, 0, &context.queue);
  VkCommandPoolCreateInfo pool{};
  pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; pool.queueFamilyIndex = physical.family;
  check(context.CreateCommandPool(context.device, &pool, nullptr, &context.command_pool), "vkCreateCommandPool");
  std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
  for (std::uint32_t i = 0; i < bindings.size(); ++i) {
    bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  }
  VkDescriptorSetLayoutCreateInfo descriptors{};
  descriptors.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptors.bindingCount = bindings.size(); descriptors.pBindings = bindings.data();
  check(context.CreateDescriptorSetLayout(context.device, &descriptors, nullptr, &context.descriptor_layout),
        "vkCreateDescriptorSetLayout");
  VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
  VkPipelineLayoutCreateInfo layout{};
  layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layout.setLayoutCount = 1; layout.pSetLayouts = &context.descriptor_layout;
  layout.pushConstantRangeCount = 1; layout.pPushConstantRanges = &push;
  check(context.CreatePipelineLayout(context.device, &layout, nullptr, &context.pipeline_layout),
        "vkCreatePipelineLayout");
  VkShaderModuleCreateInfo shader_info{};
  shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shader_info.codeSize = sizeof(elementwise_spirv); shader_info.pCode = elementwise_spirv;
  VkShaderModule shader = VK_NULL_HANDLE;
  check(context.CreateShaderModule(context.device, &shader_info, nullptr, &shader), "vkCreateShaderModule");
  VkComputePipelineCreateInfo pipeline{};
  pipeline.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; pipeline.stage.module = shader;
  pipeline.stage.pName = "main"; pipeline.layout = context.pipeline_layout;
  const auto result = context.CreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &pipeline, nullptr,
                                                     &context.pipeline);
  context.DestroyShaderModule(context.device, shader, nullptr);
  check(result, "vkCreateComputePipelines");
}
}  // namespace

bool completion_allows_release(VkResult result) {
  return result == VK_SUCCESS || result == VK_ERROR_DEVICE_LOST;
}

TimestampInterval timestamp_interval(std::uint64_t begin, std::uint64_t end,
    std::uint32_t bits, float period, std::uint64_t host_elapsed_ns) {
  if (!bits || bits > 64 || !std::isfinite(period) || period <= 0)
    return {std::nullopt, "timestamps_unsupported"};
  const auto wrap_ns = std::ldexp(static_cast<long double>(period), static_cast<int>(bits));
  if (static_cast<long double>(host_elapsed_ns) >= wrap_ns)
    return {std::nullopt, "timestamp_wrap_ambiguous"};
  const auto mask = bits == 64 ? UINT64_MAX : (UINT64_C(1) << bits) - 1;
  const auto ticks = (end - begin) & mask;
  const auto ms = static_cast<float>(static_cast<long double>(ticks) * period / 1000000.0L);
  if (!std::isfinite(ms)) return {std::nullopt, "invalid_duration"};
  return {ms, nullptr};
}

std::size_t device_count() { return discovery().contexts.size(); }
std::string discovery_error() { return discovery().error; }
std::uint64_t validation_errors() { return error_count.load(); }
Expected<std::shared_ptr<Context>> get_context(int index) {
  auto& state = discovery();
  std::lock_guard lock(state.mutex);
  if (index < 0 || static_cast<std::size_t>(index) >= state.contexts.size()) {
    return Status(StatusCode::kUnavailable, state.error.empty() ? "Vulkan device index is unavailable" : state.error);
  }
  if (!state.contexts[index]) {
    try {
      auto context = std::make_shared<Context>();
      context->instance = state.instance; context->index = index;
      initialize_context(*context, state.instance->physicals[index]);
      state.contexts[index] = std::move(context);
    } catch (const std::exception& e) { return Status(StatusCode::kUnavailable, e.what()); }
  }
  if (state.contexts[index]->poisoned) return Status(StatusCode::kUnavailable, "Vulkan context is poisoned after a device error");
  return state.contexts[index];
}
bool available(int index) { return static_cast<bool>(get_context(index)); }
Expected<DeviceInfo> device_info(int index) {
  auto context = get_context(index);
  if (!context) return context.status();
  return context.value()->info;
}

Context::~Context() {
  // All public operations are synchronous. Pending resources on an unproven
  // failed wait retain their context in quarantine instead of reaching here.
  if (pipeline) DestroyPipeline(device, pipeline, nullptr);
  if (pipeline_layout) DestroyPipelineLayout(device, pipeline_layout, nullptr);
  if (descriptor_layout) DestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
  if (command_pool) DestroyCommandPool(device, command_pool, nullptr);
  if (device && DestroyDevice) DestroyDevice(device, nullptr);
}

Allocation::~Allocation() {
  ProfileScope profile(ProfileCategory::kDeallocation, "vulkan_buffer_free", "vulkan", memory_bytes);
  if (mapped) owner->UnmapMemory(owner->device, memory);
  if (buffer) owner->DestroyBuffer(owner->device, buffer, nullptr);
  if (memory) {
    ProfileScope api(ProfileCategory::kDeallocation, "vkFreeMemory", "vulkan", memory_bytes);
    owner->FreeMemory(owner->device, memory, nullptr);
    std::lock_guard lock(owner->stats_mutex_);
    --owner->stats_.physical_blocks; owner->stats_.physical_bytes -= memory_bytes;
    ++owner->stats_.free_calls;
  }
  if (allocation_reserved) --owner->allocation_count;
}

ContextStats Context::statistics() const { std::lock_guard lock(stats_mutex_); return stats_; }
void Context::count(std::uint64_t ContextStats::* field, std::uint64_t value) {
  std::lock_guard lock(stats_mutex_); stats_.*field += value;
}
void Context::cache_change(const Allocation& allocation, bool insert) {
  std::lock_guard lock(stats_mutex_);
  if (insert) { ++stats_.cached_blocks; stats_.cached_bytes += allocation.memory_bytes; }
  else { --stats_.cached_blocks; stats_.cached_bytes -= allocation.memory_bytes; }
}
void Context::lease_change(Allocation& allocation, bool insert) {
  std::lock_guard lock(stats_mutex_);
  const auto kind = static_cast<std::size_t>(allocation.kind);
  if (insert) { ++stats_.live_blocks[kind]; stats_.live_bytes[kind] += allocation.requested_bytes; }
  else { --stats_.live_blocks[kind]; stats_.live_bytes[kind] -= allocation.requested_bytes; }
  allocation.leased = insert;
}

std::shared_ptr<Allocation> Context::allocate(std::size_t bytes, bool host_visible, AllocationKind kind) {
  ProfileScope profile(ProfileCategory::kAllocation, "vulkan_buffer_allocate", "vulkan", bytes);
  if (poisoned) throw std::runtime_error("Vulkan context is poisoned after a device error");
  if (static_cast<std::size_t>(kind) >= 3)
    throw std::invalid_argument("invalid Vulkan allocation kind");
  if (bytes == 0) return nullptr;
  if (bytes > static_cast<std::size_t>(limits.maxStorageBufferRange) || bytes > UINT32_MAX - 3u)
    throw std::invalid_argument("Vulkan allocation exceeds the storage-buffer limit");
  const VkDeviceSize capacity = (static_cast<VkDeviceSize>(bytes) + 3) & ~VkDeviceSize{3};
  if (capacity > limits.maxStorageBufferRange)
    throw std::invalid_argument("Vulkan padded allocation exceeds the storage-buffer limit");
  auto pool = active_memory_pool(index);
  auto allocation = pool ? pool->acquire(capacity, host_visible) : allocate_raw(capacity, host_visible);
  allocation->requested_bytes = bytes; allocation->kind = host_visible ? AllocationKind::kStaging : kind;
  lease_change(*allocation, true);
  return std::shared_ptr<Allocation>(allocation.release(), [pool](Allocation* item) noexcept {
    item->owner->lease_change(*item, false);
    if (pool) pool->release(std::unique_ptr<Allocation>(item));
    else delete item;
  });
}

std::unique_ptr<Allocation> Context::allocate_raw(std::size_t capacity, bool host_visible) {
  auto allocation = std::make_unique<Allocation>();
  allocation->owner = shared_from_this(); allocation->capacity = capacity;
  allocation->host_visible = host_visible;
  try {
  VkBufferCreateInfo buffer{};
  buffer.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer.size = capacity; buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  buffer.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  check(CreateBuffer(device, &buffer, nullptr, &allocation->buffer), "vkCreateBuffer");
  VkMemoryRequirements requirements;
  GetBufferMemoryRequirements(device, allocation->buffer, &requirements);
  const VkMemoryPropertyFlags required = host_visible ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                     : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  std::uint32_t selected = UINT32_MAX;
  for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
    const auto flags = memory_properties.memoryTypes[i].propertyFlags;
    if ((requirements.memoryTypeBits & (1u << i)) && (flags & required) == required) {
      selected = i;
      if (!host_visible || (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) break;
    }
  }
  if (selected == UINT32_MAX) throw std::runtime_error("Vulkan has no compatible memory type");
  allocation->coherent = memory_properties.memoryTypes[selected].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  VkMemoryAllocateInfo memory{};
  memory.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  memory.allocationSize = requirements.size; memory.memoryTypeIndex = selected;
  const auto heap_index = memory_properties.memoryTypes[selected].heapIndex;
  if (requirements.size > memory_properties.memoryHeaps[heap_index].size)
    throw std::invalid_argument("Vulkan allocation exceeds its memory heap");
  auto live = allocation_count.load();
  do {
    if (live >= limits.maxMemoryAllocationCount)
      throw std::runtime_error("Vulkan device memory allocation count limit reached");
  } while (!allocation_count.compare_exchange_weak(live, live + 1));
  allocation->allocation_reserved = true;
  count(&ContextStats::allocation_calls);
  {
    ProfileScope api(ProfileCategory::kAllocation, "vkAllocateMemory", "vulkan", requirements.size);
    check(AllocateMemory(device, &memory, nullptr, &allocation->memory), "vkAllocateMemory");
  }
  allocation->memory_bytes = requirements.size;
  {
    std::lock_guard lock(stats_mutex_);
    ++stats_.physical_blocks; stats_.physical_bytes += requirements.size;
    stats_.peak_physical_bytes = std::max(stats_.peak_physical_bytes, stats_.physical_bytes);
  }
  check(BindBufferMemory(device, allocation->buffer, allocation->memory, 0), "vkBindBufferMemory");
  if (host_visible) check(MapMemory(device, allocation->memory, 0, VK_WHOLE_SIZE, 0, &allocation->mapped), "vkMapMemory");
  return allocation;
  } catch (...) { count(&ContextStats::allocation_failures); throw; }
}

void Context::submit(const std::vector<std::shared_ptr<Allocation>>& resources,
                     const std::function<void(Frame&)>& encode, const char* device_name) {
  auto pool = active_memory_pool(index);
  auto frame = pool ? pool->take_frame() : nullptr;
  const auto token = reserve_device_profile(device_name, "vulkan");
  try {
    if (!frame) {
      frame = std::make_unique<Frame>(); frame->owner = shared_from_this();
      VkCommandBufferAllocateInfo allocate{};
      allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      allocate.commandPool = command_pool; allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      allocate.commandBufferCount = 1;
      check(AllocateCommandBuffers(device, &allocate, &frame->command), "vkAllocateCommandBuffers");
      count(&ContextStats::command_buffer_creations);
      VkFenceCreateInfo fence{}; fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
      check(CreateFence(device, &fence, nullptr, &frame->fence), "vkCreateFence");
      count(&ContextStats::fence_creations);
    }
    frame->resources = resources;
    if (poisoned) throw std::runtime_error("Vulkan context is poisoned after a device error");
    const char* query_error = nullptr;
    bool timing = false;
    if (token) {
      if (!info.timestamp_bits || info.timestamp_bits > 64 ||
          !std::isfinite(info.timestamp_period) || info.timestamp_period <= 0)
        query_error = "timestamps_unsupported";
      else if (!CreateQueryPool || !DestroyQueryPool || !CmdResetQueryPool ||
               !CmdWriteTimestamp || !GetQueryPoolResults)
        query_error = "timestamp_functions_unavailable";
      else {
        if (!frame->queries) {
          VkQueryPoolCreateInfo info{}; info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
          info.queryType = VK_QUERY_TYPE_TIMESTAMP; info.queryCount = 2;
          VkQueryPool queries = VK_NULL_HANDLE;
          if (CreateQueryPool(device, &info, nullptr, &queries) == VK_SUCCESS) {
            frame->queries = queries; count(&ContextStats::query_pool_creations);
          } else query_error = "timestamp_query_creation_failed";
        }
        timing = !query_error;
      }
    }
    {
      ProfileScope record(ProfileCategory::kDispatch, "vulkan_record_commands", "vulkan");
      VkCommandBufferBeginInfo begin{}; begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      check(BeginCommandBuffer(frame->command, &begin), "vkBeginCommandBuffer");
      if (timing) CmdResetQueryPool(frame->command, frame->queries, 0, 2);
      memory_barrier(*this, frame->command);
      if (timing) CmdWriteTimestamp(frame->command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame->queries, 0);
      encode(*frame);
      if (timing) CmdWriteTimestamp(frame->command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame->queries, 1);
      memory_barrier(*this, frame->command);
      check(EndCommandBuffer(frame->command), "vkEndCommandBuffer");
    }
    VkSubmitInfo submission{}; submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submission.commandBufferCount = 1; submission.pCommandBuffers = &frame->command;
    const auto started = token ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    VkResult result;
    {
      ProfileScope api(ProfileCategory::kCompute, "vkQueueSubmit", "vulkan");
      count(&ContextStats::submissions);
      result = QueueSubmit(queue, 1, &submission, frame->fence);
    }
    if (result == VK_SUCCESS) {
      ProfileScope wait(ProfileCategory::kSynchronize, "vkWaitForFences", "vulkan");
      result = WaitForFences(device, 1, &frame->fence, VK_TRUE, UINT64_MAX);
    }
    if (result != VK_SUCCESS) {
      poisoned = true;
      const VkResult idle = DeviceWaitIdle(device);
      if (!completion_allows_release(idle)) quarantine(std::move(frame));
      check(result, "queue submission/completion");
    }
    if (token) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - started).count();
      if (timing) {
        std::array<std::uint64_t, 4> values{}; // (timestamp, availability) per query.
        const auto result = GetQueryPoolResults(device, frame->queries, 0, 2, sizeof(values),
            values.data(), 2 * sizeof(std::uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (result != VK_SUCCESS || !values[1] || !values[3]) query_error = "timestamp_results_unavailable";
        else {
          const auto interval = timestamp_interval(values[0], values[2], info.timestamp_bits,
                                                    info.timestamp_period, elapsed);
          finish_device_profile(token, interval.milliseconds, interval.error);
        }
      }
      if (query_error) finish_device_profile(token, std::nullopt, query_error);
    }
    if (pool) {
      // Completion is proven. Reset command references before releasing any
      // buffer; a reset failure only discards this completed resource cache.
      const bool reset = ResetCommandBuffer(frame->command, 0) == VK_SUCCESS &&
                         ResetFences(device, 1, &frame->fence) == VK_SUCCESS;
      if (reset) frame->resources.clear();
      pool->return_frame(std::move(frame), reset);
    }
  } catch (...) {
    finish_device_profile(token, std::nullopt, "execution_failed");
    throw;
  }
}

void Context::upload(const std::shared_ptr<Allocation>& destination, const void* data, std::size_t bytes) {
  ProfileScope profile(ProfileCategory::kCopy, "vulkan_upload", "vulkan", bytes);
  if (!bytes) return;
  if (!destination || destination->owner.get() != this || !destination->buffer || !data || bytes > destination->capacity)
    throw std::invalid_argument("invalid Vulkan upload range or owner");
  auto staging = allocate(bytes, true);
  {
    ProfileScope host(ProfileCategory::kCopy, "vulkan_host_write", "vulkan", bytes);
    std::memset(staging->mapped, 0, staging->capacity);
    std::memcpy(staging->mapped, data, bytes);
  }
  if (!staging->coherent) {
    VkMappedMemoryRange range{}; range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = staging->memory; range.size = VK_WHOLE_SIZE;
    check(FlushMappedMemoryRanges(device, 1, &range), "vkFlushMappedMemoryRanges");
  }
  std::lock_guard lock(mutex);
  count(&ContextStats::uploads);
  submit({staging, destination}, [&](Frame& frame) {
    VkBufferCopy copy{0, 0, staging->capacity};
    CmdCopyBuffer(frame.command, staging->buffer, destination->buffer, 1, &copy);
  }, "vulkan_upload");
}

void Context::download(const std::shared_ptr<Allocation>& source, void* data, std::size_t bytes,
                       std::size_t offset) {
  ProfileScope profile(ProfileCategory::kCopy, "vulkan_download", "vulkan", bytes);
  if (!bytes) return;
  if (!source || source->owner.get() != this || !source->buffer || !data || offset % 4 ||
      offset > source->capacity || bytes > source->capacity - offset ||
      (bytes + 3u) / 4u > (source->capacity - offset) / 4u)
    throw std::invalid_argument("invalid Vulkan download range or owner");
  auto staging = allocate(bytes, true);
  std::lock_guard lock(mutex);
  count(&ContextStats::downloads);
  submit({source, staging}, [&](Frame& frame) {
    VkBufferCopy copy{offset, 0, staging->capacity};
    CmdCopyBuffer(frame.command, source->buffer, staging->buffer, 1, &copy);
  }, "vulkan_download");
  if (!staging->coherent) {
    VkMappedMemoryRange range{}; range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = staging->memory; range.size = VK_WHOLE_SIZE;
    check(InvalidateMappedMemoryRanges(device, 1, &range), "vkInvalidateMappedMemoryRanges");
  }
  ProfileScope host(ProfileCategory::kCopy, "vulkan_host_read", "vulkan", bytes);
  std::memcpy(data, staging->mapped, bytes);
}

void Context::compute(const std::array<std::shared_ptr<Allocation>, 4>& buffers, PushConstants push) {
  ProfileScope profile(ProfileCategory::kCompute, "vulkan_compute", "vulkan");
  if (!push.count) return;
  const std::uint64_t groups = (static_cast<std::uint64_t>(push.count) + 63) / 64;
  push.groups_x = static_cast<std::uint32_t>(std::min<std::uint64_t>({groups, limits.maxComputeWorkGroupCount[0], 65535u}));
  const std::uint64_t groups_y = (groups + push.groups_x - 1) / push.groups_x;
  if (groups_y > limits.maxComputeWorkGroupCount[1] || groups_y * push.groups_x > UINT32_MAX / 64u)
    throw std::invalid_argument("Vulkan dispatch exceeds the supported grid limit");
  std::lock_guard lock(mutex);
  if (poisoned) throw std::runtime_error("Vulkan context is poisoned after a device error");
  count(&ContextStats::dispatches);
  submit({buffers.begin(), buffers.end()}, [&](Frame& frame) {
    if (!frame.descriptors) {
      VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
      VkDescriptorPoolCreateInfo pool_info{}; pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      pool_info.maxSets = 1; pool_info.poolSizeCount = 1; pool_info.pPoolSizes = &size;
      auto pool = std::make_shared<DescriptorPool>(); pool->owner = shared_from_this();
      check(CreateDescriptorPool(device, &pool_info, nullptr, &pool->handle), "vkCreateDescriptorPool");
      count(&ContextStats::descriptor_pool_creations);
      VkDescriptorSetAllocateInfo allocate{}; allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      allocate.descriptorPool = pool->handle; allocate.descriptorSetCount = 1; allocate.pSetLayouts = &descriptor_layout;
      check(AllocateDescriptorSets(device, &allocate, &frame.set), "vkAllocateDescriptorSets");
      count(&ContextStats::descriptor_set_allocations);
      frame.descriptors = std::move(pool);
    }
    std::array<VkDescriptorBufferInfo, 4> info{};
    std::array<VkWriteDescriptorSet, 4> writes{};
    for (std::uint32_t i = 0; i < buffers.size(); ++i) {
      info[i] = {buffers[i]->buffer, 0, buffers[i]->capacity};
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = frame.set; writes[i].dstBinding = i; writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo = &info[i];
    }
    UpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);
    CmdBindPipeline(frame.command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    CmdBindDescriptorSets(frame.command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &frame.set, 0, nullptr);
    CmdPushConstants(frame.command, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    CmdDispatch(frame.command, push.groups_x, static_cast<std::uint32_t>(groups_y), 1);
  }, "vulkan_compute");
}

}  // namespace tensorcx::vulkan
