// Standalone software Vulkan experiment. Never registers a tensor.cx device.
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
std::atomic<std::uint64_t> validation_errors{0};
std::uint64_t created_handles = 0, released_handles = 0;
bool device_failed = false;

void require(bool valid, const char* message) {
  if (!valid) throw std::runtime_error(message);
}
void check(VkResult result, const char* operation) {
  if (result != VK_SUCCESS)
    throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(result));
}

template<class T> class Owned {
 public:
  Owned(T value, std::function<void(T)> release) : value_(value), release_(std::move(release)) {
    if (value_) ++created_handles;
  }
  Owned(const Owned&) = delete;
  Owned& operator=(const Owned&) = delete;
  Owned(Owned&& other) noexcept
      : value_(std::exchange(other.value_, {})), release_(std::move(other.release_)) {}
  ~Owned() {
    if (value_) { release_(value_); ++released_handles; }
  }
  void adopt(T value) noexcept { value_ = value; if (value_) ++created_handles; }
  operator T() const { return value_; }
 private:
  T value_{};
  std::function<void(T)> release_;
};

template<class T, class Create, class Destroy>
Owned<T> make(const char* name, Create create, Destroy destroy) {
  Owned<T> owner({}, destroy);  // Prepare the deleter before allocating a handle.
  T value{};
  auto result = create(&value);
  owner.adopt(value);  // Also releases a partial pipeline on error.
  check(result, name);
  return owner;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debug_message(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT*, void*) {
  if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ++validation_errors;
  return VK_FALSE;
}

void idle_or_exit(VkDevice device) {
  const auto result = vkDeviceWaitIdle(device);
  if (result == VK_ERROR_DEVICE_LOST) device_failed = true;
  if (result != VK_SUCCESS && result != VK_ERROR_DEVICE_LOST) {
    // Completion is unproven: do not unwind/free possibly in-flight objects.
    std::cerr << "unproven device idle; retaining resources until process exit\n";
    std::_Exit(2);
  }
}

std::string quoted(const char* value) {
  std::string out = "\"";
  constexpr char hex[] = "0123456789abcdef";
  for (const unsigned char* p = reinterpret_cast<const unsigned char*>(value); *p; ++p) {
    if (*p == '"' || *p == '\\') { out += '\\'; out += static_cast<char>(*p); }
    else if (*p < 32) { out += "\\u00"; out += hex[*p >> 4]; out += hex[*p & 15]; }
    else out += static_cast<char>(*p);
  }
  return out + '"';
}

std::vector<std::uint32_t> shader_words(const char* path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  require(static_cast<bool>(file), "cannot open SPIR-V fixture");
  const auto bytes = file.tellg();
  require(bytes > 0 && bytes % 4 == 0, "invalid SPIR-V fixture size");
  std::vector<std::uint32_t> words(static_cast<std::size_t>(bytes) / 4);
  file.seekg(0);
  require(static_cast<bool>(file.read(reinterpret_cast<char*>(words.data()), bytes)), "short SPIR-V read");
  require(words[0] == 0x07230203, "invalid SPIR-V magic");
  return words;
}

struct Buffer {
  VkDevice device;
  // Reverse destruction: unmap, destroy buffer, free its bound allocation.
  Owned<VkDeviceMemory> memory;
  Owned<VkBuffer> buffer;
  void* mapped = nullptr;
  bool coherent = false;
  std::size_t count;

  static Owned<VkBuffer> create_buffer(VkDevice d, std::size_t n) {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = n * sizeof(float);
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    return make<VkBuffer>("vkCreateBuffer",
        [&](auto* p) { return vkCreateBuffer(d, &info, nullptr, p); },
        [d](auto p) { vkDestroyBuffer(d, p, nullptr); });
  }
  Buffer(VkDevice d, VkPhysicalDevice physical, std::size_t n)
      : device(d), memory({}, [d](auto v) { vkFreeMemory(d, v, nullptr); }),
        buffer(create_buffer(d, n)), count(n) {
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer, &requirements);
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical, &properties);
    std::uint32_t selected = properties.memoryTypeCount;
    for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
      const auto flags = properties.memoryTypes[i].propertyFlags;
      if ((requirements.memoryTypeBits & (1u << i)) && (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
        selected = i;
        if (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) break;
      }
    }
    require(selected < properties.memoryTypeCount, "no host-visible software memory");
    coherent = properties.memoryTypes[selected].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = selected;
    VkDeviceMemory allocated{};
    const auto result = vkAllocateMemory(device, &allocation, nullptr, &allocated);
    memory.adopt(allocated);
    check(result, "vkAllocateMemory");
    check(vkBindBufferMemory(device, buffer, memory, 0), "vkBindBufferMemory");
    check(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapped), "vkMapMemory");
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() { if (mapped) vkUnmapMemory(device, memory); }
  void upload(const std::vector<float>& values) {
    require(values.size() == count, "upload size mismatch");
    std::memcpy(mapped, values.data(), count * sizeof(float));
    if (!coherent) {
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = memory; range.size = VK_WHOLE_SIZE;
      check(vkFlushMappedMemoryRanges(device, 1, &range), "vkFlushMappedMemoryRanges");
    }
  }
  void emit_bits(const char* name) {
    if (!coherent) {
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = memory; range.size = VK_WHOLE_SIZE;
      check(vkInvalidateMappedMemoryRanges(device, 1, &range), "vkInvalidateMappedMemoryRanges");
    }
    std::cout << ",\"" << name << "\":[";
    for (std::size_t i = 0; i < count; ++i) {
      std::uint32_t bits;
      std::memcpy(&bits, static_cast<const char*>(mapped) + i * 4, 4);
      if (i) std::cout << ',';
      std::cout << bits;
    }
    std::cout << ']';
  }
};

void emit_case(VkDevice device, VkPhysicalDevice physical, std::uint32_t family, VkQueue queue,
               VkDescriptorSetLayout descriptor_layout, VkPipelineLayout pipeline_layout,
               VkPipeline pipeline, unsigned cycle, std::uint32_t size, std::uint32_t active,
               unsigned operation, bool alias) {
  constexpr unsigned block = 64;
  const unsigned groups = std::max(1u, (active + block - 1) / block);
  const std::size_t count = std::max(size, groups * block) + 8;
  std::vector<float> a(count, -1234), b(count, -1234), out(count, -1234);
  for (std::uint32_t i = 0; i < size; ++i) {
    a[i] = (static_cast<int>(i % 37) - 18) * 0.25f;
    b[i] = (static_cast<int>(i % 7) - 3) * 0.5f;
    out[i] = -11;
  }
  Buffer da(device, physical, count), db(device, physical, count), dout(device, physical, count);
  da.upload(a); db.upload(b); dout.upload(out);
  Buffer& target = alias ? da : dout;
  VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
  VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pool_info.maxSets = 1; pool_info.poolSizeCount = 1; pool_info.pPoolSizes = &pool_size;
  auto descriptors = make<VkDescriptorPool>("vkCreateDescriptorPool",
      [&](auto* p) { return vkCreateDescriptorPool(device, &pool_info, nullptr, p); },
      [device](auto p) { vkDestroyDescriptorPool(device, p, nullptr); });
  VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  set_info.descriptorPool = descriptors; set_info.descriptorSetCount = 1;
  set_info.pSetLayouts = &descriptor_layout;
  VkDescriptorSet set{};
  check(vkAllocateDescriptorSets(device, &set_info, &set), "vkAllocateDescriptorSets");
  std::array<VkDescriptorBufferInfo, 3> buffers{{{da.buffer, 0, count * 4},
      {db.buffer, 0, count * 4}, {target.buffer, 0, count * 4}}};
  std::array<VkWriteDescriptorSet, 3> writes{};
  for (unsigned i = 0; i < 3; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = set; writes[i].dstBinding = i; writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo = &buffers[i];
  }
  vkUpdateDescriptorSets(device, 3, writes.data(), 0, nullptr);
  VkCommandPoolCreateInfo commands_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  commands_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT; commands_info.queueFamilyIndex = family;
  auto commands = make<VkCommandPool>("vkCreateCommandPool",
      [&](auto* p) { return vkCreateCommandPool(device, &commands_info, nullptr, p); },
      [device](auto p) { vkDestroyCommandPool(device, p, nullptr); });
  VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  command_info.commandPool = commands; command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command_info.commandBufferCount = 1;
  VkCommandBuffer command{};
  check(vkAllocateCommandBuffers(device, &command_info, &command), "vkAllocateCommandBuffers");
  VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  auto fence = make<VkFence>("vkCreateFence",
      [&](auto* p) { return vkCreateFence(device, &fence_info, nullptr, p); },
      [device](auto p) { vkDestroyFence(device, p, nullptr); });
  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer");
  VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  before.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
  before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 1, &before, 0, nullptr, 0, nullptr);
  vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &set, 0, nullptr);
  const std::array<std::uint32_t, 2> push{active, operation};
  vkCmdPushConstants(command, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push.data());
  vkCmdDispatch(command, groups, 1, 1);
  VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; after.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                       0, 1, &after, 0, nullptr, 0, nullptr);
  check(vkEndCommandBuffer(command), "vkEndCommandBuffer");
  VkSubmitInfo submission{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submission.commandBufferCount = 1; submission.pCommandBuffers = &command;
  const VkFence raw_fence = fence;
  const auto submitted = vkQueueSubmit(queue, 1, &submission, fence);
  if (submitted != VK_SUCCESS) { idle_or_exit(device); check(submitted, "vkQueueSubmit"); }
  const auto waited = vkWaitForFences(device, 1, &raw_fence, VK_TRUE, 60'000'000'000ull);
  if (waited != VK_SUCCESS) { idle_or_exit(device); check(waited, "vkWaitForFences"); }
  std::cout << "{\"cycle\":" << cycle << ",\"size\":" << size << ",\"active\":" << active
            << ",\"operation\":\"" << (operation ? "multiply" : "add")
            << "\",\"alias\":" << (alias ? "true" : "false") << ",\"groups\":" << groups;
  target.emit_bits("result"); da.emit_bits("a_after"); db.emit_bits("b_after");
  std::cout << '}';
}

void cycle(const std::vector<std::uint32_t>& words, unsigned iteration, bool& first) {
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "tensor.cx lavapipe experiment"; app.apiVersion = VK_API_VERSION_1_2;
  VkDebugUtilsMessengerCreateInfoEXT debug{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
  debug.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  debug.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  debug.pfnUserCallback = debug_message;
  const char* layer = "VK_LAYER_KHRONOS_validation";
  const char* extension = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
  VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  info.pNext = &debug; info.pApplicationInfo = &app;
  info.enabledLayerCount = 1; info.ppEnabledLayerNames = &layer;
  info.enabledExtensionCount = 1; info.ppEnabledExtensionNames = &extension;
  auto instance = make<VkInstance>("vkCreateInstance",
      [&](auto* p) { return vkCreateInstance(&info, nullptr, p); },
      [](auto p) { vkDestroyInstance(p, nullptr); });
  const VkInstance raw_instance = instance;
  auto create_debug = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
      vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
  auto destroy_debug = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
      vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
  require(create_debug && destroy_debug, "requested validation debug utilities missing");
  auto messenger = make<VkDebugUtilsMessengerEXT>("vkCreateDebugUtilsMessengerEXT",
      [&](auto* p) { return create_debug(instance, &debug, nullptr, p); },
      [=](auto p) { destroy_debug(raw_instance, p, nullptr); });
  std::uint32_t count = 0;
  check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "vkEnumeratePhysicalDevices");
  require(count == 1, "software probe requires exactly one explicit CPU ICD device");
  VkPhysicalDevice physical{};
  check(vkEnumeratePhysicalDevices(instance, &count, &physical), "vkEnumeratePhysicalDevices");
  require(count == 1, "software ICD device count changed");
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(physical, &properties);
  require(properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU, "selected ICD is not a CPU device");
  require(properties.apiVersion >= VK_API_VERSION_1_2, "software probe requires Vulkan 1.2");
  VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
  VkPhysicalDeviceProperties2 details{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  details.pNext = &driver;
  vkGetPhysicalDeviceProperties2(physical, &details);
  require(driver.driverID == VK_DRIVER_ID_MESA_LLVMPIPE && std::string(driver.driverName) == "llvmpipe",
          "selected CPU ICD is not Mesa lavapipe");
  const auto& limits = properties.limits;
  require(limits.maxComputeWorkGroupInvocations >= 64 && limits.maxComputeWorkGroupSize[0] >= 64 &&
      limits.maxComputeWorkGroupCount[0] >= 17 && limits.maxPushConstantsSize >= 8 &&
      limits.maxDescriptorSetStorageBuffers >= 3 && limits.maxPerStageDescriptorStorageBuffers >= 3 &&
      limits.maxStorageBufferRange >= 4384, "CPU ICD lacks probe compute limits");
  if (iteration == 0) {
    std::cout << "{\"device_type\":\"cpu\",\"driver_id\":" << driver.driverID
              << ",\"driver_name\":" << quoted(driver.driverName)
              << ",\"driver_info\":" << quoted(driver.driverInfo)
              << ",\"api_version\":" << properties.apiVersion << ",\"cycles\":2,\"cases\":[";
  }
  vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
  std::uint32_t family = 0;
  while (family < count && !(families[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) ++family;
  require(family < count && families[family].queueCount, "software compute queue missing");
  float priority = 1;
  VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queue_info.queueFamilyIndex = family; queue_info.queueCount = 1; queue_info.pQueuePriorities = &priority;
  VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  device_info.queueCreateInfoCount = 1; device_info.pQueueCreateInfos = &queue_info;
  auto device = make<VkDevice>("vkCreateDevice",
      [&](auto* p) { return vkCreateDevice(physical, &device_info, nullptr, p); },
      [](auto p) { idle_or_exit(p); vkDestroyDevice(p, nullptr); });
  const VkDevice raw_device = device;
  VkQueue queue{};
  vkGetDeviceQueue(device, family, 0, &queue);
  VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  shader_info.codeSize = words.size() * 4; shader_info.pCode = words.data();
  auto shader = make<VkShaderModule>("vkCreateShaderModule",
      [&](auto* p) { return vkCreateShaderModule(device, &shader_info, nullptr, p); },
      [=](auto p) { vkDestroyShaderModule(raw_device, p, nullptr); });
  std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
  for (unsigned i = 0; i < 3; ++i) bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  layout_info.bindingCount = 3; layout_info.pBindings = bindings.data();
  auto layout = make<VkDescriptorSetLayout>("vkCreateDescriptorSetLayout",
      [&](auto* p) { return vkCreateDescriptorSetLayout(device, &layout_info, nullptr, p); },
      [=](auto p) { vkDestroyDescriptorSetLayout(raw_device, p, nullptr); });
  const VkDescriptorSetLayout raw_layout = layout;
  VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
  VkPipelineLayoutCreateInfo pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipeline_layout_info.setLayoutCount = 1; pipeline_layout_info.pSetLayouts = &raw_layout;
  pipeline_layout_info.pushConstantRangeCount = 1; pipeline_layout_info.pPushConstantRanges = &push;
  auto pipeline_layout = make<VkPipelineLayout>("vkCreatePipelineLayout",
      [&](auto* p) { return vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, p); },
      [=](auto p) { vkDestroyPipelineLayout(raw_device, p, nullptr); });
  VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipeline_info.layout = pipeline_layout;
  pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; pipeline_info.stage.module = shader;
  pipeline_info.stage.pName = "main";
  auto pipeline = make<VkPipeline>("vkCreateComputePipelines",
      [&](auto* p) { return vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, p); },
      [=](auto p) { vkDestroyPipeline(raw_device, p, nullptr); });
  for (const auto size : {0u, 1u, 63u, 64u, 65u, 255u, 256u, 257u, 1025u})
    for (const auto active : std::set<std::uint32_t>{0, size / 2, size})
      for (unsigned operation : {0u, 1u})
        for (bool alias : {false, true}) {
          if (!first) std::cout << ',';
          first = false;
          emit_case(device, physical, family, queue, layout, pipeline_layout, pipeline,
                    iteration, size, active, operation, alias);
        }
}
}  // namespace

int main(int argc, char** argv) {
  static_assert(sizeof(float) == 4 && std::endian::native == std::endian::little);
  try {
    require(argc == 2, "expected one SPIR-V fixture path");
    const auto words = shader_words(argv[1]);
    bool first = true;
    for (unsigned i = 0; i < 2; ++i) {
      cycle(words, i, first);
      require(created_handles == released_handles, "owned Vulkan handles not released after cycle");
      require(validation_errors == 0, "Vulkan validation errors during execution or teardown");
      require(!device_failed, "software device lost during cleanup");
    }
    std::cout << "],\"owned_handles_created\":" << created_handles
              << ",\"owned_handles_released\":" << released_handles
              << ",\"validation_errors\":" << validation_errors
              << ",\"status\":\"passed\"}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
