#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstring>

namespace {
VKAPI_ATTR VkResult VKAPI_CALL create(const VkInstanceCreateInfo*, const VkAllocationCallbacks*, VkInstance* out) {
  *out = reinterpret_cast<VkInstance>(std::uintptr_t{1});
  return VK_SUCCESS;
}
}
extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance, const char* name) {
  if (std::strcmp(name, "vkCreateInstance") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&create);
  return nullptr;
}
