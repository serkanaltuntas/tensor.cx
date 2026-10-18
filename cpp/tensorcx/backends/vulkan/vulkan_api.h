#pragma once

// All Vulkan types and entry points stay in the backend. The loader is optional
// at runtime; a Vulkan-enabled wheel must still import on a CPU-only machine.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#define TENSORCX_VK_INSTANCE_FUNCTIONS(X) \
  X(DestroyInstance) \
  X(EnumeratePhysicalDevices) \
  X(GetPhysicalDeviceProperties) \
  X(GetPhysicalDeviceMemoryProperties) \
  X(GetPhysicalDeviceQueueFamilyProperties) \
  X(CreateDevice) \
  X(GetDeviceProcAddr) \
  X(CreateDebugUtilsMessengerEXT) \
  X(DestroyDebugUtilsMessengerEXT)

#define TENSORCX_VK_DEVICE_FUNCTIONS(X) \
  X(DestroyDevice) X(GetDeviceQueue) X(DeviceWaitIdle) \
  X(CreateCommandPool) X(DestroyCommandPool) \
  X(AllocateCommandBuffers) X(FreeCommandBuffers) \
  X(BeginCommandBuffer) X(EndCommandBuffer) X(QueueSubmit) \
  X(CreateFence) X(DestroyFence) X(WaitForFences) \
  X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements) \
  X(AllocateMemory) X(FreeMemory) X(BindBufferMemory) \
  X(MapMemory) X(UnmapMemory) \
  X(FlushMappedMemoryRanges) X(InvalidateMappedMemoryRanges) \
  X(CmdCopyBuffer) X(CmdPipelineBarrier) \
  X(CreateDescriptorSetLayout) X(DestroyDescriptorSetLayout) \
  X(CreatePipelineLayout) X(DestroyPipelineLayout) \
  X(CreateShaderModule) X(DestroyShaderModule) \
  X(CreateComputePipelines) X(DestroyPipeline) \
  X(CreateDescriptorPool) X(DestroyDescriptorPool) \
  X(AllocateDescriptorSets) X(UpdateDescriptorSets) \
  X(CmdBindPipeline) X(CmdBindDescriptorSets) X(CmdPushConstants) X(CmdDispatch)
