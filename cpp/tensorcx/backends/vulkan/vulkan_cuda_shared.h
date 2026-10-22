#pragma once

#include "tensorcx/backends/cuda/cuda_backend.h"
#include "tensorcx/backends/cuda/cuda_context.h"
#include "tensorcx/backends/vulkan/vulkan_tensor.h"

namespace tensorcx::vulkan {

enum class SharedOrigin { kVulkan, kCuda };
struct SharedTensors { VulkanTensor vulkan; cuda::CudaTensor cuda; };

// Native acceptance can supply a private function-table snapshot to inject
// failures without changing global CUDA entry points or another live allocation.
struct SharedCudaApi {
  decltype(&cudaDeviceSynchronize) synchronize = &cudaDeviceSynchronize;
  decltype(&cudaStreamCreateWithFlags) create_stream = &cudaStreamCreateWithFlags;
  decltype(&cudaStreamSynchronize) synchronize_stream = &cudaStreamSynchronize;
  decltype(&cudaStreamDestroy) destroy_stream = &cudaStreamDestroy;
  decltype(&cudaImportExternalMemory) import_memory = &cudaImportExternalMemory;
  decltype(&cudaExternalMemoryGetMappedBuffer) map_memory = &cudaExternalMemoryGetMappedBuffer;
  decltype(&cudaFree) free_mapping = &cudaFree;
  decltype(&cudaDestroyExternalMemory) destroy_memory = &cudaDestroyExternalMemory;
  decltype(&cudaImportExternalSemaphore) import_semaphore = &cudaImportExternalSemaphore;
  decltype(&cudaDestroyExternalSemaphore) destroy_semaphore = &cudaDestroyExternalSemaphore;
  decltype(&cudaSignalExternalSemaphoresAsync) signal = &cudaSignalExternalSemaphoresAsync;
  decltype(&cudaWaitExternalSemaphoresAsync) wait = &cudaWaitExternalSemaphoresAsync;
  decltype(&cuMemGetAllocationGranularity) granularity = &cuMemGetAllocationGranularity;
  decltype(&cuMemCreate) create = &cuMemCreate;
  decltype(&cuMemAddressReserve) reserve = &cuMemAddressReserve;
  decltype(&cuMemMap) map = &cuMemMap;
  decltype(&cuMemSetAccess) set_access = &cuMemSetAccess;
  decltype(&cuMemExportToShareableHandle) export_fd = &cuMemExportToShareableHandle;
  decltype(&cuMemUnmap) unmap = &cuMemUnmap;
  decltype(&cuMemAddressFree) free_address = &cuMemAddressFree;
  decltype(&cuMemRelease) release = &cuMemRelease;
};

Status shared_support(int index, SharedOrigin origin);
Expected<SharedTensors> shared_tensors(Shape shape, DType dtype, int index,
    SharedOrigin origin, const cpu::CpuTensor* initial = nullptr,
    const SharedCudaApi* api = nullptr);

}  // namespace tensorcx::vulkan
