#include "tensorcx/backends/vulkan/vulkan_cuda_shared.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unistd.h>

#include "tensorcx/backends/cuda/cuda_buffer.h"
#include "tensorcx/backends/cuda/cuda_external_storage.h"
#include "tensorcx/backends/vulkan/vulkan_context.h"
#include "tensorcx/core/profiling.h"

namespace tensorcx::vulkan {
namespace {
Status invalid(const char* message) { return {StatusCode::kInvalidArgument, message}; }
void vk_check(VkResult result, const char* name) {
  if (result != VK_SUCCESS) throw VulkanError(result, name);
}
void cuda_check(cudaError_t result, const char* name) {
  auto status = cuda::runtime_status(result, name);
  if (!status.ok()) throw std::runtime_error(status.message());
}
void driver_check(CUresult result, const char* name) {
  auto status = cuda::driver_status(result, name);
  if (!status.ok()) throw std::runtime_error(status.message());
}
void scope_check(const cuda::ContextScope& scope) {
  if (!scope.ready()) throw std::runtime_error(scope.status().message());
}
template <typename T> T vk_info(VkStructureType type) { T value{}; value.sType = type; return value; }
}  // namespace

// Sole physical owner for both origins. Allocation/CudaBuffer wrappers retain
// this object; this object never owns those wrappers. A custom final deleter
// retains the entire owner, both contexts and native handles on unproven cleanup.
struct SharedStorage final : ExternalAccess, cuda::ExternalStorage,
                             std::enable_shared_from_this<SharedStorage> {
  std::shared_ptr<Context> vk;
  std::shared_ptr<cuda::PrimaryContext> primary;
  SharedCudaApi api;
  SharedOrigin origin;
  std::uint64_t process = static_cast<std::uint64_t>(::getpid());
  std::size_t logical = 0, capacity = 0, memory_bytes = 0, vmm_bytes = 0;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  std::array<VkSemaphore, 2> semaphores{};  // Vulkan->CUDA, CUDA->Vulkan.
  std::array<cudaExternalSemaphore_t, 2> cuda_semaphores{};
  cudaExternalMemory_t cuda_memory{};
  cudaStream_t stream{};
  void* pointer = nullptr;
  CUmemGenericAllocationHandle allocation{};
  CUdeviceptr address{};
  bool mapped_vmm = false, reserved_vk = false, counted_vk = false;
  int pending_fd = -1;
  std::atomic<bool> failed{false};
  enum class State { kInitialVulkan, kCuda, kPrepared, kPreparedInitial };
  State state = State::kInitialVulkan;  // Protected by StorageAccess::mutex.

  SharedStorage(std::shared_ptr<Context> context, std::shared_ptr<cuda::PrimaryContext> cuda_context,
      SharedOrigin memory_origin, std::size_t bytes, const SharedCudaApi& functions)
      : vk(std::move(context)), primary(std::move(cuda_context)), api(functions), origin(memory_origin),
        logical(bytes), capacity((bytes + 3u) & ~std::size_t{3}) {
    if (origin == SharedOrigin::kCuda || !logical) state = State::kCuda;
  }

  Status validate() const override {
    if (process != static_cast<std::uint64_t>(::getpid()) || !vk->current_process() || !cuda::current_process())
      return {StatusCode::kUnavailable, "shared storage requires a fresh process after fork"};
    if (failed.load() || vk->poisoned.load())
      return {StatusCode::kUnavailable, "shared storage is poisoned after a native completion error"};
    return Status::Ok();
  }
  std::size_t logical_bytes() const override { return logical; }
  std::shared_ptr<StorageAccess> access() override { return shared_from_this(); }
  std::uintptr_t begin() const override { return reinterpret_cast<std::uintptr_t>(pointer); }
  std::size_t bytes() const override { return logical; }
  std::size_t mapped_bytes() const override { return origin == SharedOrigin::kCuda ? vmm_bytes : capacity; }
  StorageIdentity identity(void* data, std::size_t bytes) const override {
    return {static_cast<const StorageAccess*>(this), data ? reinterpret_cast<std::uintptr_t>(data) - begin() : 0, bytes};
  }
  Status validate_range(void* data, DType dtype, std::size_t bytes) const override {
    if (auto status = validate(); !status.ok()) return status;
    if (dtype != DType::kFloat32 && dtype != DType::kInt32 && dtype != DType::kBool)
      return invalid("unsupported shared storage dtype");
    const auto start = reinterpret_cast<std::uintptr_t>(data);
    if ((!data && bytes) || start < begin() || start - begin() > logical || bytes > logical - (start - begin()))
      return invalid("shared CUDA view exceeds its logical storage range");
    if (start % dtype_size(dtype)) return invalid("misaligned shared CUDA view");
    return Status::Ok();
  }
  void poison() noexcept override { failed.store(true); }

  ExternalSubmission prepare() override {
    if (auto status = validate(); !status.ok()) throw std::runtime_error(status.message());
    if (state == State::kInitialVulkan) {
      state = State::kPreparedInitial;
      return {buffer, VK_NULL_HANDLE, semaphores[0], false};
    }
    if (state != State::kCuda) throw std::runtime_error("invalid shared storage ownership transition");
    try {
      ProfileScope profile(ProfileCategory::kSynchronize, "shared_cuda_release", "cuda");
      cuda::ContextScope scope(primary); scope_check(scope);
      cuda_check(api.synchronize(), "shared CUDA producer completion");
      cudaExternalSemaphoreSignalParams parameters{};
      cuda_check(api.signal(&cuda_semaphores[1], &parameters, 1, stream), "shared CUDA external signal");
      state = State::kPrepared;
      return {buffer, semaphores[1], semaphores[0], true};
    } catch (...) { poison(); throw; }
  }
  void complete() override {
    try {
      ProfileScope profile(ProfileCategory::kSynchronize, "shared_cuda_acquire", "cuda");
      cuda::ContextScope scope(primary); scope_check(scope);
      cudaExternalSemaphoreWaitParams parameters{};
      cuda_check(api.wait(&cuda_semaphores[0], &parameters, 1, stream), "shared CUDA external wait");
      cuda_check(api.synchronize_stream(stream), "shared CUDA external wait completion");
      state = State::kCuda;
    } catch (...) { poison(); throw; }
  }
  void cancel() noexcept override {
    if (failed.load()) return;
    if (state == State::kPreparedInitial) { state = State::kInitialVulkan; return; }
    if (state != State::kPrepared) return;
    try {
      cuda::ContextScope scope(primary); scope_check(scope);
      // Encoding failed before vkQueueSubmit: consume our own queued signal.
      cudaExternalSemaphoreWaitParams parameters{};
      cuda_check(api.wait(&cuda_semaphores[1], &parameters, 1, stream), "shared signal cancellation");
      cuda_check(api.synchronize_stream(stream), "shared signal cancellation completion");
      state = State::kCuda;
    } catch (...) { poison(); }
  }
  Status complete_cuda() override {
    if (process != static_cast<std::uint64_t>(::getpid()) || !cuda::current_process())
      return {StatusCode::kUnavailable, "shared CUDA completion requires a fresh process after fork"};
    if (logical) {
      ProfileScope profile(ProfileCategory::kSynchronize, "shared_cuda_access_complete", "cuda");
      cuda::ContextScope scope(primary);
      if (!scope.ready()) { poison(); return scope.status(); }
      auto result = cuda::runtime_status(api.synchronize(), "shared CUDA access completion");
      if (!result.ok()) { poison(); return result; }
    }
    return validate();
  }

  void initialize() {
    if (!logical) return;
    cuda::ContextScope scope(primary); scope_check(scope);
    auto create_buffer = vk_info<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
    auto external_buffer = vk_info<VkExternalMemoryBufferCreateInfo>(VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO);
    external_buffer.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    create_buffer.pNext = &external_buffer; create_buffer.size = capacity;
    create_buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    create_buffer.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vk_check(vk->CreateBuffer(vk->device, &create_buffer, nullptr, &buffer), "vkCreateBuffer(shared)");
    VkMemoryRequirements requirements{}; vk->GetBufferMemoryRequirements(vk->device, buffer, &requirements);
    std::uint32_t type = 0;
    while (type < vk->memory_properties.memoryTypeCount && (!(requirements.memoryTypeBits & (1u << type)) ||
        !(vk->memory_properties.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))) ++type;
    if (type == vk->memory_properties.memoryTypeCount) throw std::runtime_error("no compatible shared device-local memory type");
    auto allocate = vk_info<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
    allocate.memoryTypeIndex = type; allocate.allocationSize = requirements.size;
    auto dedicated = vk_info<VkMemoryDedicatedAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO);
    dedicated.buffer = buffer;
    auto export_memory = vk_info<VkExportMemoryAllocateInfo>(VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO);
    export_memory.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT; export_memory.pNext = &dedicated;
    auto import_memory = vk_info<VkImportMemoryFdInfoKHR>(VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR);
    if (origin == SharedOrigin::kVulkan) allocate.pNext = &export_memory;
    else {
      CUmemAllocationProp properties{}; properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      properties.location = {CU_MEM_LOCATION_TYPE_DEVICE, 0};
      properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
      std::size_t granularity = 0;
      driver_check(api.granularity(&granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "shared VMM granularity");
      if (!granularity || requirements.size > std::numeric_limits<std::size_t>::max() - (granularity - 1))
        throw std::invalid_argument("invalid shared VMM allocation granularity or size");
      vmm_bytes = ((static_cast<std::size_t>(requirements.size) + granularity - 1) / granularity) * granularity;
      driver_check(api.create(&allocation, vmm_bytes, &properties, 0), "shared VMM allocation");
      driver_check(api.reserve(&address, vmm_bytes, granularity, 0, 0), "shared VMM address reservation");
      driver_check(api.map(address, vmm_bytes, 0, allocation, 0), "shared VMM mapping"); mapped_vmm = true;
      CUmemAccessDesc access{}; access.location = properties.location; access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      driver_check(api.set_access(address, vmm_bytes, &access, 1), "shared VMM access");
      pointer = reinterpret_cast<void*>(address);
      driver_check(api.export_fd(&pending_fd, allocation, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0), "shared VMM export");
      import_memory.fd = pending_fd; import_memory.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
      allocate.pNext = &import_memory; allocate.allocationSize = vmm_bytes;
    }
    const auto heap = vk->memory_properties.memoryTypes[type].heapIndex;
    if (allocate.allocationSize > vk->memory_properties.memoryHeaps[heap].size)
      throw std::invalid_argument("shared allocation exceeds the device memory heap");
    auto live = vk->allocation_count.load();
    do {
      if (live >= vk->limits.maxMemoryAllocationCount) throw std::runtime_error("shared Vulkan allocation count limit reached");
    } while (!vk->allocation_count.compare_exchange_weak(live, live + 1));
    reserved_vk = true; vk->count(&ContextStats::allocation_calls);
    vk_check(vk->AllocateMemory(vk->device, &allocate, nullptr, &memory), "vkAllocateMemory(shared)");
    if (origin == SharedOrigin::kCuda) pending_fd = -1;  // Successful Vulkan import consumes the FD.
    memory_bytes = static_cast<std::size_t>(allocate.allocationSize);
    {
      std::lock_guard lock(vk->stats_mutex_);
      ++vk->stats_.physical_blocks; vk->stats_.physical_bytes += memory_bytes;
      vk->stats_.peak_physical_bytes = std::max(vk->stats_.peak_physical_bytes, vk->stats_.physical_bytes);
    }
    counted_vk = true;
    vk_check(vk->BindBufferMemory(vk->device, buffer, memory, 0), "vkBindBufferMemory(shared)");
    if (origin == SharedOrigin::kVulkan) {
      auto handle = vk_info<VkMemoryGetFdInfoKHR>(VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR);
      handle.memory = memory; handle.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
      vk_check(vk->GetMemoryFdKHR(vk->device, &handle, &pending_fd), "vkGetMemoryFdKHR");
      cudaExternalMemoryHandleDesc import{}; import.type = cudaExternalMemoryHandleTypeOpaqueFd;
      import.handle.fd = pending_fd; import.size = memory_bytes; import.flags = cudaExternalMemoryDedicated;
      cuda_check(api.import_memory(&cuda_memory, &import), "CUDA shared memory import"); pending_fd = -1;
      cudaExternalMemoryBufferDesc view{}; view.size = capacity;
      cuda_check(api.map_memory(&pointer, cuda_memory, &view), "CUDA shared memory mapping");
    }
    cuda_check(api.create_stream(&stream, cudaStreamNonBlocking), "shared CUDA stream creation");
    for (std::size_t i = 0; i < semaphores.size(); ++i) {
      auto export_sem = vk_info<VkExportSemaphoreCreateInfo>(VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO);
      export_sem.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
      auto create = vk_info<VkSemaphoreCreateInfo>(VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO); create.pNext = &export_sem;
      vk_check(vk->CreateSemaphore(vk->device, &create, nullptr, &semaphores[i]), "vkCreateSemaphore(shared)");
      auto handle = vk_info<VkSemaphoreGetFdInfoKHR>(VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR);
      handle.semaphore = semaphores[i]; handle.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
      vk_check(vk->GetSemaphoreFdKHR(vk->device, &handle, &pending_fd), "vkGetSemaphoreFdKHR");
      cudaExternalSemaphoreHandleDesc import{}; import.type = cudaExternalSemaphoreHandleTypeOpaqueFd; import.handle.fd = pending_fd;
      cuda_check(api.import_semaphore(&cuda_semaphores[i], &import), "CUDA shared semaphore import"); pending_fd = -1;
    }
  }

  std::shared_ptr<Allocation> view() {
    auto allocation = std::make_unique<Allocation>();
    allocation->owner = vk; allocation->buffer = buffer; allocation->memory = memory;
    allocation->capacity = capacity; allocation->requested_bytes = logical; allocation->kind = AllocationKind::kTensor;
    allocation->external = shared_from_this();
    if (logical) vk->lease_change(*allocation, true);
    return std::shared_ptr<Allocation>(allocation.release(), [](Allocation* allocation) noexcept {
      if (!allocation->owner->current_process()) return;
      if (allocation->leased) allocation->owner->lease_change(*allocation, false);
      delete allocation;
    });
  }

  bool try_release() noexcept {
    if (process != static_cast<std::uint64_t>(::getpid()) || !vk->current_process() || !cuda::current_process()) return false;
    try {
      std::lock_guard lock(mutex);
      // No Vulkan work can reference the physical owner without a Frame retaining
      // it. Failed/unproven Vulkan waits retain that Frame in quarantine.
      cuda::ContextScope scope(primary); if (!scope.ready()) return false;
      if (logical && api.synchronize() != cudaSuccess) return false;
      if (pending_fd >= 0) { (void)::close(pending_fd); pending_fd = -1; }
      for (auto& semaphore : cuda_semaphores) {
        if (semaphore && api.destroy_semaphore(semaphore) != cudaSuccess) return false;
        semaphore = nullptr;
      }
      if (stream && api.destroy_stream(stream) != cudaSuccess) return false;
      stream = nullptr;
      if (origin == SharedOrigin::kVulkan) {
        if (pointer && api.free_mapping(pointer) != cudaSuccess) return false;
        pointer = nullptr;
        if (cuda_memory && api.destroy_memory(cuda_memory) != cudaSuccess) return false;
        cuda_memory = nullptr;
      }
      for (auto& semaphore : semaphores) {
        if (semaphore) vk->DestroySemaphore(vk->device, semaphore, nullptr);
        semaphore = VK_NULL_HANDLE;
      }
      if (buffer) vk->DestroyBuffer(vk->device, buffer, nullptr);
      buffer = VK_NULL_HANDLE;
      if (memory) vk->FreeMemory(vk->device, memory, nullptr);
      memory = VK_NULL_HANDLE;
      if (counted_vk) {
        std::lock_guard stats_lock(vk->stats_mutex_);
        --vk->stats_.physical_blocks; vk->stats_.physical_bytes -= memory_bytes; ++vk->stats_.free_calls;
        counted_vk = false;
      }
      if (reserved_vk) { --vk->allocation_count; reserved_vk = false; }
      if (mapped_vmm && api.unmap(address, vmm_bytes) != CUDA_SUCCESS) return false;
      mapped_vmm = false; pointer = nullptr;
      if (address && api.free_address(address, vmm_bytes) != CUDA_SUCCESS) return false;
      address = 0;
      if (allocation && api.release(allocation) != CUDA_SUCCESS) return false;
      allocation = 0;
      return true;
    } catch (...) { return false; }
  }
};

Status shared_support(int index, SharedOrigin origin) {
  if (origin != SharedOrigin::kVulkan && origin != SharedOrigin::kCuda) return invalid("unsupported shared allocation origin");
  auto context = get_context(index); if (!context) return context.status();
  const auto& vk = *context.value();
  const auto needed = origin == SharedOrigin::kVulkan ? VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT
                                                     : VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT;
  if (!(vk.external_memory & needed) || !(vk.external_semaphore & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT) ||
      !vk.GetMemoryFdKHR || !vk.GetSemaphoreFdKHR || !vk.CreateSemaphore || !vk.DestroySemaphore)
    return {StatusCode::kUnavailable, "Vulkan opaque-FD memory/semaphore sharing is unavailable"};
  if (origin == SharedOrigin::kCuda && (vk.external_memory & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT))
    return {StatusCode::kUnavailable, "CUDA-origin sharing requires non-dedicated Vulkan import support"};
  cuda::ContextScope scope; if (!scope.ready()) return scope.status();
  cudaDeviceProp info{};
  if (auto s = cuda::runtime_status(cudaGetDeviceProperties(&info, 0), "shared CUDA device properties"); !s.ok()) return s;
  if (std::memcmp(info.uuid.bytes, vk.device_uuid.data(), VK_UUID_SIZE))
    return {StatusCode::kUnavailable, "CUDA and Vulkan sharing requires the same physical GPU"};
  if (origin == SharedOrigin::kCuda) {
    for (auto attribute : {CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                           CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED}) {
      int supported = 0;
      if (auto s = cuda::driver_status(cuDeviceGetAttribute(&supported, attribute, 0), "shared CUDA VMM capability"); !s.ok()) return s;
      if (!supported) return {StatusCode::kUnavailable, "CUDA VMM POSIX-FD sharing is unavailable"};
    }
  }
  return Status::Ok();
}

Expected<SharedTensors> shared_tensors(Shape shape, DType dtype, int index,
    SharedOrigin origin, const cpu::CpuTensor* initial, const SharedCudaApi* functions) {
  try {
    if (dtype != DType::kFloat32 && dtype != DType::kInt32 && dtype != DType::kBool) return invalid("unsupported shared dtype");
    const auto count = static_cast<std::uint64_t>(numel(shape)); (void)contiguous_strides(shape);
    if (count > (std::numeric_limits<std::size_t>::max() - 3u) / dtype_size(dtype)) return invalid("shared tensor size overflow");
    const auto bytes = static_cast<std::size_t>(count) * dtype_size(dtype);
    if (initial && (initial->dtype() != dtype || initial->shape() != shape)) return invalid("shared initializer dtype/shape mismatch");
    if (auto status = shared_support(index, origin); !status.ok()) return status;
    auto context = get_context(index); if (!context) return context.status();
    if (((bytes + 3u) & ~std::size_t{3}) > context.value()->limits.maxStorageBufferRange || bytes > UINT32_MAX - 3u)
      return invalid("shared tensor exceeds the Vulkan storage-buffer limit");
    cuda::ContextScope scope; if (!scope.ready()) return scope.status();
    const SharedCudaApi api = functions ? *functions : SharedCudaApi{};
    auto owner = std::shared_ptr<SharedStorage>(new SharedStorage(context.value(), scope.owner(), origin, bytes, api),
        [](SharedStorage* storage) noexcept {
          cuda::unregister_external_storage(storage->begin(), storage);
          if (storage->try_release()) delete storage;
          // Otherwise retain the entire physical allocation and both contexts.
        });
    try { owner->initialize(); }
    catch (...) { context.value()->count(&ContextStats::allocation_failures); throw; }
    auto allocation = owner->view();
    auto buffer = std::make_shared<VulkanBuffer>(dtype, bytes, context.value(), allocation);
    VulkanTensor vulkan(dtype, shape, std::move(buffer));
    if (initial && bytes && origin == SharedOrigin::kVulkan) {
      context.value()->upload(allocation, initial->data(), bytes);
    } else {
      if (initial && bytes) {
        cuda_check(cudaMemcpy(owner->pointer, initial->data(), bytes, cudaMemcpyHostToDevice), "shared CUDA initializer upload");
        cuda_check(api.synchronize(), "shared CUDA initializer completion");
      }
      context.value()->synchronize_external(allocation);
    }
    auto cuda_buffer = cuda::CudaBuffer::borrow(dtype, shape, owner->pointer, owner, owner);
    if (!cuda_buffer) return cuda_buffer.status();
    cuda::register_external_storage(owner);
    return SharedTensors{std::move(vulkan), cuda::CudaTensor(dtype, std::move(shape), cuda_buffer.move_value())};
  } catch (const std::invalid_argument& error) { return Status(StatusCode::kInvalidArgument, error.what()); }
    catch (const std::exception& error) { return Status(StatusCode::kInternal, error.what()); }
}

}  // namespace tensorcx::vulkan
