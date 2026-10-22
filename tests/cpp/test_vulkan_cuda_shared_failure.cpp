#include <array>
#include <atomic>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>

#include "tensorcx/backends/cuda/cuda_buffer.h"
#include "tensorcx/backends/vulkan/vulkan_backend.h"
#include "tensorcx/backends/vulkan/vulkan_context.h"
#include "tensorcx/backends/vulkan/vulkan_cuda_shared.h"

using namespace tensorcx;
namespace vk = tensorcx::vulkan;
namespace {
void require(bool value, const char* text) { if (!value) throw std::runtime_error(text); }
bool fail_sync = false;
enum class HandoffFault { kNone, kSignal, kWait, kStreamWait };
HandoffFault handoff_fault = HandoffFault::kNone;
int sync_calls = 0;
cudaError_t synchronize() { ++sync_calls; return fail_sync ? cudaErrorUnknown : cudaDeviceSynchronize(); }
cudaError_t fail_import(cudaExternalMemory_t*, const cudaExternalMemoryHandleDesc*) { return cudaErrorUnknown; }
cudaError_t fail_mapping(void**, cudaExternalMemory_t, const cudaExternalMemoryBufferDesc*) { return cudaErrorUnknown; }
cudaError_t fail_stream(cudaStream_t*, unsigned) { return cudaErrorUnknown; }
cudaError_t fail_semaphore(cudaExternalSemaphore_t*, const cudaExternalSemaphoreHandleDesc*) { return cudaErrorUnknown; }
cudaError_t armed_signal(const cudaExternalSemaphore_t* handles, const cudaExternalSemaphoreSignalParams* params,
    unsigned count, cudaStream_t stream) {
  return handoff_fault == HandoffFault::kSignal ? cudaErrorUnknown
      : cudaSignalExternalSemaphoresAsync(handles, params, count, stream);
}
cudaError_t armed_wait(const cudaExternalSemaphore_t* handles, const cudaExternalSemaphoreWaitParams* params,
    unsigned count, cudaStream_t stream) {
  return handoff_fault == HandoffFault::kWait ? cudaErrorUnknown
      : cudaWaitExternalSemaphoresAsync(handles, params, count, stream);
}
cudaError_t armed_stream_wait(cudaStream_t stream) {
  return handoff_fault == HandoffFault::kStreamWait ? cudaErrorUnknown : cudaStreamSynchronize(stream);
}
cudaError_t fail_destroy(cudaExternalSemaphore_t) { return cudaErrorUnknown; }
CUresult fail_create(CUmemGenericAllocationHandle*, std::size_t, const CUmemAllocationProp*, unsigned long long) {
  return CUDA_ERROR_OUT_OF_MEMORY;
}
CUresult fail_reserve(CUdeviceptr*, std::size_t, std::size_t, CUdeviceptr, unsigned long long) {
  return CUDA_ERROR_OUT_OF_MEMORY;
}
CUresult fail_map(CUdeviceptr, std::size_t, std::size_t, CUmemGenericAllocationHandle, unsigned long long) {
  return CUDA_ERROR_OUT_OF_MEMORY;
}
CUresult fail_access(CUdeviceptr, std::size_t, const CUmemAccessDesc*, std::size_t) { return CUDA_ERROR_OUT_OF_MEMORY; }
CUresult fail_export(void*, CUmemGenericAllocationHandle, CUmemAllocationHandleType, unsigned long long) {
  return CUDA_ERROR_OUT_OF_MEMORY;
}
VKAPI_ATTR VkResult VKAPI_CALL fail_end(VkCommandBuffer) { return VK_ERROR_OUT_OF_HOST_MEMORY; }
VKAPI_ATTR VkResult VKAPI_CALL fail_fence(VkDevice, std::uint32_t, const VkFence*, VkBool32, std::uint64_t) {
  return VK_ERROR_OUT_OF_HOST_MEMORY;
}
VKAPI_ATTR VkResult VKAPI_CALL fail_idle(VkDevice) { return VK_ERROR_OUT_OF_HOST_MEMORY; }
std::size_t fd_count() {
  return static_cast<std::size_t>(std::distance(std::filesystem::directory_iterator("/proc/self/fd"),
                                               std::filesystem::directory_iterator{}));
}
vk::SharedTensors create(vk::SharedOrigin origin, const vk::SharedCudaApi* api = nullptr) {
  cpu::CpuTensor host({3}, std::vector<float>{1, 2, 3});
  auto value = vk::shared_tensors({3}, DType::kFloat32, 0, origin, &host, api);
  if (!value) throw std::runtime_error(value.status().message());
  return value.move_value();
}
void initialization(vk::SharedOrigin origin) {
  auto context = vk::get_context(0).value();
  // Warm up CUDA's process-global resources before counting OS descriptors.
  { auto warm = create(origin); }
  const auto baseline = context->statistics();
  const auto count = context->allocation_count.load();
  const auto descriptors = fd_count();
  const int cases = origin == vk::SharedOrigin::kVulkan ? 4 : 7;
  for (int fault = 0; fault < cases; ++fault) {
    vk::SharedCudaApi api;
    if (origin == vk::SharedOrigin::kVulkan) {
      if (fault == 0) api.import_memory = fail_import;
      if (fault == 1) api.map_memory = fail_mapping;
      if (fault == 2) api.create_stream = fail_stream;
      if (fault == 3) api.import_semaphore = fail_semaphore;
    } else {
      if (fault == 0) api.create = fail_create;
      if (fault == 1) api.reserve = fail_reserve;
      if (fault == 2) api.map = fail_map;
      if (fault == 3) api.set_access = fail_access;
      if (fault == 4) api.export_fd = fail_export;
      if (fault == 5) api.create_stream = fail_stream;
      if (fault == 6) api.import_semaphore = fail_semaphore;
    }
    auto rejected = vk::shared_tensors({3}, DType::kFloat32, 0, origin, nullptr, &api);
    require(!rejected, "shared initialization fault was swallowed");
    const auto now = context->statistics();
    require(now.physical_blocks == baseline.physical_blocks && now.physical_bytes == baseline.physical_bytes &&
            context->allocation_count.load() == count, "failed initialization leaked physical storage");
    require(fd_count() == descriptors, "failed initialization leaked an exported FD");
    require(!context->poisoned, "allocation failure poisoned ordinary Vulkan execution");
    { auto retry = create(origin); require(cuda::to_cpu(retry.cuda).value().float_data()[2] == 3, "retry failed"); }
  }
}
void run(std::string_view mode, vk::SharedOrigin origin) {
  if (mode == "initialization") { initialization(origin); return; }
  auto context = vk::get_context(0).value();
  const auto count = context->allocation_count.load();
  const auto before = context->statistics();
  vk::SharedCudaApi api; api.synchronize = synchronize;
  api.signal = armed_signal; api.wait = armed_wait; api.synchronize_stream = armed_stream_wait;
  auto pair_storage = std::make_unique<vk::SharedTensors>(create(origin, &api));
  auto& pair = *pair_storage;
  auto access = pair.cuda.buffer()->access();
  if (mode == "fork") {
    std::atomic<bool> locked{false}, release{false};
    std::thread parent_owner([&] {
      std::lock_guard lock(access->mutex); locked = true;
      while (!release.load()) std::this_thread::yield();
    });
    while (!locked.load()) std::this_thread::yield();
    const auto child = ::fork();
    if (child == 0) {
      ::alarm(5);
      const bool rejected = !cuda::current_process() && !context->current_process() &&
          !cuda::to_cpu(pair.cuda) && !vk::to_cpu(pair.vulkan) &&
          !vk::shared_support(0, origin).ok();
      ::_exit(rejected ? 0 : 1);
    }
    release = true; parent_owner.join();
    require(child > 0, "fork failed");
    int result = 0; require(::waitpid(child, &result, 0) == child, "child wait failed");
    require(WIFEXITED(result) && WEXITSTATUS(result) == 0, "fork child accessed native state or inherited locked mutex");
    require(cuda::to_cpu(pair.cuda).value().float_data()[0] == 1 &&
            vk::to_cpu(pair.vulkan).value().float_data()[2] == 3, "fork child damaged parent allocation");
    return;
  }
  if (mode == "foreign_release") {
    cuda::ContextScope scope; require(scope.ready(), "foreign context unavailable");
    cudaStream_t stream{}; cudaEvent_t event{};
    require(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess, "foreign stream creation failed");
    require(cudaEventCreateWithFlags(&event, cudaEventDisableTiming) == cudaSuccess, "foreign event creation failed");
    require(cudaMemsetAsync(pair.cuda.buffer()->data(), 0, pair.cuda.nbytes(), stream) == cudaSuccess, "foreign write failed");
    require(cudaEventRecord(event, stream) == cudaSuccess, "foreign event failed");
    auto foreign = std::make_unique<vk::SharedTensors>(create(origin, &api));
    auto* pointer = foreign->cuda.buffer()->data();
    require(cudaMemsetAsync(pointer, 0, foreign->cuda.nbytes(), stream) == cudaSuccess, "foreign final write failed");
    require(cudaEventRecord(event, stream) == cudaSuccess, "foreign final event failed");
    const auto calls = sync_calls;
    foreign.reset();
    require(sync_calls > calls && cudaEventQuery(event) == cudaSuccess, "last-owner cleanup did not prove foreign completion");
    require(cudaEventDestroy(event) == cudaSuccess && cudaStreamDestroy(stream) == cudaSuccess, "foreign resource release failed");
    return;
  }
  if (mode == "record") {
    const auto real = context->EndCommandBuffer; context->EndCommandBuffer = fail_end;
    require(!vk::to_cpu(pair.vulkan), "failed command encoding was accepted");
    context->EndCommandBuffer = real;
    require(!context->poisoned && access->validate().ok(), "pre-submit cancellation poisoned shared storage");
    require(vk::to_cpu(pair.vulkan).value().float_data()[2] == 3, "cancelled semaphore was not reusable");
    return;
  }
  if (mode == "final_sync" || mode == "final_destroy") {
    // Construct the owner with a private failure table; no global driver change.
    pair = create(origin, &api);
    vk::SharedCudaApi bad = api;
    if (mode == "final_destroy") bad.destroy_semaphore = fail_destroy;
    auto doomed = create(origin, &bad);
    const auto live = context->statistics();
    fail_sync = mode == "final_sync";
    { auto last = std::move(doomed); }
    fail_sync = false;
    const auto retained = context->statistics();
    require(retained.physical_blocks == live.physical_blocks && retained.physical_bytes == live.physical_bytes,
            "unproven final cleanup freed shared Vulkan physical memory");
    return;  // Intentional quarantine belongs to this isolated process.
  }
  if (mode == "cuda_signal" || mode == "cuda_wait" || mode == "cuda_stream_wait") {
    if (mode == "cuda_signal") handoff_fault = HandoffFault::kSignal;
    if (mode == "cuda_wait") handoff_fault = HandoffFault::kWait;
    if (mode == "cuda_stream_wait") handoff_fault = HandoffFault::kStreamWait;
    auto sentinel = vk::to_core_tensor(vk::empty({1}, DType::kFloat32).move_value());
    std::array<Tensor, 1> outputs{sentinel};
    const std::array<Tensor, 1> inputs{vk::to_core_tensor(pair.vulkan)};
    BackendExecution execution; execution.op.kind = OpKind::kAddScalar; execution.op.scalar_value = 1;
    execution.inputs = inputs; execution.outputs = outputs;
    vk::VulkanBackend backend;
    require(!backend.execute(execution).ok(), "CUDA handoff failure was swallowed");
    handoff_fault = HandoffFault::kNone;
    require(outputs[0].buffer == sentinel.buffer, "CUDA handoff failure published a Vulkan result");
    require(!access->validate().ok() && !vk::to_cpu(pair.vulkan) && !cuda::to_cpu(pair.cuda),
            "handoff failure left a shared alias usable");
    return;
  }
  if (mode == "cuda_completion") {
    auto sentinel = cuda::to_core_tensor(cuda::empty({1}, DType::kFloat32).move_value());
    std::array<Tensor, 1> outputs{sentinel};
    const std::array<Tensor, 1> inputs{cuda::to_core_tensor(pair.cuda)};
    BackendExecution execution; execution.op.kind = OpKind::kAddScalar; execution.op.scalar_value = 1;
    execution.inputs = inputs; execution.outputs = outputs;
    fail_sync = true;
    cuda::CudaBackend backend;
    require(!backend.execute(execution).ok(), "CUDA completion fault was swallowed");
    require(outputs[0].buffer == sentinel.buffer, "CUDA completion failure published a result");
    fail_sync = false;
    require(!access->validate().ok() && !vk::to_cpu(pair.vulkan) && !cuda::to_cpu(pair.cuda),
            "poisoned aliases remained usable after failed completion");
    return;
  }
  if (mode == "vk_wait" || mode == "vk_unproven") {
    const auto wait = context->WaitForFences; const auto idle = context->DeviceWaitIdle;
    context->WaitForFences = fail_fence;
    if (mode == "vk_unproven") context->DeviceWaitIdle = fail_idle;
    require(!vk::to_cpu(pair.vulkan), "failed Vulkan completion was accepted");
    context->WaitForFences = wait; context->DeviceWaitIdle = idle;
    require(context->poisoned && !access->validate().ok(), "Vulkan completion failure did not poison both aliases");
    // Remove every caller reference before inspecting the completion-dependent
    // physical lifetime. Only the failed submission may retain the owner now.
    pair_storage.reset(); access.reset();
    const auto after = context->statistics();
    if (mode == "vk_unproven")
      require(context->allocation_count.load() > count && after.physical_blocks > before.physical_blocks,
              "unproven Vulkan completion did not retain last-owner shared storage");
    else require(context->allocation_count.load() == count && after.physical_blocks == before.physical_blocks &&
                 after.physical_bytes == before.physical_bytes,
                 "proven idle did not release last-owner shared storage");
    return;
  }
  throw std::runtime_error("unknown shared fault mode");
}
}  // namespace
int main(int argc, char** argv) {
  try {
    require(argc == 3, "expected fault mode and origin");
    const auto origin = std::string_view(argv[2]) == "cuda" ? vk::SharedOrigin::kCuda : vk::SharedOrigin::kVulkan;
    auto support = vk::shared_support(0, origin);
    if (!support.ok()) { std::cerr << support.message() << '\n'; return 77; }
    run(argv[1], origin);
    require(vk::validation_errors() == 0, "shared failure contract triggered Vulkan validation errors");
    std::cout << "shared fault contract passed: " << argv[1] << '\n'; return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
