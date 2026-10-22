#include "tensorcx/backends/vulkan/vulkan_kernel.h"
#include "tensorcx/backends/vulkan/vulkan_memory_pool.h"
#include "tensorcx/core/profiling.h"

#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>

using namespace tensorcx;
namespace vk = tensorcx::vulkan;
namespace {
void require(bool value, const char* text) { if (!value) throw std::runtime_error(text); }
vk::KernelSignature signature{"tttu", 2, 3};
std::string artifact(const char* path) {
  std::ifstream stream(path, std::ios::binary);
  std::string data{std::istreambuf_iterator<char>(stream), {}};
  require(data.size() >= 20 && data.size() % 4 == 0, "missing compiled Vulkan fixture");
  std::vector<std::uint32_t> input(data.size() / 4), words;
  std::memcpy(input.data(), data.data(), data.size());
  words.assign(input.begin(), input.begin() + 5);
  for (std::size_t i = 5; i < input.size();) {
    const auto count = input[i] >> 16, opcode = input[i] & 65535;
    require(count && count <= input.size() - i, "bad fixture instruction");
    // The generic GLSL compiler emits an unused fixed WorkgroupSize constant.
    // This fixture, like the real MLIR shader, uses push geometry and LocalSize.
    if (!(opcode == 71 && count == 4 && input[i + 2] == 11 && input[i + 3] == 25))
      words.insert(words.end(), input.begin() + i, input.begin() + i + count);
    i += count;
  }
  return vk::kernel_manifest(signature) + "\n" + std::string(reinterpret_cast<const char*>(words.data()), words.size() * 4);
}
std::vector<Tensor> tensors(std::int64_t n) {
  std::vector<Tensor> values;
  for (float value : {1.0F, 2.0F, -7.0F}) {
    auto tensor = vk::from_cpu(cpu::CpuTensor({n}, std::vector<float>(n, value))).move_value();
    values.push_back(vk::to_core_tensor(tensor));
  }
  return values;
}
std::array<KernelArgument, 4> arguments(std::vector<Tensor>& values, std::uint32_t threads) {
  return {{{KernelArgumentKind::kTensor, &values[0], 0}, {KernelArgumentKind::kTensor, &values[1], 0},
           {KernelArgumentKind::kTensor, &values[2], 0}, {KernelArgumentKind::kUInt32, nullptr, threads}}};
}
void expect_values(const vk::VulkanTensor& result, std::size_t prefix) {
  auto cpu = vk::to_cpu(result).move_value();
  for (std::int64_t i = 0; i < numel(cpu.shape()); ++i)
    require(cpu.float_data()[i] == (static_cast<std::size_t>(i) < prefix ? 3.0F : -7.0F), "private prefix/suffix mismatch");
}
void contracts(const std::string& binary) {
  auto loaded = vk::VulkanKernelModule::load(binary, signature, 0);
  if (!loaded) throw std::runtime_error(loaded.status().message());
  auto module = loaded.move_value();
  auto values = tensors(17); auto args = arguments(values, 17);
  const auto token = begin_profile(10000, true);
  auto pool = std::make_shared<vk::MemoryPool>(0, 8192, 32); require(pool->enter().ok(), "pool entry failed");
  for (auto block : {1u, 7u, 64u, 256u}) {
    for (auto count : {0u, 5u, 17u}) {
      args[3].uint32_value = count;
      auto result = vk::launch_compiled_kernel(module, args, count, block);
      if (!result) throw std::runtime_error(result.status().message());
      expect_values(result.value(), count);
    }
  }
  require(pool->stats().frame_hits > 0, "compiled kernels did not reuse command frames");
  require(pool->exit().ok(), "pool exit failed");
  const auto report = end_profile(token);
  require(!report.device_events.empty(), "compiled kernel GPU timing missing");
  for (const auto& event : report.device_events) {
    if (vk::get_context(0).value()->info.timestamp_bits)
      require(!event.error && event.elapsed_ms && *event.elapsed_ms >= 0, "compiled kernel GPU timing invalid");
    else require(event.error && !event.elapsed_ms, "unsupported queue reported successful GPU timing");
  }
  expect_values(vk::from_core_tensor(values[2]).value(), 0);
  for (auto block : {0u, UINT32_MAX}) {
    const auto before = vk::get_context(0).value()->allocation_count.load();
    auto result = vk::launch_compiled_kernel(module, args, 17, block);
    require(!result && result.status().code() == StatusCode::kInvalidArgument, "bad block accepted");
    require(vk::get_context(0).value()->allocation_count == before, "bad block allocated output");
  }
  // Force a real two-dimensional Vulkan grid with one logical dimension.
  auto large = tensors(65539); auto large_args = arguments(large, 65539);
  auto tiled = vk::launch_compiled_kernel(module, large_args, 65539, 1);
  if (!tiled) throw std::runtime_error(tiled.status().message());
  expect_values(tiled.value(), 65539);
  const auto id = module->artifact_id(); module.reset();
  std::array<Tensor, 1> out{values[2]};
  BackendExecution execution{BackendOpClass::kKernel, {}, {}, out, LaunchConfig{17,1,1,1,1,1},
      CompilationTarget{KernelArtifactKind::kBinary, id, "tensorcx_kernel_v1"}, args};
  vk::VulkanBackend backend;
  require(!backend.execute(execution).ok() && out[0].buffer == values[2].buffer,
          "expired artifact was usable or changed output");
}
void fork_rejection(const std::string& binary) {
  auto loaded = vk::VulkanKernelModule::load(binary, signature, 0);
  if (!loaded) throw std::runtime_error(loaded.status().message());
  auto module = loaded.move_value();
  auto values = tensors(17); auto args = arguments(values, 17);
  const auto child = ::fork();
  require(child >= 0, "fork failed");
  if (!child) {
    ::alarm(5); // A regression must fail instead of hanging on inherited locks.
    bool rejected = !vk::compiled_kernel_support(0).ok() &&
        !vk::VulkanKernelModule::load(binary, signature, 0);
    for (auto count : {0u, 17u}) {
      args[3].uint32_value = count;
      rejected = rejected && !module->launch(args, values[2], count, 1) &&
          !vk::launch_compiled_kernel(module, args, count, 1);
    }
    args[3].uint32_value = 17;
    std::array<Tensor, 1> out{values[2]};
    BackendExecution execution{BackendOpClass::kKernel, {}, {}, out, LaunchConfig{17,1,1,1,1,1},
        CompilationTarget{KernelArtifactKind::kBinary, module->artifact_id(), "tensorcx_kernel_v1"}, args};
    rejected = rejected && !vk::VulkanBackend{}.execute(execution).ok();
    module.reset(); // The foreign-process destructor must not touch the driver.
    ::_exit(rejected ? 0 : 1);
  }
  int status = 0;
  require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "inherited compiler API failed to reject before driver/registry access");
  expect_values(vk::launch_compiled_kernel(module, args, 17, 1).value(), 17);
}
PFN_vkWaitForFences real_wait = nullptr;
PFN_vkDestroyPipeline real_destroy = nullptr;
std::size_t waits = 0, destroyed = 0;
VKAPI_ATTR VkResult VKAPI_CALL failed_second_wait(VkDevice device, std::uint32_t count,
    const VkFence* fences, VkBool32 all, std::uint64_t timeout) {
  if (++waits == 2) return VK_ERROR_OUT_OF_HOST_MEMORY;
  return real_wait(device, count, fences, all, timeout);
}
VKAPI_ATTR VkResult VKAPI_CALL failed_idle(VkDevice) { return VK_ERROR_OUT_OF_HOST_MEMORY; }
VKAPI_ATTR void VKAPI_CALL counted_destroy(VkDevice device, VkPipeline pipeline,
    const VkAllocationCallbacks* callbacks) {
  ++destroyed; real_destroy(device, pipeline, callbacks);
}
void completion_failure(const std::string& binary, bool unproven) {
  auto loaded = vk::VulkanKernelModule::load(binary, signature, 0);
  if (!loaded) throw std::runtime_error(loaded.status().message());
  auto module = loaded.move_value();
  auto context = vk::get_context(0).value();
  auto values = tensors(17); auto args = arguments(values, 17);
  auto pool = std::make_shared<vk::MemoryPool>(0, 8192, 32); require(pool->enter().ok(), "pool entry failed");
  real_wait = context->WaitForFences; real_destroy = context->DestroyPipeline;
  context->WaitForFences = &failed_second_wait; context->DestroyPipeline = &counted_destroy;
  if (unproven) context->DeviceWaitIdle = &failed_idle;
  auto result = vk::launch_compiled_kernel(module, args, 17, 1);
  require(!result && context->poisoned && waits == 2, "compiled failed completion was hidden");
  module.reset();
  require(destroyed == (unproven ? 0u : 1u), "pending pipeline was destroyed or completed pipeline leaked");
  require(!pool->stats().cached_frames && pool->exit().ok(), "failed frame returned to pool");
  require(!pool->stats().cached_bytes, "closed poisoned pool retains idle buffers");
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (!vk::available()) return 77;
    require(argc >= 2, "fixture path required");
    const auto binary = artifact(argv[1]);
    if (argc > 2) completion_failure(binary, std::string_view(argv[2]) == "unproven");
    else { contracts(binary); fork_rejection(binary); }
    require(!vk::validation_errors(), "Vulkan validation reported an error");
    std::cout << "Vulkan compiled kernel contract passed\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
