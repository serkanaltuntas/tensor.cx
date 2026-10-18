#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

#include "tensorcx/backends/cpu/cpu_backend.h"
#include "tensorcx/backends/vulkan/vulkan_backend.h"
#include "tensorcx/backends/vulkan/vulkan_context.h"

using namespace tensorcx;
namespace vk = tensorcx::vulkan;
namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void invalid(const Status& status) { require(status.code() == StatusCode::kInvalidArgument, status.message().c_str()); }

VKAPI_ATTR VkResult VKAPI_CALL failed_wait(VkDevice, std::uint32_t, const VkFence*, VkBool32, std::uint64_t) {
  return VK_ERROR_OUT_OF_HOST_MEMORY;
}
VKAPI_ATTR VkResult VKAPI_CALL failed_idle(VkDevice) { return VK_ERROR_OUT_OF_HOST_MEMORY; }

void failed_completion(bool unproven_idle) {
  auto context = vk::get_context(0).value();
  auto a = vk::from_cpu(cpu::CpuTensor({3}, std::vector<float>{1, 2, 3})).move_value();
  auto staging = context->allocate(12, true);
  auto target = context->allocate(12);
  const auto references = target.use_count();
  const auto original_wait = context->WaitForFences;
  const auto original_idle = context->DeviceWaitIdle;
  context->WaitForFences = &failed_wait;
  if (unproven_idle) context->DeviceWaitIdle = &failed_idle;
  bool threw = false;
  try { const float values[]{4, 5, 6}; context->upload(target, values, sizeof(values)); }
  catch (const std::runtime_error&) { threw = true; }
  context->WaitForFences = original_wait;
  context->DeviceWaitIdle = original_idle;
  require(threw && context->poisoned, "failed completion did not poison context");
  require((target.use_count() > references) == unproven_idle, "pending resources released without idle proof");
  // Restore and complete the actual GPU work before this test process exits.
  require(original_idle(context->device) == VK_SUCCESS, "real idle failed after fault injection");
  require(!vk::available(), "poisoned device was rediscovered as healthy");
  require(vk::validation_errors() == 0, "fault path produced Vulkan validation errors");
}

void contract() {
  vk::VulkanBackend backend;
  auto created = vk::from_cpu(cpu::CpuTensor({3}, std::vector<float>{1, 2, 3}));
  require(static_cast<bool>(created), "Vulkan upload failed");
  const auto tensor = created.move_value();
  const auto original = vk::to_core_tensor(tensor);
  for (int mutation = 0; mutation < 10; ++mutation) {
    auto input = original;
    switch (mutation) {
      case 0: input.device.type = "cpu"; break;
      case 1: input.device.index = -1; break;
      case 2: input.device.index = 1; break;
      case 3: input.offset = 1; break;
      case 4: input.strides = {2}; break;
      case 5: input.dtype = DType::kInt32; break;
      case 6: input.shape = {4}; break;
      case 7: input.buffer.reset(); break;
      case 8: input.buffer = std::make_shared<cpu::CpuBuffer>(DType::kFloat32, 3); break;
      case 9: input.shape = {std::numeric_limits<std::int64_t>::max(), 3}; break;
    }
    require(!vk::from_core_tensor(input), "malformed native metadata accepted");
    std::array<Tensor, 1> inputs{input}, outputs{original};
    BackendExecution execution;
    execution.op = OpDesc{OpKind::kNegate}; execution.inputs = inputs; execution.outputs = outputs;
    invalid(backend.execute(execution));
    require(outputs[0].buffer == original.buffer, "failure published an output");
  }
  auto fake_context = std::make_shared<vk::Context>();
  auto fake_buffer = std::make_shared<vk::VulkanBuffer>(DType::kFloat32, 0, fake_context, nullptr);
  auto foreign = original;
  foreign.shape = {0}; foreign.buffer = fake_buffer;
  require(!vk::from_core_tensor(foreign), "foreign context accepted");

  for (DType dtype : {DType::kFloat32, DType::kInt32, DType::kBool}) {
    auto empty = vk::empty({1000000000000, 0}, dtype);
    require(empty && empty.value().nbytes() == 0, "empty logical storage failed");
    require(!empty.value().buffer()->allocation(), "empty tensor allocated Vulkan storage");
    auto back = vk::to_cpu(empty.value());
    require(back && back.value().size() == 0, "empty download failed");
  }
  auto context = tensor.buffer()->owner();
  const auto live_before = context->allocation_count.load();
  const auto max_count = context->limits.maxMemoryAllocationCount;
  context->limits.maxMemoryAllocationCount = live_before + 1;
  auto one = context->allocate(4);
  bool rejected = false;
  try { static_cast<void>(context->allocate(4)); }
  catch (const std::runtime_error&) { rejected = true; }
  context->limits.maxMemoryAllocationCount = max_count;
  require(rejected && context->allocation_count == live_before + 1, "allocation reservation limit failed");
  one.reset();
  require(context->allocation_count == live_before, "allocation reservation leaked");
  require(!vk::empty({4294967296}, DType::kFloat32), "oversized allocation accepted");
  require(!vk::empty({-1}, DType::kFloat32), "negative shape accepted");
  require(!vk::empty({1}, DType::kFloat32, -1), "negative device accepted");
  require(!vk::empty({1}, DType::kFloat32, 99999), "unavailable device accepted");

  std::array<Tensor, 1> output{Tensor{DType::kFloat32, {3}, {1}, {"vulkan", 0}, nullptr, 0}};
  BackendExecution fill;
  fill.op = OpDesc{OpKind::kFill}; fill.op.scalar_value = 2.5; fill.outputs = output;
  require(backend.execute(fill).ok(), "fill execution failed");
  auto downloaded = vk::to_cpu(vk::from_core_tensor(output[0]).value()).move_value();
  for (float value : downloaded.float_data()) require(value == 2.5, "fill parity failed");
  const auto sentinel = output[0].buffer;
  fill.op.kind = OpKind::kMatmul;
  invalid(backend.execute(fill));
  require(output[0].buffer == sentinel, "unsupported execution changed output");

  // The common ABI accepts concrete nonnegative identities; legacy backends
  // remain responsible for rejecting unsupported indices.
  output[0] = Tensor{DType::kFloat32, {3}, {1}, {"vulkan", 1}, nullptr, 0};
  require(validate_fill_output_descriptor(output[0], "vulkan").ok(), "ABI rejected indexed identity");
  output[0].device = {"cpu", 1}; fill.op.kind = OpKind::kFill;
  cpu::CpuBackend cpu_backend;
  invalid(cpu_backend.execute(fill));
  require(vk::completion_allows_release(VK_SUCCESS) && vk::completion_allows_release(VK_ERROR_DEVICE_LOST),
          "device-lost lifetime contract incorrect");
  require(!vk::completion_allows_release(VK_TIMEOUT) && !vk::completion_allows_release(VK_ERROR_OUT_OF_HOST_MEMORY),
          "failed waits treated as completion");
  require(vk::validation_errors() == 0, "Vulkan validation errors detected");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--no-loader") {
      require(vk::device_count() == 0 && !vk::available(), "missing loader discovered a GPU");
      cpu::CpuTensor x({2}, std::vector<float>{1, 2});
      std::array<Tensor, 2> inputs{cpu::to_core_tensor(x), cpu::to_core_tensor(x)};
      std::array<Tensor, 1> outputs;
      BackendExecution e; e.op = OpDesc{OpKind::kAdd}; e.inputs = inputs; e.outputs = outputs;
      cpu::CpuBackend backend;
      require(backend.execute(e).ok(), "CPU execution failed without Vulkan loader");
      return 0;
    }
    if (!vk::available()) { std::cout << "Vulkan GPU unavailable\n"; return 77; }
    if (argc == 2) failed_completion(std::string_view(argv[1]) == "--unproven-idle");
    else contract();
    std::cout << "Vulkan backend contract passed\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
