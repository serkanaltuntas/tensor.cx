#include <array>
#include <cstdint>
#include <iostream>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>

#include "tensorcx/backends/cpu/cpu_backend.h"
#include "tensorcx/backends/vulkan/vulkan_backend.h"
#include "tensorcx/backends/vulkan/vulkan_context.h"
#include "batched_matmul_contract.h"
#include "math_contract.h"

using namespace tensorcx;
namespace vk = tensorcx::vulkan;
namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void invalid(const Status& status) { require(status.code() == StatusCode::kInvalidArgument, status.message().c_str()); }

VKAPI_ATTR VkResult VKAPI_CALL failed_wait(VkDevice, std::uint32_t, const VkFence*, VkBool32, std::uint64_t) {
  return VK_ERROR_OUT_OF_HOST_MEMORY;
}
VKAPI_ATTR VkResult VKAPI_CALL failed_idle(VkDevice) { return VK_ERROR_OUT_OF_HOST_MEMORY; }

PFN_vkAllocateDescriptorSets real_allocate_descriptors = nullptr;
int descriptor_calls = 0;
VKAPI_ATTR VkResult VKAPI_CALL fail_second_descriptors(
    VkDevice device, const VkDescriptorSetAllocateInfo* info, VkDescriptorSet* sets) {
  if (++descriptor_calls == 2) return VK_ERROR_OUT_OF_HOST_MEMORY;
  return real_allocate_descriptors(device, info, sets);
}

void math_failure_contract(vk::VulkanBackend& backend) {
  auto x = vk::from_cpu(cpu::CpuTensor({3}, std::vector<float>{1, 2, 3})).move_value();
  auto lo = vk::from_cpu(cpu::CpuTensor({}, std::vector<float>{0})).move_value();
  auto hi = vk::from_cpu(cpu::CpuTensor({}, std::vector<float>{2})).move_value();
  std::array<Tensor, 3> inputs{vk::to_core_tensor(x), vk::to_core_tensor(lo), vk::to_core_tensor(hi)};
  std::array<Tensor, 2> outputs{inputs[1], inputs[2]};
  BackendExecution e;
  e.op = OpDesc{OpKind::kTopK}; e.op.k = 2;
  e.inputs = std::span<const Tensor>(inputs).first(1); e.outputs = outputs;
  auto context = x.buffer()->owner();
  const auto before = context->allocation_count.load();
  const auto limit = context->limits.maxMemoryAllocationCount;
  context->limits.maxMemoryAllocationCount = before + 1; // Fail the second result allocation.
  const auto status = backend.execute(e);
  context->limits.maxMemoryAllocationCount = limit;
  require(!status.ok(), "topk accepted an allocation failure");
  require(outputs[0].buffer == inputs[1].buffer && outputs[1].buffer == inputs[2].buffer,
          "topk allocation failure partially published results");
  require(context->allocation_count == before, "topk failed allocation leaked private output");
  require(backend.execute(e).ok(), "topk did not recover after allocation failure");

  outputs = {inputs[1], inputs[2]};
  e.op = OpDesc{OpKind::kClip}; e.inputs = inputs; e.outputs = std::span<Tensor>(outputs).first(1);
  const auto clip_before = context->allocation_count.load();
  real_allocate_descriptors = context->AllocateDescriptorSets; descriptor_calls = 0;
  context->AllocateDescriptorSets = &fail_second_descriptors;
  const auto clip_status = backend.execute(e);
  context->AllocateDescriptorSets = real_allocate_descriptors;
  require(!clip_status.ok() && descriptor_calls == 2, "clip second-pass fault was not exercised");
  require(outputs[0].buffer == inputs[1].buffer && outputs[1].buffer == inputs[2].buffer,
          "clip second-pass failure published private intermediate");
  require(context->allocation_count == clip_before && !context->poisoned,
          "pre-submit clip failure leaked storage or poisoned device");
  require(backend.execute(e).ok(), "clip did not recover after second-pass rejection");
  const auto unchanged = vk::to_cpu(x).value();
  require(unchanged.float_data()[0] == 1 && unchanged.float_data()[2] == 3,
          "failed math modified input storage");
}

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
  math_contract(backend,
      [](const cpu::CpuTensor& input) { return vk::to_core_tensor(vk::from_cpu(input).value()); },
      [](const Tensor& output) { return vk::to_cpu(vk::from_core_tensor(output).value()).move_value(); });
  math_failure_contract(backend);
  for (auto preference : {MatmulPreference::kAuto, MatmulPreference::kCustom})
    batched_matmul_contract(backend,
        [](const cpu::CpuTensor& input) { return vk::to_core_tensor(vk::from_cpu(input).value()); },
        [](const Tensor& output) { return vk::to_cpu(vk::from_core_tensor(output).value()).move_value(); },
        preference);
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
    for (auto kind : {OpKind::kSum, OpKind::kMax, OpKind::kMean, OpKind::kMin, OpKind::kArgmax}) {
      execution.op = OpDesc{kind}; execution.op.axis = 0;
      invalid(backend.execute(execution));
      require(outputs[0].buffer == original.buffer, "rejected reduction published an output");
    }
    for (auto kind : {OpKind::kCast, OpKind::kTranspose, OpKind::kSlice, OpKind::kConcat}) {
      execution.op = OpDesc{kind};
      execution.op.target_dtype = DType::kInt32;
      execution.op.axes = {0};
      execution.op.slice_starts = {0}; execution.op.slice_steps = {1}; execution.op.slice_shape = {3};
      invalid(backend.execute(execution));
      require(outputs[0].buffer == original.buffer, "rejected transform published an output");
    }
  }
  auto fake_context = std::make_shared<vk::Context>();
  auto fake_buffer = std::make_shared<vk::VulkanBuffer>(DType::kFloat32, 0, fake_context, nullptr);
  auto foreign = original;
  foreign.shape = {0}; foreign.buffer = fake_buffer;
  require(!vk::from_core_tensor(foreign), "foreign context accepted");

  const auto mask = vk::to_core_tensor(vk::from_cpu(cpu::CpuTensor({3}, std::vector<std::uint8_t>{0, 2, 255})).value());
  const auto check_predicate_rejection = [&](OpDesc op, std::vector<Tensor> arguments) {
    std::array<Tensor, 1> output{original};
    BackendExecution execution;
    execution.op = op; execution.inputs = arguments; execution.outputs = output;
    invalid(backend.execute(execution));
    require(output[0].buffer == original.buffer, "rejected predicate published output");
  };
  check_predicate_rejection(OpDesc{OpKind::kWhere}, {original, original, original});
  check_predicate_rejection(OpDesc{OpKind::kLogicalAnd}, {original, original});
  check_predicate_rejection(OpDesc{OpKind::kMaskedSelect}, {original, original});
  check_predicate_rejection(OpDesc{OpKind::kAny}, {original});
  auto bad_axis = OpDesc{OpKind::kAll}; bad_axis.reduction_axes = Shape{0, 0};
  check_predicate_rejection(bad_axis, {mask});
  for (auto kind : {OpKind::kSum, OpKind::kMax, OpKind::kMean, OpKind::kMin, OpKind::kArgmax}) {
    auto op = OpDesc{kind}; op.reduction_axes = Shape{0, 0};
    check_predicate_rejection(op, {original});
    op.reduction_axes.reset(); op.axis = 1;
    check_predicate_rejection(op, {original});
    op.axis = 0;
    check_predicate_rejection(op, {mask});
    check_predicate_rejection(op, {foreign});
  }
  for (auto preference : {MatmulPreference::kOptimized, static_cast<MatmulPreference>(-1)}) {
    auto op = OpDesc{OpKind::kMatmul}; op.matmul_preference = preference;
    check_predicate_rejection(op, {original, original});
  }
  for (std::size_t position = 0; position < 2; ++position) {
    for (const auto& malformed : {foreign, mask}) {
      std::vector<Tensor> args{original, original}; args[position] = malformed;
      check_predicate_rejection(OpDesc{OpKind::kMatmul}, args);
    }
    for (int mutation = 0; mutation < 3; ++mutation) {
      std::vector<Tensor> args{original, original};
      if (mutation == 0) args[position].offset = 1;
      if (mutation == 1) args[position].device.index = 1;
      if (mutation == 2) args[position].buffer = fake_buffer;
      check_predicate_rejection(OpDesc{OpKind::kMatmul}, args);
    }
  }
  for (auto kind : {OpKind::kEqual, OpKind::kWhere, OpKind::kMaskedSelect, OpKind::kClip}) {
    auto args = kind == OpKind::kWhere ? std::vector<Tensor>{mask, original, original}
              : kind == OpKind::kClip ? std::vector<Tensor>{original, original, original}
              : kind == OpKind::kMaskedSelect ? std::vector<Tensor>{original, mask}
              : std::vector<Tensor>{original, original};
    for (std::size_t i = 0; i < args.size(); ++i) {
      auto corrupted = args; corrupted[i].offset = 1;
      check_predicate_rejection(OpDesc{kind}, corrupted);
      corrupted = args; corrupted[i].device.index = 1;
      check_predicate_rejection(OpDesc{kind}, corrupted);
      corrupted = args; corrupted[i].buffer = fake_buffer;
      check_predicate_rejection(OpDesc{kind}, corrupted);
    }
  }

  float last = 0;
  tensor.buffer()->owner()->download(tensor.buffer()->allocation(), &last, sizeof(last), 8);
  require(last == 3, "offset control readback failed");
  for (std::size_t offset : {1u, 12u, 16u}) {
    bool rejected = false; last = 47;
    try { tensor.buffer()->owner()->download(tensor.buffer()->allocation(), &last, sizeof(last), offset); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && last == 47, "invalid offset download changed destination");
  }

  // Primitive transform rejections must preserve the caller's output slot.
  std::array<Tensor, 1> inputs{original}, unchanged{original};
  BackendExecution transform;
  transform.inputs = inputs; transform.outputs = unchanged;
  for (auto op : {OpDesc{OpKind::kTranspose}, OpDesc{OpKind::kSlice}, OpDesc{OpKind::kConcat}}) {
    op.axis = 8; transform.op = op;
    invalid(backend.execute(transform));
    require(unchanged[0].buffer == original.buffer, "bad transform plan published an output");
  }
  auto bad_cast = vk::from_cpu(cpu::CpuTensor({3}, std::vector<float>{1, NAN, 3})).move_value();
  inputs[0] = vk::to_core_tensor(bad_cast);
  transform.op = OpDesc{OpKind::kCast}; transform.op.target_dtype = DType::kInt32;
  invalid(backend.execute(transform));
  require(unchanged[0].buffer == original.buffer, "invalid float conversion published an output");
  require(!bad_cast.buffer()->owner()->poisoned, "conversion validation poisoned the device");
  require(std::isnan(vk::to_cpu(bad_cast).value().float_data()[1]), "failed cast modified its input");
  inputs[0] = original;
  require(backend.execute(transform).ok(), "conversion did not recover after invalid input");

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
  require(output[0].buffer == sentinel, "invalid execution changed output");

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
