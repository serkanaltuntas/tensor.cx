#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <limits>

#include "tensorcx/backends/cpu/cpu_backend.h"
#include "tensorcx/backends/cpu/cpu_kernel.h"
#include "tensorcx/backends/null/null_backend.h"

using namespace tensorcx;
using namespace tensorcx::cpu;

class ForeignBuffer final : public Buffer {
 public:
  std::size_t nbytes() const override { return 16; }
};

int main(int argc, char** argv) {
  int failures = 0;
  auto check = [&](bool result, const char* message) {
    if (!result) { std::cerr << message << '\n'; ++failures; }
  };
  if (argc != 4) return 2;
  check(!CpuKernelModule::load("/missing/tensorcx.so", {"ttu", 1, 2}), "missing library accepted");
  check(!CpuKernelModule::load(argv[2], {"ttu", 1, 2}), "bad manifest accepted");
  check(!CpuKernelModule::load(argv[3], {"ttu", 1, 2}), "missing symbol accepted");
  check(!CpuKernelModule::load(argv[1], {"tzu", 1, 2}), "invalid signature accepted");
  check(!CpuKernelModule::load(argv[1], {"ttu", 0, 2}), "forged signature accepted");
  auto loaded = CpuKernelModule::load(argv[1], {"ttu", 1, 2});
  if (!loaded) { std::cerr << loaded.status().message(); return 1; }
  auto module = loaded.move_value();
  const auto artifact = module->artifact_id();
  CpuTensor input({4}, std::vector<float>{2, 3, 4, 5});
  CpuTensor output({4}, std::vector<float>{-1, -2, -3, -4});
  auto a = to_core_tensor(input);
  auto out = to_core_tensor(output);
  std::array<Tensor, 1> outputs{out};
  std::array<KernelArgument, 3> arguments{{
      {KernelArgumentKind::kTensor, &a, 0},
      {KernelArgumentKind::kTensor, &out, 0},
      {KernelArgumentKind::kUInt32, nullptr, 2}}};
  BackendExecution execution{BackendOpClass::kKernel, {}, {}, outputs,
      LaunchConfig{2, 1, 1, 7, 1, 1},
      CompilationTarget{KernelArtifactKind::kBinary, artifact, "tensorcx_launch_v1"}, arguments};
  CpuBackend backend;
  check(backend.execute(execution).ok(), "valid execution rejected");
  check(std::ranges::equal(from_core_tensor(outputs[0]).float_data(), std::vector<float>({3, 4, -3, -4})),
        "partial output mismatch");
  check(std::ranges::equal(output.float_data(), std::vector<float>({-1, -2, -3, -4})), "source output mutated");
  auto reject = [&](const char* name, const std::function<void()>& mutate) {
    a = to_core_tensor(input);
    out = to_core_tensor(output);
    outputs[0] = out;
    arguments = {{{KernelArgumentKind::kTensor, &a, 0},
                  {KernelArgumentKind::kTensor, &out, 0},
                  {KernelArgumentKind::kUInt32, nullptr, 2}}};
    execution = {BackendOpClass::kKernel, {}, {}, outputs,
      LaunchConfig{2, 1, 1, 7, 1, 1},
      CompilationTarget{KernelArtifactKind::kBinary, artifact, "tensorcx_launch_v1"}, arguments};
    mutate();
    const auto original = outputs[0].buffer;
    check(!backend.execute(execution).ok(), name);
    check(outputs[0].buffer == original, "failed execution published output");
  };
  reject("wrong device accepted", [&] { a.device = {"cuda", 0}; });
  reject("nonzero device index accepted", [&] { a.device.index = 1; });
  reject("offset accepted", [&] { a.offset = 1; });
  reject("noncontiguous strides accepted", [&] { a.strides = {2}; });
  reject("dtype accepted", [&] { a.dtype = DType::kInt32; });
  reject("foreign concrete buffer accepted", [&] { a.buffer = std::make_shared<ForeignBuffer>(); });
  reject("missing buffer accepted", [&] { a.buffer.reset(); });
  reject("wrong shape accepted", [&] { a.shape = {2, 2}; a.strides = {2, 1}; });
  reject("undersized buffer accepted", [&] { a.shape = {8}; });
  reject("overflow shape accepted", [&] { a.shape = {std::numeric_limits<std::int64_t>::max(), 3}; });
  reject("count mismatch accepted", [&] { execution.kernel_arguments = std::span(arguments).first(2); });
  reject("scalar kind accepted", [&] { arguments[2].kind = KernelArgumentKind::kTensor; });
  reject("tensor kind accepted", [&] { arguments[0].kind = KernelArgumentKind::kUInt32; });
  reject("null tensor accepted", [&] { arguments[0].tensor = nullptr; });
  reject("scalar tensor accepted", [&] { arguments[2].tensor = &a; });
  reject("guard mismatch accepted", [&] { arguments[2].uint32_value = 1; });
  reject("excess threads accepted", [&] { execution.launch->grid_x = 5; arguments[2].uint32_value = 5; });
  reject("zero backend launch accepted", [&] { execution.launch->grid_x = 0; });
  reject("zero block accepted", [&] { execution.launch->threads_per_group_x = 0; });
  reject("two dimensional launch accepted", [&] { execution.launch->grid_y = 2; });
  reject("wrong output argument accepted", [&] { arguments[1].tensor = &a; });
  reject("source artifact accepted", [&] { execution.compilation_target->artifact_kind = KernelArtifactKind::kSource; });
  reject("wrong entry accepted", [&] { execution.compilation_target->entry_point = "other"; });
  reject("unknown artifact accepted", [&] { execution.compilation_target->artifact = "unknown"; });
  reject("expired artifact accepted", [&] { module.reset(); });

  return failures ? 1 : 0;
}
