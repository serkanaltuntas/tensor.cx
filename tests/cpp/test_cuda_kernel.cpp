#include <cuda.h>
#include <cuda_runtime_api.h>
#include <array>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <iterator>
#include <limits>
#include <vector>
#include "cortex/backends/cuda/cuda_kernel.h"

using namespace cortex;
using namespace cortex::cuda;
class ForeignBuffer final : public Buffer {
 public:
  std::size_t nbytes() const override { return 16; }
};
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  auto support = compiled_kernel_support();
  if (!support.ok()) {
    std::cout << support.message() << '\n';
    return std::getenv("CORTEX_REQUIRE_MLIR_CUDA") ? 1 : 77;
  }
  int failures{};
  auto check = [&](bool good, const char* message) {
    if (!good) { std::cerr << message << '\n'; ++failures; }
  };
  std::ifstream input(argv[1]);
  const std::string ptx((std::istreambuf_iterator<char>(input)), {});
  check(!ptx.empty(), "missing fixture");
  CUcontext foreign{};
  if (cuCtxCreate(&foreign, 0, 0) != CUDA_SUCCESS) return 1;
  auto restored = [&] {
    CUcontext current{};
    check(cuCtxGetCurrent(&current) == CUDA_SUCCESS && current == foreign, "foreign context not restored");
    int device{-1};
    check(cudaGetDevice(&device) == cudaSuccess && device == 0, "Runtime device not restored");
  };
  auto loaded = CudaKernelModule::load(ptx, "cortex_add_v1", {"tttu", 2, 3});
  if (!loaded) { std::cerr << loaded.status().message() << '\n'; cuCtxDestroy(foreign); return 1; }
  auto module = loaded.move_value();
  restored();
  check(!CudaKernelModule::load("bad", "cortex_add_v1", {"tttu", 2, 3}), "bad PTX accepted");
  check(!CudaKernelModule::load(ptx, "missing", {"tttu", 2, 3}), "bad entry accepted");
  check(!CudaKernelModule::load(ptx, "cortex_add_v1", {"tttu", 1, 3}), "bad manifest accepted");
  auto malformed = ptx;
  malformed.replace(malformed.find(".entry cortex_add_v1"), 20, ".entry missing_entry");
  check(!CudaKernelModule::load(malformed, "cortex_add_v1", {"tttu", 2, 3}), "missing function accepted");
  auto wrong_abi = ptx;
  wrong_abi.replace(wrong_abi.find(".u32 cortex_add_v1_param_3"), 4, ".u64");
  check(!CudaKernelModule::load(wrong_abi, "cortex_add_v1", {"tttu", 2, 3}), "wrong parameter ABI accepted");
  restored();

  {
    auto left = from_cpu(cpu::CpuTensor({4}, std::vector<float>{1, 2, 3, 4}));
    auto right = from_cpu(cpu::CpuTensor({4}, std::vector<float>{5, 6, 7, 8}));
    auto destination = from_cpu(cpu::CpuTensor({4}, std::vector<float>{-1, -2, -3, -4}));
    if (!left || !right || !destination) return 1;
    restored();
    auto a = to_core_tensor(left.value()), b = to_core_tensor(right.value()), out = to_core_tensor(destination.value());
    std::array<Tensor, 1> outputs{out};
    std::array<KernelArgument, 4> args{{
      {KernelArgumentKind::kTensor, &a, 0}, {KernelArgumentKind::kTensor, &b, 0},
      {KernelArgumentKind::kTensor, &out, 0}, {KernelArgumentKind::kUInt32, nullptr, 2}}};
    BackendExecution e{BackendOpClass::kKernel, {}, {}, outputs, LaunchConfig{2, 1, 1, 7, 1, 1},
      CompilationTarget{KernelArtifactKind::kBinary, module->artifact_id(), "cortex_add_v1"}, args};
    CudaBackend backend;
    check(backend.execute(e).ok(), "Cortex buffer fixture launch failed");
    auto result = from_core_tensor(outputs[0]);
    check(result && to_cpu(result.value()).value().float_data() == std::vector<float>({6, 8, -3, -4}),
          "partial Cortex output mismatch");
    check(to_cpu(destination.value()).value().float_data() == std::vector<float>({-1, -2, -3, -4}),
          "caller output mutated");
    restored();
    // Static primitive paths must share the same primary context/restoration.
    std::array<Tensor, 2> primitive_inputs{a, b};
    BackendExecution primitive{BackendOpClass::kPrimitive, OpDesc{OpKind::kAdd}, primitive_inputs, outputs,
                              std::nullopt, std::nullopt, {}};
    check(backend.execute(primitive).ok(), "static add regression");
    restored();
    auto reject = [&](const char* name, const std::function<void()>& change) {
      a = to_core_tensor(left.value()); b = to_core_tensor(right.value()); out = to_core_tensor(destination.value());
      outputs[0] = out;
      args = {{{KernelArgumentKind::kTensor, &a, 0}, {KernelArgumentKind::kTensor, &b, 0},
               {KernelArgumentKind::kTensor, &out, 0}, {KernelArgumentKind::kUInt32, nullptr, 2}}};
      e = {BackendOpClass::kKernel, {}, {}, outputs, LaunchConfig{2, 1, 1, 7, 1, 1},
        CompilationTarget{KernelArtifactKind::kBinary, module->artifact_id(), "cortex_add_v1"}, args};
      change();
      const auto before = outputs[0].buffer;
      check(!backend.execute(e).ok(), name);
      check(outputs[0].buffer == before, "failure published output");
      restored();
    };
    reject("device accepted", [&] { a.device = {"cpu", 0}; });
    reject("device index accepted", [&] { a.device.index = 1; });
    reject("dtype accepted", [&] { a.dtype = DType::kInt32; });
    reject("shape accepted", [&] { a.shape = {2, 2}; a.strides = {2, 1}; });
    reject("strides accepted", [&] { a.strides = {2}; });
    reject("offset accepted", [&] { a.offset = 1; });
    reject("missing buffer accepted", [&] { a.buffer.reset(); });
    reject("foreign buffer accepted", [&] { a.buffer = std::make_shared<ForeignBuffer>(); });
    reject("buffer capacity accepted", [&] { a.shape = {8}; });
    reject("overflow accepted", [&] { a.shape = {std::numeric_limits<std::int64_t>::max(), 3}; });
    reject("argument count accepted", [&] { e.kernel_arguments = std::span(args).first(3); });
    reject("tensor kind accepted", [&] { args[0].kind = KernelArgumentKind::kUInt32; });
    reject("scalar pointer accepted", [&] { args[3].tensor = &a; });
    reject("guard accepted", [&] { args[3].uint32_value = 1; });
    reject("threads accepted", [&] { e.launch->grid_x = 5; args[3].uint32_value = 5; });
    reject("excess grid accepted", [&] { e.launch->grid_x = UINT32_MAX; e.launch->threads_per_group_x = 1; args[3].uint32_value = UINT32_MAX; });
    reject("zero block accepted", [&] { e.launch->threads_per_group_x = 0; });
    reject("excess block accepted", [&] { e.launch->threads_per_group_x = 1025; });
    reject("2D accepted", [&] { e.launch->grid_y = 2; });
    reject("zero backend launch accepted", [&] { e.launch->grid_x = 0; });
    reject("output argument accepted", [&] { args[2].tensor = &a; });
    reject("entry accepted", [&] { e.compilation_target->entry_point = "missing"; });
    reject("artifact kind accepted", [&] { e.compilation_target->artifact_kind = KernelArtifactKind::kSource; });
    reject("unknown artifact accepted", [&] { e.compilation_target->artifact = "unknown"; });
    const auto id = module->artifact_id();
    std::vector<std::future<bool>> pending;
    for (int i = 0; i < 4; ++i) {
      pending.push_back(std::async(std::launch::async, [owner = module, tensor = left.value()] {
        auto t = to_core_tensor(tensor);
        std::array<KernelArgument, 4> arguments{{
          {KernelArgumentKind::kTensor, &t, 0}, {KernelArgumentKind::kTensor, &t, 0},
          {KernelArgumentKind::kTensor, &t, 0}, {KernelArgumentKind::kUInt32, nullptr, 4}}};
        auto value = launch_compiled_kernel(owner, arguments, 4, 7);
        return value && to_cpu(value.value()).value().float_data() == std::vector<float>({2, 4, 6, 8});
      }));
    }
    for (auto& future : pending) check(future.get(), "concurrent alias launch failed");
    pending.clear();
    reject("expired artifact accepted", [&] { e.compilation_target->artifact = id; module.reset(); });
  }
  restored(); // Buffer and module destruction also restored the foreign context.
  cuCtxDestroy(foreign);
  return failures ? 1 : 0;
}
