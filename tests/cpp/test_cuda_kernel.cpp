#include <algorithm>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <array>
#include <fstream>
#include <filesystem>
#include <functional>
#include <future>
#include <iostream>
#include <iterator>
#include <limits>
#include <vector>
#include "tensorcx/backends/cuda/cuda_kernel.h"
#include "tensorcx/backends/cuda/cuda_buffer.h"

using namespace tensorcx;
using namespace tensorcx::cuda;
class ForeignBuffer final : public Buffer {
 public:
  std::size_t nbytes() const override { return 16; }
};
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  auto support = compiled_kernel_support();
  if (!support.ok()) {
    std::cout << support.message() << '\n';
    return std::getenv("TENSORCX_REQUIRE_MLIR_CUDA") ? 1 : 77;
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
  auto loaded = CudaKernelModule::load(ptx, "tensorcx_add_v1", {"tttu", 2, 3});
  if (!loaded) { std::cerr << loaded.status().message() << '\n'; cuCtxDestroy(foreign); return 1; }
  auto module = loaded.move_value();
  restored();
  check(!CudaKernelModule::load("bad", "tensorcx_add_v1", {"tttu", 2, 3}), "bad PTX accepted");
  check(!CudaKernelModule::load(ptx, "missing", {"tttu", 2, 3}), "bad entry accepted");
  check(!CudaKernelModule::load(ptx, "tensorcx_add_v1", {"tttu", 1, 3}), "bad manifest accepted");
  auto malformed = ptx;
  malformed.replace(malformed.find(".entry tensorcx_add_v1"), 20, ".entry missing_entry");
  check(!CudaKernelModule::load(malformed, "tensorcx_add_v1", {"tttu", 2, 3}), "missing function accepted");
  auto wrong_abi = ptx;
  wrong_abi.replace(wrong_abi.find(".u32 tensorcx_add_v1_param_3"), 4, ".u64");
  check(!CudaKernelModule::load(wrong_abi, "tensorcx_add_v1", {"tttu", 2, 3}), "wrong parameter ABI accepted");
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
      CompilationTarget{KernelArtifactKind::kBinary, module->artifact_id(), "tensorcx_add_v1"}, args};
    CudaBackend backend;
    check(backend.execute(e).ok(), "tensor.cx buffer fixture launch failed");
    auto result = from_core_tensor(outputs[0]);
    check(result && std::ranges::equal(to_cpu(result.value()).value().float_data(), std::vector<float>({6, 8, -3, -4})),
          "partial tensor.cx output mismatch");
    check(std::ranges::equal(to_cpu(destination.value()).value().float_data(), std::vector<float>({-1, -2, -3, -4})),
          "caller output mutated");
    restored();
    // Each operation keeps its own entry/manifest and dispatches via the registry.
    {
      // ABI-level multi-store probe: private-output remapping must recognize
      // independent wrappers over one range. This does not expand DSL syntax.
      auto alias_ptx = ptx;
      const std::string store = "st.global.b32 \t[%rd1], %r6;";
      const auto position = alias_ptx.find(store);
      check(position != std::string::npos, "alias probe fixture store missing");
      if (position != std::string::npos) {
        alias_ptx.replace(position, store.size(),
          "st.volatile.global.b32 [%rd1], %r6;\n"
          "ld.volatile.global.b32 %r4, [%rd3];\n"
          "add.rn.f32 %r6, %r4, %r6;\n"
          "st.volatile.global.b32 [%rd1], %r6;");
        auto alias_module = CudaKernelModule::load(alias_ptx, "tensorcx_add_v1", {"tttu", 2, 3});
        check(bool(alias_module), "alias probe module failed");
        auto owner = left.value().buffer();
        auto borrowed = CudaBuffer::borrow(DType::kFloat32, {4}, owner->data(), owner);
        check(bool(borrowed), "alias probe borrow failed");
        if (alias_module && borrowed) {
          Tensor alias_out = to_core_tensor(CudaTensor(DType::kFloat32, {4}, borrowed.value()));
          auto alias_args = args;
          alias_args[2].tensor = &alias_out;
          auto result = alias_module.value()->launch(alias_args, alias_out, 2, 7);
          check(result && std::ranges::equal(to_cpu(result.value()).value().float_data(),
                std::vector<float>{12, 16, 3, 4}), "borrowed CUDA exact alias remap failed");
          check(std::ranges::equal(to_cpu(left.value()).value().float_data(), std::vector<float>{1, 2, 3, 4}),
                "borrowed CUDA alias mutated source");
          auto partial = CudaBuffer::borrow(DType::kFloat32, {3}, static_cast<float*>(owner->data()) + 1, owner).value();
          auto prefix = CudaBuffer::borrow(DType::kFloat32, {3}, owner->data(), owner).value();
          auto pa = to_core_tensor(CudaTensor(DType::kFloat32, {3}, prefix));
          auto po = to_core_tensor(CudaTensor(DType::kFloat32, {3}, partial));
          alias_args[0].tensor = &pa; alias_args[1].tensor = &pa; alias_args[2].tensor = &po;
          auto rejected = alias_module.value()->launch(alias_args, po, 2, 7);
          check(!rejected && rejected.status().message().find("overlap") != std::string::npos,
                "borrowed CUDA partial output overlap accepted");
        }
      }
      restored();
    }
    for (const std::string operation : {"sub", "mul", "expr"}) {
      const auto fixture = std::filesystem::path(argv[1]).parent_path() / ("cuda_" + operation + "_sm52.ptx");
      std::ifstream stream(fixture);
      const std::string code((std::istreambuf_iterator<char>(stream)), {});
      const auto entry = "tensorcx_" + operation + "_v1";
      auto candidate = CudaKernelModule::load(code, entry, {"tttu", 2, 3});
      check(static_cast<bool>(candidate), "arithmetic module failed to load");
      if (!candidate) continue;
      check(!CudaKernelModule::load(code, "tensorcx_add_v1", {"tttu", 2, 3}), "wrong operation manifest accepted");
      outputs[0] = out;
      e.compilation_target = CompilationTarget{KernelArtifactKind::kBinary, candidate.value()->artifact_id(), entry};
      check(backend.execute(e).ok(), "arithmetic dispatch failed");
      auto arithmetic = from_core_tensor(outputs[0]);
      const auto expected = operation == "sub" ? std::vector<float>{-4, -4, -3, -4}
                                               : operation == "mul" ? std::vector<float>{5, 12, -3, -4}
                                                                    : std::vector<float>{4, 11, -3, -4};
      check(arithmetic && std::ranges::equal(to_cpu(arithmetic.value()).value().float_data(), expected),
            "arithmetic partial output mismatch");
      outputs[0] = out;
      e.compilation_target->entry_point = "tensorcx_add_v1";
      check(!backend.execute(e).ok() && outputs[0].buffer == out.buffer,
            "artifact accepted another valid operation entry");
      restored();
    }
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
        CompilationTarget{KernelArtifactKind::kBinary, module->artifact_id(), "tensorcx_add_v1"}, args};
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
        return value && std::ranges::equal(to_cpu(value.value()).value().float_data(), std::vector<float>({2, 4, 6, 8}));
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
