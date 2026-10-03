#include "cortex/backends/cuda/cuda_kernel.h"
#include "cortex/backends/cuda/cuda_buffer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <mutex>
#include <regex>
#include <unordered_map>

namespace cortex::cuda {
namespace {
std::mutex registry_mutex;
std::unordered_map<std::string, std::weak_ptr<CudaKernelModule>> registry;
std::atomic<std::uint64_t> next_id{1};
Status invalid(std::string message) { return {StatusCode::kInvalidArgument, std::move(message)}; }
Expected<std::shared_ptr<CudaKernelModule>> resolve(const std::string& id) {
  std::lock_guard lock(registry_mutex);
  const auto found = registry.find(id);
  if (found != registry.end()) {
    if (auto module = found->second.lock()) return module;
  }
  return invalid("unknown or expired CUDA kernel artifact");
}
bool valid_signature(const KernelSignature& s) {
  return s.kinds.size() == 4 && std::count(s.kinds.begin(), s.kinds.end(), 't') == 3 &&
         std::count(s.kinds.begin(), s.kinds.end(), 'u') == 1 && s.output_index < 4 &&
         s.guard_index < 4 && s.kinds[s.output_index] == 't' && s.kinds[s.guard_index] == 'u';
}
bool once(const std::string& text, const std::string& pattern) {
  const std::regex expression(pattern);
  return std::distance(std::sregex_iterator(text.begin(), text.end(), expression),
                       std::sregex_iterator()) == 1;
}
}  // namespace

std::string kernel_manifest(const KernelSignature& s, const std::string& operation) {
  return "cortex.cuda.v1|linux-x86_64|llvm-21.1.8|sm52|ptx78|" + operation + "-f32-v1|" +
         s.kinds + "|" + std::to_string(s.output_index) + "|" + std::to_string(s.guard_index);
}

Status compiled_kernel_support() {
#if !defined(__linux__) || !defined(__x86_64__)
  return {StatusCode::kUnavailable, "MLIR CUDA kernels require Linux x86_64"};
#else
  ContextScope context;
  if (!context.ready()) return context.status();
  int major{}, minor{}, runtime_version{}, driver_version{};
  for (auto [attribute, destination] : {
           std::pair{CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, &major},
           std::pair{CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, &minor}}) {
    if (auto status = driver_status(cuDeviceGetAttribute(destination, attribute, 0),
                                     "CUDA capability"); !status.ok()) return status;
  }
  if (auto status = runtime_status(cudaRuntimeGetVersion(&runtime_version), "CUDA Runtime version");
      !status.ok()) return status;
  if (auto status = driver_status(cuDriverGetVersion(&driver_version), "CUDA Driver version");
      !status.ok()) return status;
  if (major != 5 || minor != 2 || runtime_version != 12040 || driver_version != 13000) {
    return {StatusCode::kUnavailable, "MLIR CUDA kernels require sm_52, Runtime 12.4 and Driver API 13.0"};
  }
  return Status::Ok();
#endif
}

struct CudaKernelModule::Impl {
  std::shared_ptr<PrimaryContext> context;
  CUmodule module{};
  CUfunction function{};
  int max_block{}, max_grid{};
  std::string ptx;
  ~Impl() {
    if (module) {
      ContextScope scope(context);
      if (scope.ready()) (void)cuModuleUnload(module);
    }
  }
};
CudaKernelModule::CudaKernelModule() : impl_(std::make_unique<Impl>()) {}
CudaKernelModule::~CudaKernelModule() {
  std::lock_guard lock(registry_mutex);
  if (!id_.empty()) registry.erase(id_);
}

Expected<std::shared_ptr<CudaKernelModule>> CudaKernelModule::load(
    const std::string& ptx, const std::string& entry, KernelSignature signature) {
  try {
    if (!valid_signature(signature)) return invalid("invalid CUDA kernel signature");
    std::string operation;
    if (entry == "cortex_add_v1") operation = "add";
    else if (entry == "cortex_sub_v1") operation = "sub";
    else if (entry == "cortex_mul_v1") operation = "mul";
    else if (entry == "cortex_expr_v1") operation = "expr";
    else return invalid("unsupported CUDA entry point");
    if (ptx.size() > 1024 * 1024 || ptx.find('\0') != std::string::npos ||
        !ptx.starts_with("// " + kernel_manifest(signature, operation) + "\n") ||
        !once(ptx, R"((^|\n)\.version 7\.8(\r?\n))") ||
        !once(ptx, R"((^|\n)\.target sm_52(\r?\n))") ||
        !once(ptx, R"((^|\n)\.address_size 64(\r?\n))")) {
      return invalid("CUDA PTX target/ABI manifest mismatch");
    }
    if (auto status = compiled_kernel_support(); !status.ok()) return status;
    ContextScope scope;
    if (!scope.ready()) return scope.status();
    auto module = std::shared_ptr<CudaKernelModule>(new CudaKernelModule);
    module->impl_->context = scope.owner();
    module->impl_->ptx = ptx;
    std::array<char, 4096> log{};
    CUjit_option options[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void* values[] = {log.data(), reinterpret_cast<void*>(static_cast<std::uintptr_t>(log.size()))};
    auto result = cuModuleLoadDataEx(&module->impl_->module, ptx.c_str(), 2, options, values);
    if (result != CUDA_SUCCESS) {
      log.back() = '\0';
      return invalid(driver_status(result, "CUDA PTX JIT").message() + ": " + log.data());
    }
    unsigned functions{};
    if (auto status = driver_status(cuModuleGetFunctionCount(&functions, module->impl_->module),
                                     "CUDA module function count"); !status.ok()) return status;
    if (functions != 1) return invalid("CUDA module must contain exactly one kernel");
    if (auto status = driver_status(cuModuleGetFunction(&module->impl_->function,
                      module->impl_->module, entry.c_str()), "CUDA entry lookup"); !status.ok()) return status;
    std::size_t expected_offset = 0;
    for (std::size_t i = 0; i < 4; ++i) {
      const std::size_t expected_size = signature.kinds[i] == 't' ? 8 : 4;
      expected_offset = (expected_offset + expected_size - 1) / expected_size * expected_size;
      std::size_t offset{}, size{};
      if (cuFuncGetParamInfo(module->impl_->function, i, &offset, &size) != CUDA_SUCCESS ||
          offset != expected_offset || size != expected_size) return invalid("CUDA entry parameter ABI mismatch");
      expected_offset += expected_size;
    }
    std::size_t offset{}, size{};
    if (cuFuncGetParamInfo(module->impl_->function, 4, &offset, &size) != CUDA_ERROR_INVALID_VALUE) {
      return invalid("CUDA entry parameter count mismatch");
    }
    int device_block{}, function_block{}, dimension_block{};
    if (auto status = driver_status(cuDeviceGetAttribute(&device_block,
                   CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 0), "CUDA block limit"); !status.ok()) return status;
    if (auto status = driver_status(cuDeviceGetAttribute(&dimension_block,
                   CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, 0), "CUDA block dimension limit"); !status.ok()) return status;
    if (auto status = driver_status(cuFuncGetAttribute(&function_block,
                   CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, module->impl_->function), "CUDA function limit");
        !status.ok()) return status;
    if (auto status = driver_status(cuDeviceGetAttribute(&module->impl_->max_grid,
                   CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, 0), "CUDA grid limit"); !status.ok()) return status;
    module->impl_->max_block = std::min({device_block, dimension_block, function_block});
    module->entry_ = entry;
    module->signature_ = std::move(signature);
    module->id_ = "cuda-kernel-" + std::to_string(next_id.fetch_add(1));
    {
      std::lock_guard lock(registry_mutex);
      registry.emplace(module->id_, module);
    }
    return module;
  } catch (const std::exception& error) {
    return Status(StatusCode::kInternal, error.what());
  }
}

Expected<CudaTensor> CudaKernelModule::launch(std::span<const KernelArgument> arguments,
    const Tensor& output, std::uint32_t threads, std::uint32_t block) const {
  try {
    if (arguments.size() != 4) return invalid("CUDA kernel argument count mismatch");
    if (!block || block > static_cast<unsigned>(impl_->max_block)) return invalid("CUDA block size exceeds limits");
    const auto blocks = threads ? 1 + (static_cast<std::uint64_t>(threads) - 1) / block : 0;
    if (blocks > static_cast<unsigned>(impl_->max_grid)) return invalid("CUDA grid size exceeds limits");
    auto out = from_core_tensor(output);
    if (!out) return out.status();
    if (output.dtype != DType::kFloat32 || threads > static_cast<std::uint64_t>(numel(output.shape))) {
      return invalid("CUDA kernel output dtype/size mismatch");
    }
    for (std::size_t i = 0; i < 4; ++i) {
      const auto& arg = arguments[i];
      if (signature_.kinds[i] == 'u') {
        if (arg.kind != KernelArgumentKind::kUInt32 || arg.tensor || arg.uint32_value != threads) {
          return invalid("CUDA guard argument must equal thread_count");
        }
      } else {
        if (arg.kind != KernelArgumentKind::kTensor || !arg.tensor) return invalid("CUDA tensor argument kind mismatch");
        auto tensor = from_core_tensor(*arg.tensor);
        if (!tensor) return tensor.status();
        if (tensor.value().dtype() != DType::kFloat32 || tensor.value().shape() != output.shape ||
            tensor.value().buffer()->context() != impl_->context) return invalid("CUDA tensor dtype/shape/context mismatch");
        if (i == signature_.output_index && tensor.value().buffer() != out.value().buffer()) {
          return invalid("CUDA output argument mismatch");
        }
      }
    }
    ContextScope context(impl_->context);
    if (!context.ready()) return context.status();
    auto copied = CudaBuffer::create(DType::kFloat32, output.shape);
    if (!copied) return copied.status();
    if (copied.value()->nbytes()) {
      if (auto status = runtime_status(cudaMemcpy(copied.value()->data(), out.value().buffer()->data(),
                  copied.value()->nbytes(), cudaMemcpyDeviceToDevice), "CUDA output copy"); !status.ok()) return status;
      if (auto status = runtime_status(cudaDeviceSynchronize(), "CUDA output copy synchronize"); !status.ok()) return status;
    }
    std::array<CUdeviceptr, 4> pointers{};
    std::array<std::uint32_t, 4> scalars{};
    std::array<void*, 4> slots{};
    for (std::size_t i = 0; i < 4; ++i) {
      if (signature_.kinds[i] == 'u') {
        scalars[i] = arguments[i].uint32_value;
        slots[i] = &scalars[i];
      } else {
        auto buffer = std::static_pointer_cast<CudaBuffer>(arguments[i].tensor->buffer);
        if (buffer == out.value().buffer()) buffer = copied.value();
        pointers[i] = reinterpret_cast<CUdeviceptr>(buffer->data());
        slots[i] = &pointers[i];
      }
    }
    if (threads) {
      if (auto status = driver_status(cuLaunchKernel(impl_->function, static_cast<unsigned>(blocks),
              1, 1, block, 1, 1, 0, nullptr, slots.data(), nullptr), "CUDA generated kernel launch");
          !status.ok()) return status;
      if (auto status = driver_status(cuCtxSynchronize(), "CUDA generated kernel synchronize");
          !status.ok()) return status;
    }
    return CudaTensor(DType::kFloat32, output.shape, copied.move_value());
  } catch (const std::invalid_argument& error) {
    return invalid(error.what());
  } catch (const std::exception& error) {
    return Status(StatusCode::kInternal, error.what());
  }
}

Status execute_compiled_kernel(const BackendExecution& e) {
  if (auto status = validate_kernel_execution_contract(e); !status.ok()) return status;
  if (e.outputs.size() != 1 || e.compilation_target->artifact_kind != KernelArtifactKind::kBinary) return invalid("unsupported CUDA kernel artifact/output");
  const auto& l = *e.launch;
  if (l.grid_y != 1 || l.grid_z != 1 || l.threads_per_group_y != 1 || l.threads_per_group_z != 1) {
    return invalid("CUDA kernel requires one-dimensional launch");
  }
  auto module = resolve(e.compilation_target->artifact);
  if (!module) return module.status();
  if (e.compilation_target->entry_point != module.value()->entry_point()) {
    return invalid("CUDA kernel entry does not match artifact");
  }
  auto result = module.value()->launch(e.kernel_arguments, e.outputs[0], l.grid_x, l.threads_per_group_x);
  if (!result) return result.status();
  e.outputs[0] = to_core_tensor(result.value());
  return Status::Ok();
}

Expected<CudaTensor> launch_compiled_kernel(const std::shared_ptr<CudaKernelModule>& module,
    std::span<const KernelArgument> arguments, std::uint32_t threads, std::uint32_t block) {
  if (!module) return invalid("missing CUDA module");
  const auto index = module->signature().output_index;
  if (arguments.size() != 4 || !arguments[index].tensor ||
      arguments[index].kind != KernelArgumentKind::kTensor) return invalid("missing CUDA output");
  std::array<Tensor, 1> outputs{*arguments[index].tensor};
  if (!threads) return module->launch(arguments, outputs[0], threads, block);
  BackendExecution e{BackendOpClass::kKernel, {}, {}, outputs, LaunchConfig{threads, 1, 1, block, 1, 1},
    CompilationTarget{KernelArtifactKind::kBinary, module->artifact_id(), module->entry_point()}, arguments};
  CudaBackend backend;
  if (auto status = backend.execute(e); !status.ok()) return status;
  return from_core_tensor(outputs[0]);
}
}  // namespace cortex::cuda
