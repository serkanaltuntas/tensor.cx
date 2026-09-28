#include "cortex/backends/cpu/cpu_kernel.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include "cortex/backends/cpu/cpu_backend.h"

#if defined(__linux__) && defined(__x86_64__)
#include <dlfcn.h>
#define CORTEX_CPU_MODULE_SUPPORTED 1
#else
#define CORTEX_CPU_MODULE_SUPPORTED 0
#endif

namespace cortex::cpu {
namespace {
std::mutex registry_mutex;
std::unordered_map<std::string, std::weak_ptr<CpuKernelModule>> registry;
std::atomic<std::uint64_t> next_id{1};

Status invalid(std::string message) {
  return {StatusCode::kInvalidArgument, std::move(message)};
}

// This is the private, 64-bit C interface layout emitted by LLVM 21.1.8.
struct MemRef1D {
  float* allocated;
  float* aligned;
  std::int64_t offset;
  std::int64_t size;
  std::int64_t stride;
};
#if CORTEX_CPU_MODULE_SUPPORTED
static_assert(sizeof(MemRef1D) == 40);
static_assert(offsetof(MemRef1D, aligned) == 8);
static_assert(offsetof(MemRef1D, offset) == 16);
static_assert(offsetof(MemRef1D, size) == 24);
static_assert(offsetof(MemRef1D, stride) == 32);
#endif

Status validate_signature(const KernelSignature& signature) {
  if (signature.kinds.empty() || signature.kinds.size() > 64 ||
      signature.output_index >= signature.kinds.size() ||
      signature.guard_index >= signature.kinds.size() ||
      signature.kinds[signature.output_index] != 't' ||
      signature.kinds[signature.guard_index] != 'u' ||
      signature.kinds.find_first_not_of("tu") != std::string::npos) {
    return invalid("invalid compiled CPU kernel signature");
  }
  return Status::Ok();
}

Expected<std::shared_ptr<CpuKernelModule>> resolve(const std::string& id) {
  std::lock_guard lock(registry_mutex);
  const auto found = registry.find(id);
  if (found == registry.end()) return invalid("unknown or expired CPU kernel artifact");
  auto module = found->second.lock();
  if (!module) return invalid("unknown or expired CPU kernel artifact");
  return module;
}

CpuTensor checked_tensor(const Tensor& tensor) {
  if (tensor.dtype != DType::kFloat32) {
    throw std::invalid_argument("compiled CPU kernels require float32 tensors");
  }
  const auto count = numel(tensor.shape);
  if (static_cast<std::uint64_t>(count) > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::invalid_argument("compiled CPU tensor byte size overflow");
  }
  return from_core_tensor(tensor);
}

Status validate_arguments(const CpuKernelModule& module,
                          std::span<const KernelArgument> arguments,
                          const Tensor& output, std::uint32_t threads,
                          std::uint32_t block_size) {
  const auto& signature = module.signature();
  if (arguments.size() != signature.kinds.size()) return invalid("CPU kernel argument count mismatch");
  if (block_size == 0) return invalid("CPU kernel block size must be positive");
  const CpuTensor out = checked_tensor(output);
  if (static_cast<std::uint64_t>(out.size()) < threads) return invalid("thread_count exceeds output size");
  for (std::size_t i = 0; i < arguments.size(); ++i) {
    const auto& argument = arguments[i];
    if (signature.kinds[i] == 'u') {
      if (argument.kind != KernelArgumentKind::kUInt32 || argument.tensor) {
        return invalid("CPU kernel scalar argument kind mismatch");
      }
    } else {
      if (argument.kind != KernelArgumentKind::kTensor || !argument.tensor) {
        return invalid("CPU kernel tensor argument kind mismatch");
      }
      const CpuTensor input = checked_tensor(*argument.tensor);
      if (input.shape() != out.shape()) return invalid("CPU kernel tensor shape mismatch");
      if (i == signature.output_index && input.buffer() != out.buffer()) {
        return invalid("CPU kernel output argument does not match output metadata");
      }
    }
  }
  if (arguments[signature.guard_index].uint32_value != threads) {
    return invalid("CPU kernel guard bound must match thread_count");
  }
  return Status::Ok();
}

CpuTensor run(const CpuKernelModule& module, std::span<const KernelArgument> arguments,
              const Tensor& output, std::uint32_t threads, std::uint32_t block_size) {
  const CpuTensor source = checked_tensor(output);
  CpuTensor result(source.shape(), source.float_data());  // private output copy
  if (threads == 0) return result;
  // Allocate all storage before storing any pointers; no vector may reallocate
  // while generated code holds an argument slot.
  std::vector<MemRef1D> descriptors(arguments.size());
  std::vector<std::uint32_t> scalars(arguments.size() + 2);
  std::vector<void*> slots(arguments.size() + 2);
  for (std::size_t i = 0; i < arguments.size(); ++i) {
    if (arguments[i].kind == KernelArgumentKind::kUInt32) {
      scalars[i] = arguments[i].uint32_value;
      slots[i] = &scalars[i];
    } else {
      const auto tensor = checked_tensor(*arguments[i].tensor);
      const auto& buffer = tensor.buffer() == source.buffer() ? result.buffer() : tensor.buffer();
      auto* data = buffer->mutable_float_data().data();
      descriptors[i] = {data, data, 0, tensor.size(), 1};
      slots[i] = &descriptors[i];
    }
  }
  scalars[arguments.size()] = threads;
  scalars[arguments.size() + 1] = block_size;
  slots[arguments.size()] = &scalars[arguments.size()];
  slots[arguments.size() + 1] = &scalars[arguments.size() + 1];
  module.invoke(slots.data());
  return result;
}
}  // namespace

bool compiled_kernel_supported() { return CORTEX_CPU_MODULE_SUPPORTED != 0; }

std::string kernel_manifest(const KernelSignature& signature) {
  return "cortex.cpu.v1|linux-x86_64|llvm-21.1.8|elementwise-f32-v1|" +
         signature.kinds + "|" + std::to_string(signature.output_index) + "|" +
         std::to_string(signature.guard_index);
}

void CpuKernelModule::LibraryDeleter::operator()(void* library) const noexcept {
#if CORTEX_CPU_MODULE_SUPPORTED
  if (library) (void)dlclose(library);
#else
  (void)library;
#endif
}

CpuKernelModule::~CpuKernelModule() {
  std::lock_guard lock(registry_mutex);
  if (!id_.empty()) registry.erase(id_);
}

Expected<std::shared_ptr<CpuKernelModule>> CpuKernelModule::load(
    const std::string& path, KernelSignature signature) {
  try {
    if (auto status = validate_signature(signature); !status.ok()) return status;
    if (path.empty() || path.find('\0') != std::string::npos) return invalid("invalid CPU kernel library path");
#if CORTEX_CPU_MODULE_SUPPORTED
    auto module = std::shared_ptr<CpuKernelModule>(new CpuKernelModule);
    module->library_.reset(dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL));
    if (!module->library_) return invalid(std::string("CPU kernel library load failed: ") + dlerror());
    auto manifest = reinterpret_cast<const char* (*)()>(dlsym(module->library_.get(), "cortex_manifest_v1"));
    module->entry_ = reinterpret_cast<void (*)(void**)>(dlsym(module->library_.get(), "cortex_launch_v1"));
    if (!manifest || !module->entry_) return invalid("CPU kernel library missing ABI symbols");
    const char* actual = manifest();
    if (!actual || kernel_manifest(signature) != actual) return invalid("CPU kernel ABI/manifest mismatch");
    module->signature_ = std::move(signature);
    module->id_ = "cpu-kernel-" + std::to_string(next_id.fetch_add(1));
    {
      std::lock_guard lock(registry_mutex);
      registry.emplace(module->id_, module);
    }
    return module;
#else
    return Status(StatusCode::kUnavailable, "compiled CPU kernels require Linux x86_64");
#endif
  } catch (const std::exception& error) {
    return Status(StatusCode::kInternal, error.what());
  }
}

Status execute_compiled_kernel(const BackendExecution& execution) {
  try {
    if (auto status = validate_kernel_execution_contract(execution); !status.ok()) return status;
    if (execution.outputs.size() != 1) return invalid("CPU kernel requires one output");
    const auto& target = *execution.compilation_target;
    if (target.artifact_kind != KernelArtifactKind::kBinary || target.entry_point != "cortex_launch_v1") {
      return invalid("unsupported CPU kernel artifact kind or entry point");
    }
    const auto& launch = *execution.launch;
    if (launch.grid_y != 1 || launch.grid_z != 1 ||
        launch.threads_per_group_y != 1 || launch.threads_per_group_z != 1) {
      return invalid("CPU kernel requires a one-dimensional launch");
    }
    auto module = resolve(target.artifact);
    if (!module) return module.status();
    if (auto status = validate_arguments(*module.value(), execution.kernel_arguments,
                                        execution.outputs[0], launch.grid_x,
                                        launch.threads_per_group_x); !status.ok()) return status;
    auto result = run(*module.value(), execution.kernel_arguments, execution.outputs[0],
                      launch.grid_x, launch.threads_per_group_x);
    execution.outputs[0] = to_core_tensor(result);
    return Status::Ok();
  } catch (const std::invalid_argument& error) {
    return invalid(error.what());
  } catch (const std::exception& error) {
    return {StatusCode::kInternal, error.what()};
  }
}

Expected<CpuTensor> launch_compiled_kernel(
    const std::shared_ptr<CpuKernelModule>& module,
    std::span<const KernelArgument> arguments, std::uint32_t threads,
    std::uint32_t block_size) {
  try {
    if (!module) return invalid("missing CPU kernel module");
    const auto index = module->signature().output_index;
    if (index >= arguments.size() || arguments[index].kind != KernelArgumentKind::kTensor ||
        !arguments[index].tensor) return invalid("missing CPU kernel output tensor");
    std::array<Tensor, 1> outputs{*arguments[index].tensor};
    if (threads == 0) {
      auto live = resolve(module->artifact_id());
      if (!live) return live.status();
      if (auto status = validate_arguments(*live.value(), arguments, outputs[0], threads, block_size);
          !status.ok()) return status;
      return run(*live.value(), arguments, outputs[0], threads, block_size);
    }
    BackendExecution execution{
        BackendOpClass::kKernel, {}, {}, outputs,
        LaunchConfig{threads, 1, 1, block_size, 1, 1},
        CompilationTarget{KernelArtifactKind::kBinary, module->artifact_id(), "cortex_launch_v1"},
        arguments};
    CpuBackend backend;
    if (auto status = backend.execute(execution); !status.ok()) return status;
    return from_core_tensor(outputs[0]);
  } catch (const std::invalid_argument& error) {
    return invalid(error.what());
  } catch (const std::exception& error) {
    return Status(StatusCode::kInternal, error.what());
  }
}
}  // namespace cortex::cpu
