#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cortex/backends/cpu/cpu_tensor.h"
#include "cortex/core/backend.h"
#include "cortex/core/expected.h"

namespace cortex::cpu {

struct KernelSignature {
  std::string kinds;  // t: float32 tensor, u: uint32 scalar, in parameter order
  std::uint32_t output_index{0};
  std::uint32_t guard_index{0};
};

std::string kernel_manifest(const KernelSignature& signature);
bool compiled_kernel_supported();

class CpuKernelModule {
 public:
  ~CpuKernelModule();
  CpuKernelModule(const CpuKernelModule&) = delete;
  CpuKernelModule& operator=(const CpuKernelModule&) = delete;
  const std::string& artifact_id() const { return id_; }
  const KernelSignature& signature() const { return signature_; }

  static Expected<std::shared_ptr<CpuKernelModule>> load(
      const std::string& path, KernelSignature signature);
  void invoke(void** arguments) const { entry_(arguments); }

 private:
  CpuKernelModule() = default;
  struct LibraryDeleter { void operator()(void* library) const noexcept; };
  std::unique_ptr<void, LibraryDeleter> library_;
  void (*entry_)(void**){nullptr};
  KernelSignature signature_;
  std::string id_;
};

// Called only by CpuBackend::execute after selecting the kernel op class.
Status execute_compiled_kernel(const BackendExecution& execution);
// Typed binding entry; zero work validates the module/arguments but does not
// manufacture a non-zero backend launch or invoke generated code.
Expected<CpuTensor> launch_compiled_kernel(
    const std::shared_ptr<CpuKernelModule>& module,
    std::span<const KernelArgument> arguments, std::uint32_t threads,
    std::uint32_t block_size);

}  // namespace cortex::cpu
