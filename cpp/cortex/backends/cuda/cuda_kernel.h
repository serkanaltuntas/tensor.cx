#pragma once
#include <memory>
#include <string>
#include "cortex/backends/cuda/cuda_backend.h"

namespace cortex::cuda {
struct KernelSignature {
  std::string kinds;
  std::uint32_t output_index{0}, guard_index{0};
};
std::string kernel_manifest(const KernelSignature& signature, const std::string& operation = "add");
Status compiled_kernel_support();
class CudaKernelModule {
 public:
  ~CudaKernelModule();
  CudaKernelModule(const CudaKernelModule&) = delete;
  CudaKernelModule& operator=(const CudaKernelModule&) = delete;
  static Expected<std::shared_ptr<CudaKernelModule>> load(
      const std::string& ptx, const std::string& entry, KernelSignature signature);
  const std::string& artifact_id() const { return id_; }
  const std::string& entry_point() const { return entry_; }
  const KernelSignature& signature() const { return signature_; }
  Expected<CudaTensor> launch(std::span<const KernelArgument> arguments,
                             const Tensor& output, std::uint32_t threads,
                             std::uint32_t block) const;
 private:
  CudaKernelModule();
  struct Impl;
  std::unique_ptr<Impl> impl_;
  KernelSignature signature_;
  std::string id_, entry_;
};
Status execute_compiled_kernel(const BackendExecution& execution);
Expected<CudaTensor> launch_compiled_kernel(
    const std::shared_ptr<CudaKernelModule>& module, std::span<const KernelArgument> arguments,
    std::uint32_t threads, std::uint32_t block);
}  // namespace cortex::cuda
