#pragma once

#include "tensorcx/backends/vulkan/vulkan_backend.h"

namespace tensorcx::vulkan {
struct KernelSignature {
  std::string kinds;
  std::uint32_t output_index{}, guard_index{};
};
std::string kernel_manifest(const KernelSignature& signature);
Status compiled_kernel_support(int index = 0);
class VulkanKernelModule {
 public:
  ~VulkanKernelModule();
  VulkanKernelModule(const VulkanKernelModule&) = delete;
  VulkanKernelModule& operator=(const VulkanKernelModule&) = delete;
  static Expected<std::shared_ptr<VulkanKernelModule>> load(
      const std::string& artifact, KernelSignature signature, int index);
  const std::string& artifact_id() const { return id_; }
  const KernelSignature& signature() const { return signature_; }
  bool current_process() const;
  Expected<VulkanTensor> launch(std::span<const KernelArgument> arguments,
      const Tensor& output, std::uint32_t threads, std::uint32_t block) const;
 private:
  VulkanKernelModule();
  struct Impl;
  std::unique_ptr<Impl> impl_;
  KernelSignature signature_;
  std::string id_;
};
Status execute_compiled_kernel(const BackendExecution& execution);
Expected<VulkanTensor> launch_compiled_kernel(const std::shared_ptr<VulkanKernelModule>& module,
    std::span<const KernelArgument> arguments, std::uint32_t threads, std::uint32_t block);
}  // namespace tensorcx::vulkan
