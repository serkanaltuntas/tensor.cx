#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "tensorcx/core/operation.h"
#include "tensorcx/core/status.h"
#include "tensorcx/core/tensor.h"

namespace tensorcx {

enum class BackendOpClass {
  kPrimitive,
  kKernel,
};

enum class KernelArtifactKind {
  kNone,
  kStaticLibrary,
  kSource,
  kBinary,
  kIntermediateRepresentation,
};

enum class KernelArgumentKind {
  kTensor,
  kUInt32,
};

struct LaunchConfig {
  // Backend-neutral global work-item dimensions, not native CUDA block counts
  // or Metal threadgroup counts. Backends derive native grid/block counts from
  // these dimensions and the threads-per-group values.
  std::uint32_t grid_x{0};
  std::uint32_t grid_y{1};
  std::uint32_t grid_z{1};
  std::uint32_t threads_per_group_x{0};
  std::uint32_t threads_per_group_y{1};
  std::uint32_t threads_per_group_z{1};
};

struct CompilationTarget {
  KernelArtifactKind artifact_kind{KernelArtifactKind::kNone};
  std::string artifact;
  std::string entry_point;
};

struct KernelArgument {
  KernelArgumentKind kind{KernelArgumentKind::kUInt32};
  const Tensor* tensor{nullptr};
  std::uint32_t uint32_value{0};
};

struct BackendExecution {
  BackendOpClass op_class{BackendOpClass::kPrimitive};
  OpDesc op;
  std::span<const Tensor> inputs;
  std::span<Tensor> outputs;
  std::optional<LaunchConfig> launch;
  std::optional<CompilationTarget> compilation_target;
  std::span<const KernelArgument> kernel_arguments;
};

class Backend {
 public:
  virtual ~Backend() = default;

  virtual std::string name() const = 0;
  virtual Status execute(const BackendExecution& execution) = 0;
};

Status validate_fill_output_descriptor(
    const Tensor& output,
    std::string_view expected_device_type = {});

Status validate_primitive_execution_contract(
    const BackendExecution& execution,
    std::string_view fill_device_type = {});

Status validate_kernel_execution_contract(const BackendExecution& execution);

}  // namespace tensorcx
