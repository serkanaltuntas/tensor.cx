#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "cortex/core/operation.h"
#include "cortex/core/status.h"
#include "cortex/core/tensor.h"

namespace cortex {

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

struct LaunchConfig {
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

struct BackendExecution {
  BackendOpClass op_class{BackendOpClass::kPrimitive};
  OpDesc op;
  std::span<const Tensor> inputs;
  std::span<Tensor> outputs;
  std::optional<LaunchConfig> launch;
  std::optional<CompilationTarget> compilation_target;
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

}  // namespace cortex
