#pragma once
#include "tensorcx/core/backend.h"
namespace tensorcx {
bool is_math_operation(OpKind kind);
// Kernel operation codes are shared by host dispatch and static GPU kernels.
struct MathPlan {
  Shape shape, metadata;
  DType dtype;
  Dim groups{0}, reduce{1}, inner{1}, k{1};
  std::size_t rank{0};
  std::uint32_t code{0};
};
MathPlan make_math_plan(const OpDesc& op, std::span<const Tensor> inputs);
}
