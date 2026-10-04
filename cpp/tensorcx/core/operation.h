#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include "tensorcx/core/dtype.h"

namespace tensorcx {

enum class OpKind {
  kFill,
  kAdd,
  kMultiply,
  kMatmul,
  kSum,
  kMax,
  kMean,
  kExp,
  kGelu,
  kSilu,
  kSoftmax,
  kRmsNorm,
  kLayerNorm,
  kSubtract,
  kDivide,
  kNegate,
  kAddScalar,
  kSubtractScalar,
  kMultiplyScalar,
  kDivideScalar,
  kCast,
};

enum class MatmulPreference {
  kAuto,
  kCustom,
  kOptimized,
};

// §5.6 specifies OpDesc as "an op enum plus attributes". Through Phase 5 it
// carried only the kind: it tagged the per-op entry points while op parameters
// were still passed as explicit function arguments. Phase 6 added the minimal
// attribute surface needed for reductions: axis is used by reduction entry
// points and axis-aware transforms such as softmax/rmsnorm/layernorm, epsilon
// is used by normalization ops, and both are ignored by ops that do not need
// them. Phase 8 adds scalar_value for fill so allocation-style primitive
// execution can move through BackendExecution without adding a per-op virtual.
// MatmulPreference keeps matmul algorithm selection backend-neutral while
// preserving public auto/custom/optimized routing.
struct OpDesc {
  OpKind kind{OpKind::kFill};
  std::int64_t axis{0};
  double epsilon{1.0e-5};
  double scalar_value{0.0};
  MatmulPreference matmul_preference{MatmulPreference::kAuto};
  // Scalar arithmetic reuses scalar_value; true places it left of the tensor.
  bool scalar_left{false};
  DType target_dtype{DType::kFloat32};
};

struct PrimitiveOpSchema {
  std::size_t input_count{0};
  std::size_t output_count{0};
};

// Returns the primitive operation shape expressible by BackendExecution today.
// Fill has no inputs and uses output[0] as the backend-neutral allocation
// descriptor: dtype, shape, device, contiguous strides, no buffer, and offset 0.
std::optional<PrimitiveOpSchema> primitive_op_schema(OpKind kind);

}  // namespace tensorcx
