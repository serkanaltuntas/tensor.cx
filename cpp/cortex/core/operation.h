#pragma once

#include <cstdint>

namespace cortex {

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
};

// §5.6 specifies OpDesc as "an op enum plus attributes". Through Phase 5 it
// carried only the kind: it tagged the per-op entry points (execute_binary,
// fill), while op parameters (fill value, matmul backend selection) were still
// passed as explicit function arguments. Phase 6 starts the minimal attribute
// surface needed for reductions without taking the full Phase 8 dispatch
// migration: axis is used by reduction entry points and axis-aware transforms
// such as softmax/rmsnorm, epsilon is used by normalization ops, and both are
// ignored by ops that do not need them.
struct OpDesc {
  OpKind kind;
  std::int64_t axis{0};
  double epsilon{1.0e-5};
};

}  // namespace cortex
