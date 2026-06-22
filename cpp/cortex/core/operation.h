#pragma once

namespace cortex {

enum class OpKind {
  kFill,
  kAdd,
  kMultiply,
  kMatmul,
};

// §5.6 specifies OpDesc as "an op enum plus attributes". Through Phase 5 it
// carries only the kind: it tags the per-op entry points (execute_binary,
// fill), while op parameters (fill value, matmul backend selection) are still
// passed as explicit function arguments. Attributes are added here in Phase 8
// when dispatch is unified onto Backend::execute (see backend.h).
struct OpDesc {
  OpKind kind;
};

}  // namespace cortex
