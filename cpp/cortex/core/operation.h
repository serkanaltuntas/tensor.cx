#pragma once

namespace cortex {

enum class OpKind {
  kFill,
  kAdd,
  kMultiply,
  kMatmul,
};

struct OpDesc {
  OpKind kind;
};

}  // namespace cortex
