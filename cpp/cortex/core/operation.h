#pragma once

namespace cortex {

enum class OpKind {
  kFill,
  kAdd,
  kMultiply,
};

struct OpDesc {
  OpKind kind;
};

}  // namespace cortex
