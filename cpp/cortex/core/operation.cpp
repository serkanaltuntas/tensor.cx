#include "cortex/core/operation.h"

namespace cortex {

std::optional<PrimitiveOpSchema> primitive_op_schema(OpKind kind) {
  switch (kind) {
    case OpKind::kAdd:
    case OpKind::kMultiply:
    case OpKind::kMatmul:
      return PrimitiveOpSchema{2, 1};
    case OpKind::kSum:
    case OpKind::kMax:
    case OpKind::kMean:
    case OpKind::kExp:
    case OpKind::kGelu:
    case OpKind::kSilu:
    case OpKind::kSoftmax:
    case OpKind::kRmsNorm:
    case OpKind::kLayerNorm:
      return PrimitiveOpSchema{1, 1};
    case OpKind::kFill:
      return std::nullopt;
  }
  return std::nullopt;
}

}  // namespace cortex
