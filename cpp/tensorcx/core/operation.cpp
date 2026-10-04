#include "tensorcx/core/operation.h"

namespace tensorcx {

std::optional<PrimitiveOpSchema> primitive_op_schema(OpKind kind) {
  switch (kind) {
    case OpKind::kAdd:
    case OpKind::kMultiply:
    case OpKind::kSubtract:
    case OpKind::kDivide:
    case OpKind::kMatmul:
    case OpKind::kEqual:
    case OpKind::kNotEqual:
    case OpKind::kLess:
    case OpKind::kLessEqual:
    case OpKind::kGreater:
    case OpKind::kGreaterEqual:
    case OpKind::kLogicalAnd:
    case OpKind::kLogicalOr:
    case OpKind::kLogicalXor:
    case OpKind::kMaskedSelect:
      return PrimitiveOpSchema{2, 1};
    case OpKind::kLog: case OpKind::kSqrt: case OpKind::kAbs:
    case OpKind::kMin: case OpKind::kArgmax:
    case OpKind::kSum:
    case OpKind::kMax:
    case OpKind::kMean:
    case OpKind::kExp:
    case OpKind::kGelu:
    case OpKind::kSilu:
    case OpKind::kSoftmax:
    case OpKind::kRmsNorm:
    case OpKind::kLayerNorm:
    case OpKind::kNegate:
    case OpKind::kAddScalar:
    case OpKind::kSubtractScalar:
    case OpKind::kMultiplyScalar:
    case OpKind::kDivideScalar:
    case OpKind::kCast:
    case OpKind::kTranspose:
    case OpKind::kSlice:
    case OpKind::kLogicalNot:
    case OpKind::kAny:
    case OpKind::kAll:
      return PrimitiveOpSchema{1, 1};
    case OpKind::kLinear: return PrimitiveOpSchema{2, 1, true};
    case OpKind::kAffineRmsNorm: case OpKind::kAffineLayerNorm:
    case OpKind::kAttentionSoftmax: return PrimitiveOpSchema{1, 1, true};
    case OpKind::kEmbedding: return PrimitiveOpSchema{2, 1};
    case OpKind::kAttention: return PrimitiveOpSchema{3, 1, true};
    case OpKind::kTopK: return PrimitiveOpSchema{1, 2};
    case OpKind::kClip:
    case OpKind::kWhere:
      return PrimitiveOpSchema{3, 1};
    case OpKind::kConcat:
      return PrimitiveOpSchema{1, 1, true};
    case OpKind::kFill:
      return PrimitiveOpSchema{0, 1};
  }
  return std::nullopt;
}

}  // namespace tensorcx
