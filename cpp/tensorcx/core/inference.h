#pragma once
#include "tensorcx/core/backend.h"
namespace tensorcx {
bool is_inference_composite(OpKind kind);
// These composites only call the supplied backend through its public execution ABI.
Status execute_inference_composite(Backend& backend, const BackendExecution& execution);
struct EmbeddingPlan { Shape shape; Dim indices, vocabulary, width; };
EmbeddingPlan make_embedding_plan(std::span<const Tensor> inputs);
struct AttentionSoftmaxPlan {
  Shape shape, metadata;
  Dim rows, columns;
  float scale;
  std::uint32_t mask_kind;  // 0 absent, 1 boolean, 2 additive float32.
};
AttentionSoftmaxPlan make_attention_softmax_plan(const OpDesc& op, std::span<const Tensor> inputs);
}
