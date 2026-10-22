#include "tensorcx/backends/vulkan/vulkan_nn.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "tensorcx/backends/vulkan/vulkan_context.h"
#include "tensorcx/backends/vulkan/vulkan_tensor.h"
#include "tensorcx/core/inference.h"

namespace tensorcx::vulkan {
namespace {
std::uint32_t narrow(Dim value) {
  if (value < 0 || static_cast<std::uint64_t>(value) > UINT32_MAX)
    throw std::invalid_argument("Vulkan neural-network metadata exceeds uint32 limits");
  return static_cast<std::uint32_t>(value);
}
std::uint32_t epsilon_bits(double value) {
  if (!std::isfinite(value) || value < 0 ||
      value > static_cast<double>(std::numeric_limits<float>::max()))
    throw std::invalid_argument("epsilon must be finite, non-negative and representable as float32");
  const auto rounded = static_cast<float>(value);
  if (value > 0 && rounded == 0)
    throw std::invalid_argument("epsilon must be finite, non-negative and representable as float32");
  return std::bit_cast<std::uint32_t>(rounded);
}
std::uint32_t operation(OpKind kind) {
  switch (kind) {
    case OpKind::kExp: return 70;
    case OpKind::kGelu: return 71;
    case OpKind::kSilu: return 72;
    case OpKind::kSoftmax: return 73;
    case OpKind::kRmsNorm: return 74;
    case OpKind::kLayerNorm: return 75;
    default: throw std::invalid_argument("unsupported Vulkan neural-network operation");
  }
}
}  // namespace

bool is_nn_operation(OpKind kind) {
  return is_inference_composite(kind) || kind == OpKind::kEmbedding ||
      kind == OpKind::kAttentionSoftmax || kind == OpKind::kExp || kind == OpKind::kGelu ||
      kind == OpKind::kSilu || kind == OpKind::kSoftmax || kind == OpKind::kRmsNorm ||
      kind == OpKind::kLayerNorm;
}

Status execute_nn(Backend& backend, const BackendExecution& e) {
  try {
    std::vector<VulkanTensor> inputs;
    for (const auto& descriptor : e.inputs) {
      auto converted = from_core_tensor(descriptor);
      if (!converted) return converted.status();
      inputs.push_back(converted.move_value());
      if (inputs.back().buffer()->owner() != inputs.front().buffer()->owner())
        throw std::invalid_argument("Vulkan inference requires the same device owner");
    }
    // Validate every concrete input before any shared composite can allocate,
    // including masks, affine parameters and inputs with no visited elements.
    if (is_inference_composite(e.op.kind)) return execute_inference_composite(backend, e);
    auto context = inputs[0].buffer()->owner();
    const auto& source = inputs[0].buffer()->allocation();
    if (e.op.kind == OpKind::kEmbedding) {
      const auto plan = make_embedding_plan(e.inputs);
      const auto indices = narrow(plan.indices), count = narrow(numel(plan.shape));
      if (indices) {
        // Only a private four-byte validation result crosses back to the host.
        // Validate indices even for a zero-width output. Vocabularies >= 2^31
        // admit every nonnegative int32 without narrowing an unused wide axis.
        auto flag = context->allocate(sizeof(std::uint32_t));
        const std::uint32_t zero = 0;
        context->upload(flag, &zero, sizeof(zero));
        const auto bound = static_cast<std::uint32_t>(std::min<Dim>(plan.vocabulary, INT64_C(2147483648)));
        context->compute({source, flag, flag, flag}, {indices, 76u, bound, 0, 0, 0, 0});
        std::uint32_t invalid = 0;
        context->download(flag, &invalid, sizeof(invalid));
        if (invalid) throw std::invalid_argument("embedding index out of range");
      }
      if (count) (void)narrow(inputs[1].size());
      auto allocated = empty(plan.shape, DType::kFloat32, inputs[0].device().index);
      if (!allocated) return allocated.status();
      auto output = allocated.move_value();
      if (count) {
        const auto& weights = inputs[1].buffer()->allocation();
        const auto& target = output.buffer()->allocation();
        context->compute({source, weights, target, target}, {count, 77u, narrow(plan.width), 0, 0, 0, 0});
      }
      e.outputs[0] = to_core_tensor(output);
      return Status::Ok();
    }

    Shape shape = inputs[0].shape();
    std::vector<std::uint32_t> words;
    std::uint32_t count = 0, groups = 0, reduce = 0, inner = 0, rank = 0, flags = 0, eps = 0;
    std::uint32_t code = 0;
    if (e.op.kind == OpKind::kAttentionSoftmax) {
      const auto plan = make_attention_softmax_plan(e.op, e.inputs);
      shape = plan.shape;
      count = narrow(numel(shape)); groups = narrow(plan.rows); code = 78;
      if (count) {
        for (const auto& input : inputs) (void)narrow(input.size());
        for (auto value : plan.metadata) words.push_back(narrow(value));
        reduce = narrow(plan.columns); rank = narrow(shape.size());
      }
      inner = std::bit_cast<std::uint32_t>(plan.scale);
      flags = plan.mask_kind | (static_cast<std::uint32_t>(e.op.causal) << 2u);
    } else {
      code = operation(e.op.kind);
      if (inputs[0].dtype() != DType::kFloat32)
        throw std::invalid_argument("Vulkan activations and normalization require float32 tensors");
      count = narrow(inputs[0].size()); groups = count;
      if (code >= 73) {
        auto axis = e.op.axis;
        const auto dimensions = static_cast<Dim>(shape.size());
        if (!dimensions) {
          if (axis != 0 && axis != -1) throw std::invalid_argument("normalization axis out of range");
        } else {
          if (axis < 0) axis += dimensions;
          if (axis < 0 || axis >= dimensions) throw std::invalid_argument("normalization axis out of range");
        }
        if (code >= 74) eps = epsilon_bits(e.op.epsilon);
        if (count) {
          reduce = dimensions ? narrow(shape[axis]) : 1u;
          inner = dimensions ? narrow(inputs[0].strides()[axis]) : 1u;
          groups = count / reduce;
          words.push_back(std::bit_cast<std::uint32_t>(static_cast<float>(reduce)));
        }
      }
    }
    if (words.empty()) words.push_back(0);
    auto allocated = empty(shape, DType::kFloat32, inputs[0].device().index);
    if (!allocated) return allocated.status();
    auto output = allocated.move_value();
    if (groups) {
      auto meta = context->allocate(words.size() * sizeof(std::uint32_t));
      context->upload(meta, words.data(), words.size() * sizeof(std::uint32_t));
      const auto& right = inputs.size() == 2 ? inputs[1].buffer()->allocation() : source;
      context->compute({source, right, output.buffer()->allocation(), meta},
                       {groups, code, reduce, inner, 0, rank, flags, eps});
    }
    e.outputs[0] = to_core_tensor(output);
    return Status::Ok();
  } catch (const std::invalid_argument& error) {
    return Status(StatusCode::kInvalidArgument, error.what());
  } catch (const std::exception& error) {
    return Status(StatusCode::kInternal, error.what());
  }
}
}  // namespace tensorcx::vulkan
