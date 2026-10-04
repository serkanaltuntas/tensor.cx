#include "tensorcx/core/inference.h"
#include "tensorcx/core/expected.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace tensorcx {
namespace {
void require_float(const Tensor& t) {
  if (t.dtype != DType::kFloat32) throw std::invalid_argument("inference values require float32 tensors");
  (void)numel(t.shape); (void)contiguous_strides(t.shape);
}
float checked_scale(double scale) {
  const float narrowed = static_cast<float>(scale);
  if (!std::isfinite(scale) || !std::isfinite(narrowed) || (scale != 0 && narrowed == 0))
    throw std::invalid_argument("attention scale must be finite and representable as float32");
  return narrowed;
}
Tensor view(const Tensor& input, Shape shape) {
  if (numel(input.shape) != numel(shape)) throw std::invalid_argument("inference view size mismatch");
  Tensor result = input; result.shape = std::move(shape); result.strides = contiguous_strides(result.shape);
  return result;
}
Expected<Tensor> run(Backend& backend, OpDesc op, std::vector<Tensor> inputs) {
  std::array<Tensor, 1> outputs;
  BackendExecution e{BackendOpClass::kPrimitive, std::move(op), inputs, outputs, std::nullopt, std::nullopt, {}};
  auto status = backend.execute(e);
  if (!status.ok()) return status;
  return std::move(outputs[0]);
}
Expected<Tensor> transpose_last(Backend& backend, const Tensor& input) {
  OpDesc op{OpKind::kTranspose}; op.axes.resize(input.shape.size());
  std::iota(op.axes.begin(), op.axes.end(), 0);
  std::swap(op.axes[op.axes.size()-1], op.axes[op.axes.size()-2]);
  return run(backend, op, {input});
}
}

bool is_inference_composite(OpKind kind) {
  return kind == OpKind::kLinear || kind == OpKind::kAffineRmsNorm ||
         kind == OpKind::kAffineLayerNorm || kind == OpKind::kAttention;
}

EmbeddingPlan make_embedding_plan(std::span<const Tensor> inputs) {
  if (inputs.size() != 2) throw std::invalid_argument("embedding requires indices and weight");
  const auto& indices = inputs[0]; const auto& weight = inputs[1];
  if (indices.dtype != DType::kInt32) throw std::invalid_argument("embedding indices must be int32");
  (void)numel(indices.shape); (void)contiguous_strides(indices.shape);
  require_float(weight);
  if (weight.shape.size() != 2) throw std::invalid_argument("embedding weight must have rank 2");
  Shape shape = indices.shape; shape.push_back(weight.shape[1]);
  (void)numel(shape); (void)contiguous_strides(shape);
  return {shape, numel(indices.shape), weight.shape[0], weight.shape[1]};
}

AttentionSoftmaxPlan make_attention_softmax_plan(const OpDesc& op, std::span<const Tensor> inputs) {
  if (inputs.empty() || inputs.size() > 2) throw std::invalid_argument("attention softmax requires scores and optional mask");
  require_float(inputs[0]);
  if (inputs[0].shape.size() < 2 || op.attention_shape.size() < 2)
    throw std::invalid_argument("attention scores must have rank >= 2");
  const auto& shape = op.attention_shape;
  if (make_broadcast_plan(inputs[0].shape, shape).output_shape != shape)
    throw std::invalid_argument("attention scores cannot broadcast to target");
  std::uint32_t mask_kind = 0;
  if (inputs.size() == 2) {
    const auto& mask = inputs[1];
    if (mask.dtype != DType::kBool && mask.dtype != DType::kFloat32)
      throw std::invalid_argument("attention mask must be bool or float32");
    if (make_broadcast_plan(mask.shape, shape).output_shape != shape)
      throw std::invalid_argument("attention mask cannot broadcast to scores");
    mask_kind = mask.dtype == DType::kBool ? 1 : 2;
  }
  const auto count = numel(shape); (void)contiguous_strides(shape);
  Shape metadata = shape;
  for (const auto& t : inputs) {
    auto strides = make_broadcast_plan(t.shape, shape).lhs_strides;
    metadata.insert(metadata.end(), strides.begin(), strides.end());
  }
  const Dim columns = shape.back();
  return {shape, metadata, columns ? count / columns : 0, columns,
          checked_scale(op.attention_scale.value_or(1.0)), mask_kind};
}

Status execute_inference_composite(Backend& backend, const BackendExecution& e) {
  // Concrete backends validate all native buffer/device metadata before entering.
  const auto& op = e.op;
  if (op.kind == OpKind::kLinear) {
    if (e.inputs.size() < 2 || e.inputs.size() > 3) throw std::invalid_argument("linear requires input, weight and optional bias");
    for (const auto& t : e.inputs) require_float(t);
    const auto& x=e.inputs[0]; const auto& w=e.inputs[1];
    if (x.shape.empty() || w.shape.size()!=2 || x.shape.back()!=w.shape[1])
      throw std::invalid_argument("linear requires (..., in_features) and (out_features, in_features)");
    if (e.inputs.size()==3 && e.inputs[2].shape!=Shape{w.shape[0]})
      throw std::invalid_argument("linear bias must have shape (out_features,)");
    Shape shape=x.shape;shape.back()=w.shape[0];(void)numel(shape);(void)contiguous_strides(shape);
    auto transposed=transpose_last(backend,w);if(!transposed)return transposed.status();
    auto result=run(backend,OpDesc{OpKind::kMatmul},{x,transposed.move_value()});if(!result)return result.status();
    if(e.inputs.size()==3){result=run(backend,OpDesc{OpKind::kAdd},{result.move_value(),e.inputs[2]});if(!result)return result.status();}
    e.outputs[0]=result.move_value();return Status::Ok();
  }
  if (op.kind==OpKind::kAffineRmsNorm || op.kind==OpKind::kAffineLayerNorm) {
    if(e.inputs.size()!=1U+op.has_weight+op.has_bias || (op.kind==OpKind::kAffineRmsNorm && op.has_bias))
      throw std::invalid_argument("invalid affine normalization parameters");
    for(const auto& t:e.inputs)require_float(t);
    const auto& x=e.inputs[0];Dim axis=op.axis;
    if(x.shape.empty()) {if(axis!=0 && axis!=-1)throw std::invalid_argument("normalization axis out of range");axis=0;}
    else {if(axis<0)axis+=x.shape.size();if(axis<0 || axis>=static_cast<Dim>(x.shape.size()))throw std::invalid_argument("normalization axis out of range");}
    const Shape expected=x.shape.empty()?Shape{}:Shape{x.shape[axis]};
    for(std::size_t i=1;i<e.inputs.size();++i)if(e.inputs[i].shape!=expected)throw std::invalid_argument("affine parameters must match the normalized axis");
    const float eps=static_cast<float>(op.epsilon);
    if(!std::isfinite(op.epsilon)||!std::isfinite(eps)||op.epsilon<0||(op.epsilon>0 && eps==0))
      throw std::invalid_argument("epsilon must be finite, non-negative and representable as float32");
    OpDesc base{op.kind==OpKind::kAffineRmsNorm?OpKind::kRmsNorm:OpKind::kLayerNorm,op.axis,op.epsilon};
    auto result=run(backend,base,{x});if(!result)return result.status();
    Shape broadcast(x.shape.size(),1);if(!broadcast.empty())broadcast[axis]=x.shape[axis];
    std::size_t next=1;
    if(op.has_weight){result=run(backend,OpDesc{OpKind::kMultiply},{result.move_value(),view(e.inputs[next++],broadcast)});if(!result)return result.status();}
    if(op.has_bias){result=run(backend,OpDesc{OpKind::kAdd},{result.move_value(),view(e.inputs[next],broadcast)});if(!result)return result.status();}
    e.outputs[0]=result.move_value();return Status::Ok();
  }
  if(op.kind!=OpKind::kAttention || e.inputs.size()<3 || e.inputs.size()>4)
    throw std::invalid_argument("attention requires query, key, value and optional mask");
  for(std::size_t i=0;i<3;++i){require_float(e.inputs[i]);if(e.inputs[i].shape.size()<2)throw std::invalid_argument("attention tensors require rank >= 2");}
  const auto& q=e.inputs[0];const auto& k=e.inputs[1];const auto& v=e.inputs[2];
  const Dim features=q.shape.back(),rows=q.shape[q.shape.size()-2],columns=k.shape[k.shape.size()-2];
  if(features<=0 || features!=k.shape.back() || columns!=v.shape[v.shape.size()-2])
    throw std::invalid_argument("attention query/key features and key/value sequence sizes must match; features must be positive");
  Shape batch;
  // Validate only complete shapes: empty matrix axes can make an otherwise
  // overflowing batch product unvisited, just as in batched matmul.
  for (std::size_t i = 0; i < 3; ++i) {
    const auto& input = e.inputs[i].shape;
    const auto rank = std::max(batch.size(), input.size() - 2);
    Shape merged(rank, 1);
    for (std::size_t a = 0; a < rank; ++a) {
      const Dim left = a < rank - batch.size() ? 1 : batch[a - (rank - batch.size())];
      const Dim right = a < rank - (input.size() - 2) ? 1 : input[a - (rank - (input.size() - 2))];
      if (left != right && left != 1 && right != 1)
        throw std::invalid_argument("attention batch dimensions cannot broadcast");
      merged[a] = left == 1 ? right : left;
    }
    batch = std::move(merged);
  }
  Shape scores=batch;scores.push_back(rows);scores.push_back(columns);
  Shape shape=batch;shape.push_back(rows);shape.push_back(v.shape.back());
  (void)numel(scores);(void)contiguous_strides(scores);(void)numel(shape);(void)contiguous_strides(shape);
  OpDesc softmax{OpKind::kAttentionSoftmax};softmax.attention_shape=scores;softmax.causal=op.causal;
  softmax.attention_scale=checked_scale(op.attention_scale.value_or(1.0/std::sqrt(static_cast<double>(features))));
  // Validate mask broadcasting/dtype before running any stage, including empty outputs.
  Tensor descriptor=q;descriptor.shape=scores;descriptor.strides=contiguous_strides(scores);
  std::vector<Tensor> preview{descriptor};if(e.inputs.size()==4)preview.push_back(e.inputs[3]);
  (void)make_attention_softmax_plan(softmax,preview);
  auto transposed=transpose_last(backend,k);if(!transposed)return transposed.status();
  auto product=run(backend,OpDesc{OpKind::kMatmul},{q,transposed.move_value()});if(!product)return product.status();
  std::vector<Tensor> softmax_inputs{product.move_value()};if(e.inputs.size()==4)softmax_inputs.push_back(e.inputs[3]);
  auto probabilities=run(backend,softmax,std::move(softmax_inputs));if(!probabilities)return probabilities.status();
  auto result=run(backend,OpDesc{OpKind::kMatmul},{probabilities.move_value(),v});if(!result)return result.status();
  e.outputs[0]=result.move_value();return Status::Ok();
}
}
