#pragma once
#include <algorithm>
#include <initializer_list>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "tensorcx/backends/cpu/cpu_backend.h"

// Exercise the native ABI directly, independently of Python's validation.
template<typename Upload, typename Download>
void inference_contract(tensorcx::Backend& backend, Upload upload, Download download) {
  using namespace tensorcx;
  auto check = [](bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
  };
  cpu::CpuTensor values({2, 2}, std::vector<float>{1.0F, -0.5F, 2.0F, 3.0F});
  cpu::CpuTensor weights({2, 2}, std::vector<float>{0.5F, 1.0F, -0.5F, 2.0F});
  cpu::CpuTensor vector({2}, std::vector<float>{1.5F, -0.5F});
  cpu::CpuTensor indices(DType::kInt32, {2});
  std::ranges::copy(std::initializer_list<std::int32_t>{1, 0}, indices.mutable_int32_data().begin());
  cpu::CpuTensor mask(DType::kBool, {2, 2});
  std::ranges::copy(std::initializer_list<std::uint8_t>{1, 0, 0, 0}, mask.mutable_bool_data().begin());
  for (auto kind : {OpKind::kLinear, OpKind::kAffineRmsNorm, OpKind::kAffineLayerNorm,
                   OpKind::kEmbedding, OpKind::kAttention, OpKind::kAttentionSoftmax}) {
    OpDesc op{kind};
    std::vector<cpu::CpuTensor> sources;
    if (kind == OpKind::kLinear) sources = {values, weights, vector};
    if (kind == OpKind::kAffineRmsNorm || kind == OpKind::kAffineLayerNorm) {
      op.axis = -1; op.epsilon = 1e-5; op.has_weight = true;
      sources = {values, vector};
      if (kind == OpKind::kAffineLayerNorm) { op.has_bias = true; sources.push_back(vector); }
    }
    if (kind == OpKind::kEmbedding) sources = {indices, weights};
    if (kind == OpKind::kAttention) { sources = {values, weights, values, mask}; op.causal = true; }
    if (kind == OpKind::kAttentionSoftmax) {
      sources = {values, mask}; op.attention_shape = {2, 2}; op.attention_scale = 0.5;
    }
    std::vector<Tensor> inputs, cpu_inputs;
    for (const auto& source : sources) { inputs.push_back(upload(source)); cpu_inputs.push_back(cpu::to_core_tensor(source)); }
    std::array<Tensor, 1> outputs{inputs[0]}, reference;
    BackendExecution e{BackendOpClass::kPrimitive, op, inputs, outputs, std::nullopt, std::nullopt, {}};
    BackendExecution ce{BackendOpClass::kPrimitive, op, cpu_inputs, reference, std::nullopt, std::nullopt, {}};
    cpu::CpuBackend cpu_backend;
    auto status = cpu_backend.execute(ce); check(status.ok(), status.message().c_str());
    status = backend.execute(e); check(status.ok(), status.message().c_str());
    const auto actual = download(outputs[0]), expected = cpu::from_core_tensor(reference[0]);
    check(actual.shape() == expected.shape() && actual.dtype() == expected.dtype(), "inference metadata mismatch");
    check(outputs[0].buffer != inputs[0].buffer, "inference output aliases input");
    for (Dim i = 0; i < actual.size(); ++i)
      check(std::abs(actual.float_data()[i] - expected.float_data()[i]) < 1e-4F, "inference CPU parity");
    const auto saved = outputs[0];
    auto invalid = [&] {
      check(backend.execute(e).code() == StatusCode::kInvalidArgument, "inference accepted invalid descriptor");
      check(outputs[0].buffer == saved.buffer && outputs[0].shape == saved.shape, "inference failure published output");
    };
    // Optional parameters must receive the same native validation as the input.
    for (std::size_t input = 0; input < inputs.size(); ++input) {
      const auto original = inputs[input];
      for (int scenario = 0; scenario < 6; ++scenario) {
        inputs[input] = original;
        switch (scenario) {
          case 0: inputs[input].buffer.reset(); break;
          case 1: inputs[input].offset = 1; break;
          case 2: inputs[input].device.index = 1; break;
          case 3: inputs[input].shape = {-1}; break;
          case 4: inputs[input].strides.clear(); break;
          case 5: inputs[input].shape = {999}; inputs[input].strides = {1}; break;
        }
        invalid();
      }
      inputs[input] = original;
    }
    e.inputs = {}; invalid(); e.inputs = inputs;
    e.outputs = {};
    check(backend.execute(e).code() == StatusCode::kInvalidArgument, "inference output arity");
    e.outputs = outputs;
    if (kind == OpKind::kEmbedding) {
      cpu::CpuTensor bad(DType::kInt32, {2}); std::ranges::copy(std::initializer_list<std::int32_t>{0, -1}, bad.mutable_int32_data().begin());
      inputs[0] = upload(bad); invalid();
    }
    if (kind == OpKind::kAffineRmsNorm || kind == OpKind::kAffineLayerNorm) {
      e.op.epsilon = -1; invalid(); e.op.epsilon = 1e-5;
      e.op.has_weight = false; invalid();
    }
    if (kind == OpKind::kAttention || kind == OpKind::kAttentionSoftmax) {
      e.op.attention_scale = std::numeric_limits<double>::infinity(); invalid();
    }
  }
}
