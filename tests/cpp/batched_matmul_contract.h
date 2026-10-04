#pragma once

#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tensorcx/backends/cpu/cpu_backend.h"

// Shared native coverage, independent of Python's constructor validation.
template <typename Upload, typename Download>
void batched_matmul_contract(tensorcx::Backend& backend, Upload upload, Download download,
                             tensorcx::MatmulPreference preference = tensorcx::MatmulPreference::kAuto) {
  using namespace tensorcx;
  const auto check = [](bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
  };
  const std::vector<std::pair<Shape, Shape>> cases{
      {{3}, {3}}, {{2, 3}, {3}}, {{3}, {2, 3, 4}},
      {{2, 1, 17, 19}, {1, 3, 19, 33}}, {{3, 4}, {2, 4, 5}},
      {{2, 3, 0}, {1, 0, 4}}, {{0, 3, 4}, {1, 4, 5}}, {{2, 0, 4}, {4, 3}}};
  for (const auto& [left, right] : cases) {
    cpu::CpuTensor a(DType::kFloat32, left), b(DType::kFloat32, right);
    for (Dim i = 0; i < a.size(); ++i) a.mutable_float_data()[i] = static_cast<float>(i % 11 - 5) / 8;
    for (Dim i = 0; i < b.size(); ++i) b.mutable_float_data()[i] = static_cast<float>(i % 7 - 3) / 8;
    std::array<Tensor, 2> inputs{upload(a), upload(b)};
    std::array<Tensor, 1> outputs{};
    OpDesc op{OpKind::kMatmul}; op.matmul_preference = preference;
    BackendExecution request{BackendOpClass::kPrimitive, op, inputs, outputs, std::nullopt, std::nullopt, {}};
    const auto status = backend.execute(request);
    check(status.ok(), status.message().c_str());
    const auto actual = download(outputs[0]), expected = cpu::matmul(a, b);
    check(actual.shape() == expected.shape() && actual.dtype() == DType::kFloat32, "batch matmul metadata mismatch");
    for (Dim i = 0; i < actual.size(); ++i)
      check(std::abs(actual.float_data()[i] - expected.float_data()[i]) < 1e-4F, "batch matmul CPU parity failed");
    check(outputs[0].buffer != inputs[0].buffer && outputs[0].buffer != inputs[1].buffer,
          "batch matmul result aliases input");
    const auto original = inputs[1], previous = outputs[0];
    for (int scenario = 0; scenario < 6; ++scenario) {
      inputs[1] = original;
      switch (scenario) {
        case 0: inputs[1].buffer.reset(); break;
        case 1: inputs[1].offset = 1; break;
        case 2: inputs[1].device.index = 1; break;
        case 3: inputs[1].dtype = DType::kBool; break;
        case 4: inputs[1].shape = {-1}; break;
        case 5: inputs[1].strides.clear(); break;
      }
      check(backend.execute(request).code() == StatusCode::kInvalidArgument,
            "batch matmul accepted malformed descriptor");
      check(outputs[0].buffer == previous.buffer && outputs[0].shape == previous.shape,
            "batch matmul failure replaced output");
    }
    inputs[1] = original;
    request.inputs = std::span<const Tensor>(inputs).first(1);
    check(backend.execute(request).code() == StatusCode::kInvalidArgument, "matmul accepted wrong arity");
    check(outputs[0].buffer == previous.buffer, "matmul arity failure replaced output");
  }
  // Zero-size storage makes these otherwise huge descriptors affordable.
  cpu::CpuTensor a(DType::kFloat32, {0, 2, 3, 4}), b(DType::kFloat32, {0, 5, 4, 3});
  std::array<Tensor, 2> inputs{upload(a), upload(b)};
  std::array<Tensor, 1> outputs{inputs[0]};
  OpDesc op{OpKind::kMatmul}; op.matmul_preference = preference;
  BackendExecution request{BackendOpClass::kPrimitive, op, inputs, outputs, std::nullopt, std::nullopt, {}};
  check(backend.execute(request).code() == StatusCode::kInvalidArgument,
        "empty matmul accepted incompatible batch dimensions");
  check(outputs[0].buffer == inputs[0].buffer, "batch mismatch replaced output");
}
