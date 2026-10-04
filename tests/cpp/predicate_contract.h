#pragma once

#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "tensorcx/backends/cpu/cpu_backend.h"

// Exercise the same native contract with CPU storage and device-owned storage.
// The callbacks only transfer test inputs/results; execution stays in Backend.
template <typename Upload, typename Download>
void predicate_contract(tensorcx::Backend& backend, Upload upload, Download download) {
  using namespace tensorcx;
  const auto check = [](bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
  };
  cpu::CpuBackend reference;
  for (auto dtype : {DType::kFloat32, DType::kInt32, DType::kBool}) {
    for (const Shape& shape : {Shape{}, Shape{257, 2}, Shape{257, 0}, Shape{0, 2}}) {
      cpu::CpuTensor data(dtype, shape), other(dtype, shape);
      for (Dim i = 0; i < data.size(); ++i) {
        if (dtype == DType::kFloat32) {
          data.mutable_float_data()[i] = static_cast<float>(i - 200);
          other.mutable_float_data()[i] = static_cast<float>(100 - i);
        } else if (dtype == DType::kInt32) {
          data.mutable_int32_data()[i] = static_cast<std::int32_t>(i - 200);
          other.mutable_int32_data()[i] = static_cast<std::int32_t>(100 - i);
        } else {
          data.mutable_bool_data()[i] = i % 3 == 0 ? 255 : 0;
          other.mutable_bool_data()[i] = i % 5 == 0;
        }
      }
      const Shape mask_shape = shape.empty() ? Shape{} : Shape{shape[0]};
      cpu::CpuTensor mask(DType::kBool, mask_shape);
      for (Dim i = 0; i < mask.size(); ++i) mask.mutable_bool_data()[i] = i % 3 == 0;
      cpu::CpuTensor condition(DType::kBool, {});
      condition.mutable_bool_data()[0] = 1;
      const auto a = upload(data), b = upload(other), m = upload(mask), c = upload(condition);
      for (auto kind : {OpKind::kEqual, OpKind::kNotEqual, OpKind::kLess, OpKind::kLessEqual,
                        OpKind::kGreater, OpKind::kGreaterEqual, OpKind::kWhere,
                        OpKind::kMaskedSelect, OpKind::kLogicalAnd, OpKind::kLogicalOr,
                        OpKind::kLogicalXor, OpKind::kLogicalNot, OpKind::kAny, OpKind::kAll, OpKind::kCast}) {
        if (dtype != DType::kBool && (kind == OpKind::kLogicalAnd || kind == OpKind::kLogicalOr ||
            kind == OpKind::kLogicalXor || kind == OpKind::kLogicalNot ||
            kind == OpKind::kAny || kind == OpKind::kAll)) continue;
        std::vector<Tensor> inputs{a, b}, host_inputs{cpu::to_core_tensor(data), cpu::to_core_tensor(other)};
        if (kind == OpKind::kWhere) {
          inputs = {c, a, b}; host_inputs = {cpu::to_core_tensor(condition), host_inputs[0], host_inputs[1]};
        } else if (kind == OpKind::kMaskedSelect) {
          inputs[1] = m; host_inputs[1] = cpu::to_core_tensor(mask);
        } else if (kind == OpKind::kCast || kind == OpKind::kLogicalNot || kind == OpKind::kAny || kind == OpKind::kAll) {
          inputs.resize(1); host_inputs.resize(1);
        }
        OpDesc op{kind}; op.target_dtype = dtype;
        if (kind == OpKind::kAny || kind == OpKind::kAll) {
          op.reduction_axes = Shape{};
          for (std::size_t axis = 0; axis < shape.size(); ++axis) op.reduction_axes->push_back(axis);
        }
        std::array<Tensor, 1> outputs{}, expected{};
        BackendExecution request{BackendOpClass::kPrimitive, op, inputs, outputs};
        BackendExecution host{BackendOpClass::kPrimitive, op, host_inputs, expected};
        check(reference.execute(host).ok(), "predicate CPU reference failed");
        check(backend.execute(request).ok(), "predicate backend failed");
        const auto actual = download(outputs[0]), wanted = cpu::from_core_tensor(expected[0]);
        check(actual.shape() == wanted.shape() && actual.dtype() == wanted.dtype(), "predicate metadata mismatch");
        const auto bytes = actual.buffer()->nbytes();
        check(bytes == wanted.buffer()->nbytes(), "predicate buffer size mismatch");
        check(!bytes || std::memcmp(actual.data(), wanted.data(), bytes) == 0, "predicate values mismatch");
        check(outputs[0].buffer != inputs[0].buffer, "predicate output aliases input");
        if (actual.dtype() == DType::kBool)
          check(bytes == static_cast<std::size_t>(actual.size()), "bool storage must be one byte");
        const auto previous = outputs[0].buffer;
        const auto original = inputs.back();
        for (int scenario = 0; scenario < 5; ++scenario) {
          inputs.back() = original;
          switch (scenario) {
            case 0: inputs.back().offset = 1; break;
            case 1: inputs.back().buffer.reset(); break;
            case 2: inputs.back().device.index = 1; break;
            case 3: inputs.back().dtype = static_cast<DType>(99); break;
            case 4: inputs.back().shape = {-1}; break;
          }
          check(backend.execute(request).code() == StatusCode::kInvalidArgument,
                "predicate accepted invalid input descriptor");
          check(outputs[0].buffer == previous, "predicate failure changed output");
        }
        inputs.back() = original;
        request.inputs = {};
        check(backend.execute(request).code() == StatusCode::kInvalidArgument, "predicate accepted wrong arity");
        check(outputs[0].buffer == previous, "predicate arity failure changed output");
      }
    }
  }
}
