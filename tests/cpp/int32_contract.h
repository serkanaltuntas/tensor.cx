#pragma once

#include <algorithm>
#include <array>
#include <climits>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#include "tensorcx/backends/cpu/cpu_backend.h"

// Exercise the native ABI directly, including invalid OpDesc values that the
// Python scalar parser normally rejects, and publish-on-success ownership.
template<typename Upload, typename Download>
void int32_contract(tensorcx::Backend& backend, Upload upload, Download download) {
  using namespace tensorcx;
  auto check = [](bool ok, const char* message) { if (!ok) throw std::runtime_error(message); };
  const std::array<std::int32_t, 8> edges{INT32_MIN, INT32_MAX, -16777217, 16777217, -1, 0, 1, 65537};
  cpu::CpuBackend reference;
  const auto device = upload(cpu::CpuTensor(DType::kInt32, {0})).device;
  for (const auto& [left_shape, right_shape] : std::vector<std::pair<Shape, Shape>>{
      {{}, {}}, {{2, 0}, {2, 0}}, {{257}, {257}}, {{2, 1, 257}, {3, 1}}}) {
    cpu::CpuTensor left(DType::kInt32, left_shape), right(DType::kInt32, right_shape);
    for (Dim i = 0; i < left.size(); ++i) left.mutable_int32_data()[i] = edges[i % edges.size()];
    for (Dim i = 0; i < right.size(); ++i) right.mutable_int32_data()[i] = edges[edges.size() - 1 - i % edges.size()];
    std::array<Tensor, 2> inputs{upload(left), upload(right)};
    std::array<Tensor, 2> cpu_inputs{cpu::to_core_tensor(left), cpu::to_core_tensor(right)};
    std::array<Tensor, 1> outputs{inputs[0]}, expected{};
    auto verify = [&](OpDesc op, std::size_t arity) {
      BackendExecution e{BackendOpClass::kPrimitive, op, std::span(inputs).first(arity), outputs};
      BackendExecution ce{BackendOpClass::kPrimitive, op, std::span(cpu_inputs).first(arity), expected};
      const auto status = reference.execute(ce);
      if (!status.ok()) {
        const auto previous = outputs[0].buffer;
        check(backend.execute(e).code() == status.code(), "int32 invalid reduction accepted");
        check(outputs[0].buffer == previous, "int32 failure changed result");
        return;
      }
      const auto actual_status = backend.execute(e);
      check(actual_status.ok(), actual_status.message().c_str());
      const auto actual = download(outputs[0]), wanted = cpu::from_core_tensor(expected[0]);
      check(actual.dtype() == DType::kInt32 && actual.shape() == wanted.shape(), "int32 result metadata");
      check(std::ranges::equal(actual.int32_data(), wanted.int32_data()), "int32 exact parity");
      check(outputs[0].buffer != inputs[0].buffer && outputs[0].buffer != inputs[1].buffer, "int32 result aliases input");
      const auto previous = outputs[0].buffer;
      inputs[0].offset = 1;
      check(backend.execute(e).code() == StatusCode::kInvalidArgument, "int32 bad input offset accepted");
      inputs[0].offset = 0;
      check(outputs[0].buffer == previous, "int32 metadata failure changed result");
    };
    for (const auto kind : {OpKind::kAdd, OpKind::kSubtract, OpKind::kMultiply}) verify(OpDesc{kind}, 2);
    verify(OpDesc{OpKind::kNegate}, 1);
    for (const auto kind : {OpKind::kAddScalar, OpKind::kSubtractScalar, OpKind::kMultiplyScalar}) {
      for (const double scalar : {double(INT32_MIN), double(INT32_MAX), 16777217.0}) {
        for (const bool reverse : {false, true}) {
          OpDesc op{kind}; op.scalar_value = scalar; op.scalar_left = reverse;
          verify(op, 1);
        }
      }
      for (const auto scalar : {0.5, 2147483648.0, -2147483649.0,
                               std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        OpDesc op{kind}; op.scalar_value = scalar;
        const auto previous = outputs[0].buffer;
        BackendExecution e{BackendOpClass::kPrimitive, op, std::span(inputs).first(1), outputs};
        check(backend.execute(e).code() == StatusCode::kInvalidArgument, "int32 invalid scalar accepted");
        check(outputs[0].buffer == previous, "int32 scalar error changed result");
      }
    }
    for (const auto kind : {OpKind::kSum, OpKind::kMax}) {
      OpDesc op{kind}; op.axis = -1; verify(op, 1);
      op.reduction_axes = Shape{}; verify(op, 1);
      if (left_shape.size() == 3) { op.reduction_axes = Shape{2, 0}; verify(op, 1); }
    }
    for (const auto kind : {OpKind::kDivideScalar, OpKind::kMean, OpKind::kExp}) {
      OpDesc op{kind}; op.reduction_axes = Shape{};
      BackendExecution e{BackendOpClass::kPrimitive, op, std::span(inputs).first(1), outputs};
      const auto previous = outputs[0].buffer;
      check(backend.execute(e).code() == StatusCode::kInvalidArgument, "int32 float-only operation accepted");
      check(outputs[0].buffer == previous, "int32 dtype error changed result");
    }
    check(std::ranges::equal(download(inputs[0]).int32_data(), left.int32_data()), "int32 input changed");
  }
  for (const Shape& shape : {Shape{0}, Shape{257}}) {
    for (const auto value : {double(INT32_MIN), double(INT32_MAX), 16777217.0, 0.0, 1.0, 0.5,
                            2147483648.0, -2147483649.0, std::numeric_limits<double>::quiet_NaN()}) {
      std::array<Tensor, 1> output{Tensor{DType::kInt32, shape, contiguous_strides(shape), device, nullptr, 0}};
      OpDesc op{OpKind::kFill}; op.scalar_value = value;
      BackendExecution e{BackendOpClass::kPrimitive, op, {}, output};
      if (!is_int32_representable(value)) {
        check(backend.execute(e).code() == StatusCode::kInvalidArgument, "invalid int32 fill accepted");
        check(!output[0].buffer && output[0].shape == shape, "invalid fill published result");
      } else {
        check(backend.execute(e).ok(), "int32 fill failed");
        const auto actual = download(output[0]);
        check(actual.dtype() == DType::kInt32 && actual.shape() == shape, "int32 fill metadata");
        check(std::ranges::all_of(actual.int32_data(), [&](auto item) { return item == value; }), "int32 fill value");
      }
    }
  }
}
