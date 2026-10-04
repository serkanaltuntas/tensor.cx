#include <algorithm>
#include "predicate_contract.h"
#include "batched_matmul_contract.h"
#include "math_contract.h"
#include "inference_contract.h"

#include <array>
#include <bit>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <utility>
#include <vector>

#include "tensorcx/backends/cuda/cuda_backend.h"
#include "tensorcx/backends/cuda/cuda_buffer.h"
#include "tensorcx/backends/cpu/cpu_backend.h"

using namespace tensorcx;
namespace {
void require(bool result, const char* message) {
  if (!result) throw std::runtime_error(message);
}
void invalid(const Status& status) {
  require(status.code() == StatusCode::kInvalidArgument, status.message().c_str());
}
void same_float(float actual, float expected) {
  if (std::isnan(expected)) {
    require(std::isnan(actual), "arithmetic NaN mismatch");
  } else {
    require(actual == expected, "arithmetic value mismatch");
    if (expected == 0.0f || std::isinf(expected))
      require(std::signbit(actual) == std::signbit(expected), "arithmetic sign mismatch");
  }
}

void indexing_contract(cuda::CudaBackend& backend) {
  cpu::CpuBackend reference;
  for (auto dtype : {DType::kFloat32, DType::kInt32}) {
    for (const Shape& shape : {Shape{3, 257}, Shape{3, 0}}) {
      cpu::CpuTensor source(dtype, shape);
      for (Dim i = 0; i < source.size(); ++i) {
        if (dtype == DType::kFloat32) source.mutable_float_data()[i] = static_cast<float>(i - 300);
        else source.mutable_int32_data()[i] = static_cast<std::int32_t>(i - 300);
      }
      const auto uploaded = cuda::from_cpu(source).value();
      std::array<Tensor, 2> gpu_inputs{cuda::to_core_tensor(uploaded), cuda::to_core_tensor(uploaded)};
      std::array<Tensor, 2> cpu_inputs{cpu::to_core_tensor(source), cpu::to_core_tensor(source)};
      std::array<Tensor, 1> gpu_outputs{}, cpu_outputs{};
      for (const auto kind : {OpKind::kSlice, OpKind::kConcat}) {
        OpDesc op{kind}; op.axis = -1;
        op.slice_starts = {2, shape[1] ? 256 : 0}; op.slice_steps = {-1, -2};
        op.slice_shape = {3, shape[1] ? 129 : 0};
        const std::size_t arity = kind == OpKind::kSlice ? 1 : 2;
        BackendExecution gpu{BackendOpClass::kPrimitive, op, std::span(gpu_inputs).first(arity), gpu_outputs};
        BackendExecution cpu{BackendOpClass::kPrimitive, op, std::span(cpu_inputs).first(arity), cpu_outputs};
        require(reference.execute(cpu).ok(), "CPU indexing reference failed");
        require(backend.execute(gpu).ok(), "CUDA indexing failed");
        const auto actual = cuda::to_cpu(cuda::from_core_tensor(gpu_outputs[0]).value()).value();
        const auto expected = cpu::from_core_tensor(cpu_outputs[0]);
        require(actual.shape() == expected.shape() && actual.dtype() == dtype,
                "indexing metadata mismatch");
        require(gpu_outputs[0].buffer != gpu_inputs[0].buffer, "indexing must copy");
        if (dtype == DType::kFloat32) require(std::ranges::equal(actual.float_data(), expected.float_data()), "slice/concat mismatch");
        else require(std::ranges::equal(actual.int32_data(), expected.int32_data()), "int32 slice/concat mismatch");
        const auto previous = gpu_outputs[0].buffer;
        gpu.inputs = {};
        invalid(backend.execute(gpu));
        require(gpu_outputs[0].buffer == previous, "indexing arity failure changed output");
        gpu.inputs = std::span(gpu_inputs).first(arity);
        gpu_inputs[arity - 1].offset = 1;
        invalid(backend.execute(gpu));
        require(gpu_outputs[0].buffer == previous, "indexing metadata failure changed output");
        gpu_inputs[arity - 1].offset = 0;
        if (kind == OpKind::kSlice) gpu.op.slice_steps[0] = INT64_MIN;
        else gpu.op.axis = 2;
        invalid(backend.execute(gpu));
        require(gpu_outputs[0].buffer == previous, "indexing bounds failure changed output");
      }
    }
  }
}

void arithmetic_contract(cuda::CudaBackend& backend) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float tiny = std::numeric_limits<float>::denorm_min();
  const std::array<float, 11> left{0.0f, -0.0f, 1.0f, -1.0f, inf, -inf, nan, tiny, -tiny, 7.0f, -7.0f};
  const std::array<float, 11> right{-0.0f, 0.0f, 0.0f, -0.0f, inf, -inf, 2.0f, 2.0f, -2.0f, 2.0f, 3.0f};
  cpu::CpuBackend reference;
  // Rank zero, empty, and a partial second CUDA block. Include IEEE edge
  // values as well as asymmetric finite operands to detect operand reversal.
  for (const Shape& shape : {Shape{}, Shape{2, 0}, Shape{257}}) {
    const auto count = static_cast<std::size_t>(numel(shape));
    std::vector<float> lhs(count), rhs(count);
    for (std::size_t i = 0; i < count; ++i) {
      lhs[i] = left[i % left.size()];
      rhs[i] = right[i % right.size()];
    }
    cpu::CpuTensor a(shape, lhs), b(shape, rhs);
    auto ga = cuda::from_cpu(a).value(), gb = cuda::from_cpu(b).value();
    std::array<Tensor, 2> gpu_inputs{cuda::to_core_tensor(ga), cuda::to_core_tensor(gb)};
    std::array<Tensor, 2> cpu_inputs{cpu::to_core_tensor(a), cpu::to_core_tensor(b)};
    std::array<Tensor, 1> gpu_outputs{}, cpu_outputs{};
    auto verify = [&](OpDesc op, std::size_t arity) {
      BackendExecution gpu{BackendOpClass::kPrimitive, op, std::span(gpu_inputs).first(arity), gpu_outputs};
      BackendExecution cpu{BackendOpClass::kPrimitive, op, std::span(cpu_inputs).first(arity), cpu_outputs};
      require(reference.execute(cpu).ok(), "CPU arithmetic reference failed");
      require(backend.execute(gpu).ok(), "CUDA arithmetic failed");
      const auto actual = cuda::to_cpu(cuda::from_core_tensor(gpu_outputs[0]).value()).value();
      const auto expected = cpu::from_core_tensor(cpu_outputs[0]);
      require(actual.shape() == shape && actual.dtype() == DType::kFloat32, "arithmetic metadata mismatch");
      for (std::size_t i = 0; i < count; ++i) same_float(actual.float_data()[i], expected.float_data()[i]);
      const auto previous = gpu_outputs[0].buffer;
      gpu.inputs = {};
      invalid(backend.execute(gpu));
      require(gpu_outputs[0].buffer == previous, "arithmetic arity failure changed output");
      gpu.inputs = std::span(gpu_inputs).first(arity);
      gpu_inputs[0].offset = 1;
      invalid(backend.execute(gpu));
      require(gpu_outputs[0].buffer == previous, "arithmetic metadata failure changed output");
      gpu_inputs[0].offset = 0;
    };
    for (const auto kind : {OpKind::kAdd, OpKind::kSubtract, OpKind::kMultiply, OpKind::kDivide})
      verify(OpDesc{kind}, 2);
    verify(OpDesc{OpKind::kNegate}, 1);
    for (const auto kind : {OpKind::kAddScalar, OpKind::kSubtractScalar,
                           OpKind::kMultiplyScalar, OpKind::kDivideScalar}) {
      for (const double scalar : {-2.5, -0.0, static_cast<double>(inf), static_cast<double>(nan), 16777217.0}) {
        for (const bool scalar_left : {false, true}) {
          OpDesc op{kind}; op.scalar_value = scalar; op.scalar_left = scalar_left;
          verify(op, 1);
        }
      }
    }
    const auto original_a = cuda::to_cpu(ga).value();
    for (std::size_t i = 0; i < count; ++i) same_float(original_a.float_data()[i], lhs[i]);
  }
  // Genuine valid-but-incompatible tensor metadata, rather than only forged
  // descriptors, must reject before launching and leave the old result intact.
  auto f32 = cuda::from_cpu(cpu::CpuTensor({3}, std::vector<float>{1, 2, 3})).value();
  auto smaller = cuda::from_cpu(cpu::CpuTensor({2}, std::vector<float>{1, 2})).value();
  auto i32 = cuda::from_cpu(cpu::CpuTensor(DType::kInt32, {3})).value();
  std::array<Tensor, 2> inputs{cuda::to_core_tensor(f32), cuda::to_core_tensor(smaller)};
  std::array<Tensor, 1> outputs{cuda::to_core_tensor(f32)};
  const auto previous = outputs[0].buffer;
  for (const auto kind : {OpKind::kSubtract, OpKind::kDivide}) {
    BackendExecution execution{BackendOpClass::kPrimitive, OpDesc{kind}, inputs, outputs};
    invalid(backend.execute(execution));
    inputs[1] = cuda::to_core_tensor(i32);
    invalid(backend.execute(execution));
    require(outputs[0].buffer == previous, "binary mismatch changed output");
    inputs[1] = cuda::to_core_tensor(smaller);
  }
  inputs[0] = cuda::to_core_tensor(i32);
  for (const auto kind : {OpKind::kNegate, OpKind::kAddScalar, OpKind::kSubtractScalar,
                         OpKind::kMultiplyScalar, OpKind::kDivideScalar}) {
    BackendExecution execution{BackendOpClass::kPrimitive, OpDesc{kind}, std::span(inputs).first(1), outputs};
    invalid(backend.execute(execution));
    require(outputs[0].buffer == previous, "unary dtype failure changed output");
  }
}

void cast_contract(cuda::CudaBackend& backend) {
  const auto verify = [&](const cpu::CpuTensor& source, DType target) {
    auto device = cuda::from_cpu(source).value();
    std::array<Tensor, 1> inputs{cuda::to_core_tensor(device)}, outputs{};
    OpDesc op{OpKind::kCast};
    op.target_dtype = target;
    BackendExecution execution{BackendOpClass::kPrimitive, op, inputs, outputs};
    require(backend.execute(execution).ok(), "CUDA cast failed");
    const auto actual = cuda::to_cpu(cuda::from_core_tensor(outputs[0]).value()).value();
    require(actual.dtype() == target && actual.shape() == source.shape(), "cast metadata mismatch");
    require(outputs[0].buffer != inputs[0].buffer, "cast must create independent storage");
    if (target == DType::kFloat32) {
      for (std::size_t i = 0; i < actual.size(); ++i) {
        const float expected = source.dtype() == target ? source.float_data()[i]
            : static_cast<float>(source.int32_data()[i]);
        same_float(actual.float_data()[i], expected);
      }
    } else {
      for (std::size_t i = 0; i < actual.size(); ++i) {
        const auto expected = source.dtype() == target ? source.int32_data()[i]
            : static_cast<std::int32_t>(source.float_data()[i]);
        require(actual.int32_data()[i] == expected, "cast truncation mismatch");
      }
    }
    const auto previous = outputs[0].buffer;
    execution.op.target_dtype = static_cast<DType>(123);
    invalid(backend.execute(execution));
    require(outputs[0].buffer == previous, "invalid cast target changed output");
    execution.op.target_dtype = target;
    inputs[0].offset = 1;
    invalid(backend.execute(execution));
    require(outputs[0].buffer == previous, "malformed cast changed output");
  };
  const std::vector<float> floats{0.0F, -0.0F, 1.9F, -1.9F, 2147483520.0F, -2147483648.0F};
  const std::vector<std::int32_t> integers{0, -1, 16777217, -16777217, INT32_MIN, INT32_MAX};
  for (const Shape& shape : {Shape{}, Shape{2, 0}, Shape{257}}) {
    const auto count = static_cast<std::size_t>(numel(shape));
    std::vector<float> fvalues(count);
    std::vector<std::int32_t> ivalues(count);
    for (std::size_t i = 0; i < count; ++i) {
      fvalues[i] = floats[i % floats.size()];
      ivalues[i] = integers[i % integers.size()];
    }
    for (const auto target : {DType::kFloat32, DType::kInt32}) {
      verify(cpu::CpuTensor(shape, fvalues), target);
      verify(cpu::CpuTensor(shape, ivalues), target);
    }
  }
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  verify(cpu::CpuTensor({4}, std::vector<float>{inf, -inf, nan, -0.0F}), DType::kFloat32);
  // Put bad values past the first CUDA block, testing both conversion bounds
  // and the atomic validation status without ever publishing a partial output.
  for (const float bad : {inf, -inf, nan, 2147483648.0F, std::nextafter(-2147483648.0F, -inf)}) {
    std::vector<float> values(257, 1.5F); values.back() = bad;
    auto source = cuda::from_cpu(cpu::CpuTensor({257}, values)).value();
    auto sentinel = cuda::from_cpu(cpu::CpuTensor({1}, std::vector<std::int32_t>{42})).value();
    std::array<Tensor, 1> inputs{cuda::to_core_tensor(source)}, outputs{cuda::to_core_tensor(sentinel)};
    OpDesc op{OpKind::kCast}; op.target_dtype = DType::kInt32;
    BackendExecution execution{BackendOpClass::kPrimitive, op, inputs, outputs};
    invalid(backend.execute(execution));
    require(outputs[0].buffer == cuda::to_core_tensor(sentinel).buffer, "invalid values changed cast output");
    require(cuda::to_cpu(sentinel).value().int32_data()[0] == 42, "cast failure modified sentinel data");
  }
}

void broadcast_contract(cuda::CudaBackend& backend) {
  Shape high_rank(40, 1); high_rank[0] = 2; high_rank.back() = 3;
  const std::vector<std::pair<Shape, Shape>> pairs{
      {{2, 3}, {3}}, {{3}, {2, 3}}, {{2, 1, 3}, {1, 4, 1}},
      {{}, {257}}, {{257}, {}}, {{2, 0, 3}, {1, 3}}, {{0, 3}, {1, 3}},
      {high_rank, {3}}, {{0, 1, Dim{1} << 62, 4, 0}, {1}}};
  cpu::CpuBackend reference;
  for (const auto& [lhs_shape, rhs_shape] : pairs) {
    std::vector<float> lhs(static_cast<std::size_t>(numel(lhs_shape)));
    std::vector<float> rhs(static_cast<std::size_t>(numel(rhs_shape)));
    for (std::size_t i = 0; i < lhs.size(); ++i) lhs[i] = static_cast<float>(i % 7) - 3.5F;
    for (std::size_t i = 0; i < rhs.size(); ++i) rhs[i] = static_cast<float>(i % 3) + 0.5F;
    cpu::CpuTensor a(lhs_shape, lhs), b(rhs_shape, rhs);
    auto ga = cuda::from_cpu(a).value(), gb = cuda::from_cpu(b).value();
    std::array<Tensor, 2> gpu_inputs{cuda::to_core_tensor(ga), cuda::to_core_tensor(gb)};
    std::array<Tensor, 2> cpu_inputs{cpu::to_core_tensor(a), cpu::to_core_tensor(b)};
    std::array<Tensor, 1> gpu_outputs{}, cpu_outputs{};
    for (const auto kind : {OpKind::kAdd, OpKind::kSubtract, OpKind::kMultiply, OpKind::kDivide}) {
      BackendExecution gpu{BackendOpClass::kPrimitive, OpDesc{kind}, gpu_inputs, gpu_outputs};
      BackendExecution cpu{BackendOpClass::kPrimitive, OpDesc{kind}, cpu_inputs, cpu_outputs};
      require(reference.execute(cpu).ok(), "broadcast CPU reference failed");
      require(backend.execute(gpu).ok(), "CUDA broadcast failed");
      const auto actual = cuda::to_cpu(cuda::from_core_tensor(gpu_outputs[0]).value()).value();
      const auto expected = cpu::from_core_tensor(cpu_outputs[0]);
      require(actual.shape() == expected.shape(), "broadcast output shape mismatch");
      for (std::size_t i = 0; i < actual.size(); ++i)
        same_float(actual.float_data()[i], expected.float_data()[i]);
    }
    require(std::ranges::equal(cuda::to_cpu(ga).value().float_data(), lhs), "broadcast changed lhs input");
    require(std::ranges::equal(cuda::to_cpu(gb).value().float_data(), rhs), "broadcast changed rhs input");
  }
  auto left = cuda::from_cpu(cpu::CpuTensor({2, 0}, std::vector<float>{})).value();
  auto right = cuda::from_cpu(cpu::CpuTensor({3, 0}, std::vector<float>{})).value();
  std::array<Tensor, 2> inputs{cuda::to_core_tensor(left), cuda::to_core_tensor(right)};
  std::array<Tensor, 1> outputs{inputs[0]};
  const auto previous = outputs[0].buffer;
  for (const auto kind : {OpKind::kAdd, OpKind::kSubtract, OpKind::kMultiply, OpKind::kDivide}) {
    BackendExecution execution{BackendOpClass::kPrimitive, OpDesc{kind}, inputs, outputs};
    invalid(backend.execute(execution));
    require(outputs[0].buffer == previous, "empty incompatible broadcast changed output");
  }
}

void transpose_contract(cuda::CudaBackend& backend) {
  for (const auto dtype : {DType::kFloat32, DType::kInt32}) {
    for (const Shape shape : {Shape{}, Shape{2, 0}, Shape{3, 257}, Shape{2, 1, 3}}) {
      cpu::CpuTensor source(dtype, shape);
      for (std::size_t i = 0; i < static_cast<std::size_t>(source.size()); ++i) {
        if (dtype == DType::kFloat32) {
          const std::array<std::uint32_t, 6> bits{0, 0x80000000, 0x7fc12345, 1, 0xff800000, 0x7f812345};
          source.mutable_float_data()[i] = std::bit_cast<float>(bits[i % bits.size()]);
        } else source.mutable_int32_data()[i] = static_cast<std::int32_t>(i) - 17;
      }
      auto uploaded = cuda::from_cpu(source);
      require(bool(uploaded), "transpose input upload failed");
      std::array<Tensor, 1> inputs{cuda::to_core_tensor(uploaded.value())}, outputs{};
      OpDesc op{OpKind::kTranspose};
      for (std::size_t axis = shape.size(); axis > 0; --axis)
        op.axes.push_back(static_cast<Dim>(axis - 1));
      BackendExecution execution{BackendOpClass::kPrimitive, op, inputs, outputs};
      require(backend.execute(execution).ok(), "CUDA transpose failed");
      const auto expected = cpu::execute_unary(op, source);
      const auto actual = cuda::to_cpu(cuda::from_core_tensor(outputs[0]).value()).value();
      require(actual.shape() == expected.shape() && actual.dtype() == dtype &&
              outputs[0].buffer != inputs[0].buffer, "CUDA transpose metadata/ownership mismatch");
      for (std::size_t i = 0; i < static_cast<std::size_t>(source.size()); ++i) {
        require(dtype == DType::kInt32 ? actual.int32_data()[i] == expected.int32_data()[i] :
          std::bit_cast<std::uint32_t>(actual.float_data()[i]) ==
            std::bit_cast<std::uint32_t>(expected.float_data()[i]), "CUDA transpose bit mismatch");
      }
      const auto previous = outputs[0].buffer;
      execution.op.axes.push_back(0);
      invalid(backend.execute(execution));
      require(outputs[0].buffer == previous, "invalid transpose replaced output slot");
      execution.inputs = {};
      invalid(backend.execute(execution));
    }
  }
}

void multi_axis_contract(cuda::CudaBackend& backend) {
  Shape high_rank(300, 1); high_rank.front() = 2; high_rank.back() = 3;
  for (const auto& [shape, axes] : std::vector<std::pair<Shape, Shape>>{
         {{2, 3, 4}, {2, 0}}, {{257, 2, 3}, {1, 2}}, {{2, 0, 3}, {1, 2}},
         {{0, 2, 3}, {1, 2}}, {high_rank, {0, 299}}, {{}, {}}, {{2, 3}, {}}}) {
    cpu::CpuTensor source(DType::kFloat32, shape);
    for (std::size_t i = 0; i < static_cast<std::size_t>(source.size()); ++i)
      source.mutable_float_data()[i] = static_cast<float>(i % 13) - 3.5F;
    auto uploaded = cuda::from_cpu(source);
    require(bool(uploaded), "multi-axis input upload failed");
    for (const auto kind : {OpKind::kSum, OpKind::kMax, OpKind::kMean}) {
      OpDesc op{kind}; op.reduction_axes = axes;
      std::array<Tensor, 1> inputs{cuda::to_core_tensor(uploaded.value())}, outputs{inputs[0]};
      const auto previous = outputs[0].buffer;
      BackendExecution execution{BackendOpClass::kPrimitive, op, inputs, outputs};
      if (kind == OpKind::kMax && make_reduction_plan(shape, axes).reduction_size == 0) {
        invalid(backend.execute(execution));
        require(outputs[0].buffer == previous, "empty max changed supplied output");
        continue;
      }
      require(backend.execute(execution).ok(), "CUDA multi-axis reduction failed");
      const auto actual = cuda::to_cpu(cuda::from_core_tensor(outputs[0]).value()).value();
      const auto expected = cpu::reduce(op, source);
      require(actual.shape() == expected.shape() && outputs[0].buffer != inputs[0].buffer,
              "CUDA multi-axis shape/ownership mismatch");
      for (std::size_t i = 0; i < actual.float_data().size(); ++i)
        same_float(actual.float_data()[i], expected.float_data()[i]);
      const auto output = outputs[0].buffer;
      execution.op.reduction_axes = Shape{0, 0};
      invalid(backend.execute(execution));
      require(outputs[0].buffer == output, "invalid multi-axis request changed output");
    }
    require(std::ranges::equal(cuda::to_cpu(uploaded.value()).value().float_data(), source.float_data()),
            "multi-axis reduction changed input");
  }
}

Tensor descriptor(Shape shape = {3}) {
  const auto strides = contiguous_strides(shape);
  return {DType::kFloat32, std::move(shape), strides, {"cuda", 0}, nullptr, 0};
}
}  // namespace

int main() {
  try {
    cuda::CudaBackend backend;
    std::array<Tensor, 1> outputs{descriptor()};
    BackendExecution fill{BackendOpClass::kPrimitive, OpDesc{OpKind::kFill}, {}, outputs};
    fill.op.scalar_value = 2.5;
    auto malformed = fill;
    malformed.outputs = {};
    invalid(backend.execute(malformed));
    malformed = fill;
    malformed.launch = LaunchConfig{1, 1, 1, 1, 1, 1};
    invalid(backend.execute(malformed));
    for (int scenario = 0; scenario < 6; ++scenario) {
      outputs[0] = descriptor();
      switch (scenario) {
        case 0: outputs[0].device.type = "cpu"; break;
        case 1: outputs[0].device.index = 1; break;
        case 2: outputs[0].offset = 1; break;
        case 3: outputs[0].strides = {2}; break;
        case 4: outputs[0].dtype = DType::kInt32; break;
        case 5: outputs[0].buffer = cpu::CpuTensor(DType::kFloat32, {3}).buffer(); break;
      }
      invalid(backend.execute(fill));
    }
    malformed = fill;
    malformed.op_class = BackendOpClass::kKernel;
    invalid(backend.execute(malformed));
    outputs[0] = descriptor();
    if (!cuda::available()) {
      if (std::getenv("TENSORCX_REQUIRE_CUDA")) {
        throw std::runtime_error("required CUDA device unavailable");
      }
      std::cout << "CUDA device unavailable; metadata validation passed\n";
      return 77;
    }
    {
      auto owned = cuda::CudaBuffer::create(DType::kFloat32, {4}).value();
      std::weak_ptr<cuda::CudaBuffer> lifetime = owned;
      auto a = cuda::CudaBuffer::borrow(DType::kFloat32, {3}, owned->data(), owned).value();
      auto b = cuda::CudaBuffer::borrow(DType::kFloat32, {3}, owned->data(), owned).value();
      auto partial = cuda::CudaBuffer::borrow(DType::kFloat32, {3}, static_cast<float*>(owned->data()) + 1, owned).value();
      require(a->storage_relation(*b) == StorageRelation::kSameRange, "borrowed CUDA exact alias");
      require(a->storage_relation(*partial) == StorageRelation::kPartialOverlap, "borrowed CUDA partial alias");
      auto empty = cuda::CudaBuffer::borrow(DType::kFloat32, {0}, nullptr, owned).value();
      auto other_empty = cuda::CudaBuffer::borrow(DType::kFloat32, {0}, nullptr, owned).value();
      require(empty->storage_relation(*other_empty) == StorageRelation::kDisjoint, "empty CUDA ranges must not alias");
      float host_data[3]{};
      invalid(cuda::CudaBuffer::borrow(DType::kFloat32, {3}, host_data, owned).status());
      invalid(cuda::CudaBuffer::borrow(DType::kFloat32, {3}, owned->data(), {}).status());
      owned.reset(); a.reset(); partial.reset(); empty.reset(); other_empty.reset();
      require(!lifetime.expired(), "borrowed CUDA owner lifetime");
      b.reset();
      require(lifetime.expired(), "borrowed CUDA owner released");
    }
    require(backend.execute(fill).ok(), "fill failed");
    auto filled = cuda::from_core_tensor(outputs[0]);
    require(bool(filled), "fill metadata invalid");
    auto host = cuda::to_cpu(filled.value());
    require(bool(host) && std::ranges::equal(host.value().float_data(), std::vector<float>({2.5, 2.5, 2.5})),
            "fill value mismatch");
    const Tensor original = outputs[0];
    std::array<Tensor, 2> inputs{original, original};
    BackendExecution binary{BackendOpClass::kPrimitive, OpDesc{OpKind::kAdd}, inputs, outputs};
    for (const auto kind : {OpKind::kAdd, OpKind::kMultiply}) {
      binary.op.kind = kind;
      require(backend.execute(binary).ok(), "binary failed");
      auto result = cuda::to_cpu(cuda::from_core_tensor(outputs[0]).value());
      const float expected = kind == OpKind::kAdd ? 5.0f : 6.25f;
      require(bool(result) && std::ranges::equal(result.value().float_data(), std::vector<float>(3, expected)),
              "binary value mismatch");
    }
    {
      cpu::CpuTensor source({2,2},std::vector<float>{1,2,3,4});
      auto device=cuda::from_cpu(source);
      require(bool(device),"primitive input copy failed");
      std::array<Tensor,2> gpu_inputs{cuda::to_core_tensor(device.value()),cuda::to_core_tensor(device.value())};
      std::array<Tensor,2> cpu_inputs{cpu::to_core_tensor(source),cpu::to_core_tensor(source)};
      std::array<Tensor,1> gpu_outputs{},cpu_outputs{};
      cpu::CpuBackend reference;
      for(const auto kind:{OpKind::kMatmul,OpKind::kSum,OpKind::kMax,OpKind::kMean,
          OpKind::kExp,OpKind::kGelu,OpKind::kSilu,OpKind::kSoftmax,OpKind::kRmsNorm,OpKind::kLayerNorm}) {
        OpDesc op{kind};op.axis=1;
        const std::size_t count=kind==OpKind::kMatmul?2:1;
        BackendExecution gpu{BackendOpClass::kPrimitive,op,std::span(gpu_inputs).first(count),gpu_outputs};
        BackendExecution cpu{BackendOpClass::kPrimitive,op,std::span(cpu_inputs).first(count),cpu_outputs};
        require(reference.execute(cpu).ok(),"CPU primitive failed");
        require(backend.execute(gpu).ok(),"CUDA primitive failed");
        auto actual=cuda::to_cpu(cuda::from_core_tensor(gpu_outputs[0]).value()).value();
        auto expected=cpu::from_core_tensor(cpu_outputs[0]);
        require(actual.shape()==expected.shape(),"primitive shape mismatch");
        for(std::size_t i=0;i<actual.float_data().size();++i)
          require(std::abs(actual.float_data()[i]-expected.float_data()[i])<1e-4F,"primitive value mismatch");
        auto previous=gpu_outputs[0].buffer;
        gpu_inputs[0].offset=1;
        invalid(backend.execute(gpu));
        require(gpu_outputs[0].buffer==previous,"primitive failure published output");
        gpu_inputs[0].offset=0;
        if(kind==OpKind::kRmsNorm || kind==OpKind::kLayerNorm) {
          gpu.op.epsilon=-1;
          invalid(backend.execute(gpu));
          require(gpu_outputs[0].buffer==previous,"epsilon failure published output");
        }
      }
    }
    indexing_contract(backend);
    predicate_contract(backend,
      [](const cpu::CpuTensor& value) { return cuda::to_core_tensor(cuda::from_cpu(value).value()); },
      [](const Tensor& value) { return cuda::to_cpu(cuda::from_core_tensor(value).value()).value(); });
    batched_matmul_contract(backend,
      [](const cpu::CpuTensor& value) { return cuda::to_core_tensor(cuda::from_cpu(value).value()); },
      [](const Tensor& value) { return cuda::to_cpu(cuda::from_core_tensor(value).value()).value(); });
    math_contract(backend,
      [](const cpu::CpuTensor& value) { return cuda::to_core_tensor(cuda::from_cpu(value).value()); },
      [](const Tensor& value) { return cuda::to_cpu(cuda::from_core_tensor(value).value()).value(); });
    inference_contract(backend,
      [](const cpu::CpuTensor& value) { return cuda::to_core_tensor(cuda::from_cpu(value).value()); },
      [](const Tensor& value) { return cuda::to_cpu(cuda::from_core_tensor(value).value()).value(); });
    arithmetic_contract(backend);
    cast_contract(backend);
    broadcast_contract(backend);
    transpose_contract(backend);
    multi_axis_contract(backend);
    // Long contiguous rows exercise the cooperative path under device
    // memcheck/racecheck as well as ordinary native acceptance.
    for (const std::int64_t width : {256, 257, 4097}) {
      std::vector<float> values(3 * width);
      for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = i < static_cast<std::size_t>(2 * width) ? static_cast<float>(i % 17) / 17 : 0.3F;
      cpu::CpuTensor source({3, width}, values);
      auto device = cuda::from_cpu(source).value();
      std::array<Tensor, 1> gpu_inputs{cuda::to_core_tensor(device)}, cpu_inputs{cpu::to_core_tensor(source)};
      std::array<Tensor, 1> gpu_outputs{}, cpu_outputs{};
      cpu::CpuBackend reference;
      for (auto kind : {OpKind::kSum, OpKind::kMax, OpKind::kMean,
                        OpKind::kSoftmax, OpKind::kRmsNorm, OpKind::kLayerNorm}) {
        OpDesc op{kind}; op.axis = 1;
        BackendExecution gpu{BackendOpClass::kPrimitive, op, gpu_inputs, gpu_outputs};
        BackendExecution cpu{BackendOpClass::kPrimitive, op, cpu_inputs, cpu_outputs};
        require(reference.execute(cpu).ok(), "staged reference failed");
        require(backend.execute(gpu).ok(), "staged CUDA failed");
        auto actual = cuda::to_cpu(cuda::from_core_tensor(gpu_outputs[0]).value()).value();
        auto expected = cpu::from_core_tensor(cpu_outputs[0]);
        require(actual.shape() == expected.shape(), "staged shape mismatch");
        for (std::size_t i = 0; i < actual.float_data().size(); ++i)
          require(std::abs(actual.float_data()[i] - expected.float_data()[i]) < 1e-4F, "staged value mismatch");
      }
    }
    const Tensor previous_output = outputs[0];
    for (int scenario = 0; scenario < 9; ++scenario) {
      inputs[0] = original;
      switch (scenario) {
        case 0: inputs[0].device.type = "cpu"; break;
        case 1: inputs[0].device.index = 1; break;
        case 2: inputs[0].offset = 1; break;
        case 3: inputs[0].strides = {2}; break;
        case 4: inputs[0].buffer.reset(); break;
        case 5: inputs[0].buffer = cpu::CpuTensor(DType::kFloat32, {3}).buffer(); break;
        case 6: inputs[0].shape = {4}; break;
        case 7: inputs[0].dtype = DType::kInt32; break;
        case 8: inputs[0].shape = {-1}; break;
      }
      invalid(backend.execute(binary));
      require(outputs[0].buffer == previous_output.buffer, "failure changed output");
    }
    inputs[0] = original;
    auto unsupported = binary;
    unsupported.op.kind = OpKind::kMatmul;
    inputs[1].shape = {1, 3}; inputs[1].strides = {3, 1};
    invalid(backend.execute(unsupported));
    inputs[1] = original;
    // A kernel contract with an unknown artifact must still be rejected.
    KernelArgument arg{KernelArgumentKind::kTensor, &original, 0};
    BackendExecution kernel{BackendOpClass::kKernel, {}, {}, outputs,
      LaunchConfig{3, 1, 1, 32, 1, 1},
      CompilationTarget{KernelArtifactKind::kBinary, "unused", "entry"},
      std::span<const KernelArgument>(&arg, 1)};
    invalid(backend.execute(kernel));
    outputs[0] = descriptor({std::numeric_limits<std::int64_t>::max() / 2 + 1});
    invalid(backend.execute(fill));
    // Empty outputs retain ownership but perform no invalid zero-block launch.
    outputs[0] = descriptor({2, 0});
    require(backend.execute(fill).ok(), "empty fill failed");
    require(outputs[0].buffer->nbytes() == 0, "empty allocation size mismatch");
    std::cout << "CUDA backend contract passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CUDA backend contract failed: " << error.what() << '\n';
    return 1;
  }
}
