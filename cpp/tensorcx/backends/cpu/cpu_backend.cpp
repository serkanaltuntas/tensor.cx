#include "tensorcx/backends/cpu/cpu_backend.h"
#include "tensorcx/backends/cpu/cpu_kernel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include "tensorcx/core/dtype.h"
#include "tensorcx/core/predicate.h"
#include "tensorcx/core/math.h"
#include "tensorcx/core/inference.h"

namespace tensorcx::cpu {

namespace {

constexpr float kGeluTanhCoefficient = 0.7978845608028654F;
constexpr float kGeluCubicCoefficient = 0.044715F;

float checked_epsilon(double epsilon) {
  if (!std::isfinite(epsilon) || epsilon < 0.0 ||
      epsilon > static_cast<double>(std::numeric_limits<float>::max())) {
    throw std::invalid_argument(
        "epsilon must be finite and non-negative and representable as float32");
  }
  const float rounded = static_cast<float>(epsilon);
  if (epsilon > 0.0 && rounded == 0.0F) {
    throw std::invalid_argument(
        "epsilon must be finite and non-negative and representable as float32");
  }
  return rounded;
}

void validate_binary_inputs(const CpuTensor& lhs, const CpuTensor& rhs) {
  if (lhs.device().type != "cpu" || rhs.device().type != "cpu") {
    throw std::invalid_argument("CPU operations require CPU tensors");
  }
  if (lhs.dtype() != rhs.dtype()) {
    throw std::invalid_argument("dtype mismatch for binary operation");
  }
}

Status invalid_argument_status(const char* message) {
  return Status(StatusCode::kInvalidArgument, message);
}

Status expect_status_code(const char* scenario, const Status& status, StatusCode expected) {
  if (status.code() == expected) {
    return Status::Ok();
  }
  return Status(
      StatusCode::kInternal,
      std::string("CPU backend contract smoke test failed: ") + scenario +
          " expected status " + std::to_string(static_cast<int>(expected)) +
          " got status " + std::to_string(static_cast<int>(status.code())) +
          " message: " + status.message());
}

struct ReductionDims {
  Shape output_shape;
  std::int64_t outer;
  std::int64_t reduce;
  std::int64_t inner;
};

std::int64_t normalize_axis(std::int64_t axis, std::size_t rank) {
  if (rank == 0) {
    if (axis == 0 || axis == -1) {
      return 0;
    }
    throw std::invalid_argument("reduction axis is out of range");
  }
  const auto signed_rank = static_cast<std::int64_t>(rank);
  if (axis < 0) {
    axis += signed_rank;
  }
  if (axis < 0 || axis >= signed_rank) {
    throw std::invalid_argument("reduction axis is out of range");
  }
  return axis;
}

ReductionDims reduction_dims(const CpuTensor& input, std::int64_t axis) {
  if (input.device().type != "cpu") {
    throw std::invalid_argument("CPU reductions require CPU tensors");
  }

  const auto normalized_axis = normalize_axis(axis, input.shape().size());
  if (input.shape().empty()) {
    return ReductionDims{Shape{}, 1, 1, 1};
  }
  Shape output_shape;
  output_shape.reserve(input.shape().size() - 1);
  for (std::size_t index = 0; index < input.shape().size(); ++index) {
    if (index != static_cast<std::size_t>(normalized_axis)) {
      output_shape.push_back(input.shape()[index]);
    }
  }

  std::int64_t outer = 1;
  for (std::size_t index = 0; index < static_cast<std::size_t>(normalized_axis); ++index) {
    outer *= input.shape()[index];
  }
  // The contiguous stride is the already-validated suffix product. Computing
  // it left-to-right can overflow before reaching a trailing zero dimension.
  const auto inner = input.strides()[static_cast<std::size_t>(normalized_axis)];

  return ReductionDims{
      std::move(output_shape),
      outer,
      input.shape()[static_cast<std::size_t>(normalized_axis)],
      inner};
}

void compute_softmax(const CpuTensor& input, CpuTensor& result, const ReductionDims& dims) {
  const auto& input_data = input.float_data();
  auto& out = result.mutable_float_data();

  for (std::int64_t outer = 0; outer < dims.outer; ++outer) {
    for (std::int64_t inner = 0; inner < dims.inner; ++inner) {
      const auto base = outer * dims.reduce * dims.inner + inner;
      float max_value = -std::numeric_limits<float>::infinity();
      for (std::int64_t reduce_index = 0; reduce_index < dims.reduce; ++reduce_index) {
        const float value =
            input_data[static_cast<std::size_t>(base + reduce_index * dims.inner)];
        if (std::isnan(value)) {
          max_value = value;
          break;
        }
        max_value = std::max(max_value, value);
      }

      float denom = 0.0F;
      for (std::int64_t reduce_index = 0; reduce_index < dims.reduce; ++reduce_index) {
        denom += std::exp(
            input_data[static_cast<std::size_t>(base + reduce_index * dims.inner)] -
            max_value);
      }
      for (std::int64_t reduce_index = 0; reduce_index < dims.reduce; ++reduce_index) {
        const auto index = static_cast<std::size_t>(base + reduce_index * dims.inner);
        out[index] = std::exp(input_data[index] - max_value) / denom;
      }
    }
  }
}

void compute_rmsnorm(
    const CpuTensor& input,
    CpuTensor& result,
    const ReductionDims& dims,
    float epsilon) {
  const auto& input_data = input.float_data();
  auto& out = result.mutable_float_data();

  for (std::int64_t outer = 0; outer < dims.outer; ++outer) {
    for (std::int64_t inner = 0; inner < dims.inner; ++inner) {
      const auto base = outer * dims.reduce * dims.inner + inner;
      float sum_squares = 0.0F;
      for (std::int64_t reduce_index = 0; reduce_index < dims.reduce; ++reduce_index) {
        const float value =
            input_data[static_cast<std::size_t>(base + reduce_index * dims.inner)];
        sum_squares += value * value;
      }

      const float scale =
          1.0F / std::sqrt((sum_squares / static_cast<float>(dims.reduce)) + epsilon);
      for (std::int64_t reduce_index = 0; reduce_index < dims.reduce; ++reduce_index) {
        const auto index = static_cast<std::size_t>(base + reduce_index * dims.inner);
        out[index] = input_data[index] * scale;
      }
    }
  }
}

void compute_layernorm(
    const CpuTensor& input,
    CpuTensor& result,
    const ReductionDims& dims,
    float epsilon) {
  const auto& input_data = input.float_data();
  auto& out = result.mutable_float_data();

  for (std::int64_t outer = 0; outer < dims.outer; ++outer) {
    for (std::int64_t inner = 0; inner < dims.inner; ++inner) {
      const auto base = outer * dims.reduce * dims.inner + inner;
      float sum = 0.0F;
      const float first_value = input_data[static_cast<std::size_t>(base)];
      bool all_equal = true;
      for (std::int64_t reduce_index = 0; reduce_index < dims.reduce; ++reduce_index) {
        const float value =
            input_data[static_cast<std::size_t>(base + reduce_index * dims.inner)];
        sum += value;
        all_equal = all_equal && value == first_value;
      }
      if (all_equal && std::isfinite(first_value)) {
        const float value =
            epsilon == 0.0F ? std::numeric_limits<float>::quiet_NaN() : 0.0F;
        for (std::int64_t reduce_index = 0; reduce_index < dims.reduce; ++reduce_index) {
          const auto index = static_cast<std::size_t>(base + reduce_index * dims.inner);
          out[index] = value;
        }
        continue;
      }
      const float mean = sum / static_cast<float>(dims.reduce);

      float sum_squared_diff = 0.0F;
      for (std::int64_t reduce_index = 0; reduce_index < dims.reduce; ++reduce_index) {
        const float diff =
            input_data[static_cast<std::size_t>(base + reduce_index * dims.inner)] - mean;
        sum_squared_diff += diff * diff;
      }

      const float variance = sum_squared_diff / static_cast<float>(dims.reduce);
      const float denom = variance + epsilon;
      if (denom == 0.0F) {
        for (std::int64_t reduce_index = 0; reduce_index < dims.reduce; ++reduce_index) {
          const auto index = static_cast<std::size_t>(base + reduce_index * dims.inner);
          out[index] = std::numeric_limits<float>::quiet_NaN();
        }
        continue;
      }

      const float scale = 1.0F / std::sqrt(denom);
      for (std::int64_t reduce_index = 0; reduce_index < dims.reduce; ++reduce_index) {
        const auto index = static_cast<std::size_t>(base + reduce_index * dims.inner);
        out[index] = (input_data[index] - mean) * scale;
      }
    }
  }
}

}  // namespace

std::string CpuBackend::name() const { return "cpu"; }

Status CpuBackend::execute(const BackendExecution& execution) {
  if (execution.op_class == BackendOpClass::kKernel) return execute_compiled_kernel(execution);
  try {
    const Status contract = validate_primitive_execution_contract(execution, "cpu");
    if (!contract.ok()) {
      return contract;
    }

    if (is_inference_composite(execution.op.kind) || execution.op.kind == OpKind::kEmbedding || execution.op.kind == OpKind::kAttentionSoftmax) {
      for (const auto& input : execution.inputs) (void)from_core_tensor(input);
      if (is_inference_composite(execution.op.kind)) return execute_inference_composite(*this, execution);
      return execute_inference_primitive(execution);
    }
    if (is_math_operation(execution.op.kind)) return execute_math(execution);
    if (is_predicate_elementwise(execution.op.kind)) {
      std::vector<CpuTensor> inputs;
      for (const auto& input : execution.inputs) inputs.push_back(from_core_tensor(input));
      execution.outputs[0] = to_core_tensor(predicate(execution.op, inputs));
      return Status::Ok();
    }
    if (execution.op.kind == OpKind::kAny || execution.op.kind == OpKind::kAll) {
      execution.outputs[0] = to_core_tensor(reduce_boolean(execution.op, from_core_tensor(execution.inputs[0])));
      return Status::Ok();
    }
    if (execution.op.kind == OpKind::kMaskedSelect) {
      execution.outputs[0] = to_core_tensor(masked_select(from_core_tensor(execution.inputs[0]),
                                                        from_core_tensor(execution.inputs[1])));
      return Status::Ok();
    }
    switch (execution.op.kind) {
      case OpKind::kFill: {
        const Tensor descriptor = execution.outputs[0];
        execution.outputs[0] =
            to_core_tensor(fill(descriptor.shape, descriptor.dtype, execution.op.scalar_value));
        return Status::Ok();
      }
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
      case OpKind::kSlice:
      case OpKind::kTranspose: {
        const CpuTensor input = from_core_tensor(execution.inputs[0]);
        execution.outputs[0] = to_core_tensor(execute_unary(execution.op, input));
        return Status::Ok();
      }
      case OpKind::kSum:
      case OpKind::kMax:
      case OpKind::kMean: {
        const CpuTensor input = from_core_tensor(execution.inputs[0]);
        execution.outputs[0] = to_core_tensor(reduce(execution.op, input));
        return Status::Ok();
      }
      case OpKind::kAdd:
      case OpKind::kSubtract:
      case OpKind::kDivide:
      case OpKind::kMultiply: {
        const CpuTensor lhs = from_core_tensor(execution.inputs[0]);
        const CpuTensor rhs = from_core_tensor(execution.inputs[1]);
        execution.outputs[0] = to_core_tensor(execute_binary(execution.op, lhs, rhs));
        return Status::Ok();
      }
      case OpKind::kConcat: {
        std::vector<CpuTensor> inputs;
        std::vector<Shape> shapes;
        const auto dtype = execution.inputs[0].dtype;
        for (const auto& tensor : execution.inputs) {
          inputs.push_back(from_core_tensor(tensor));
          if (tensor.dtype != dtype) throw std::invalid_argument("concat dtypes must match");
          shapes.push_back(tensor.shape);
        }
        const auto plan = make_concat_plan(shapes, execution.op.axis);
        CpuTensor result(dtype, plan.output_shape);
        if (result.size()) {
          const auto output_block = plan.output_shape[plan.axis] * plan.inner;
          Dim offset = 0;
          for (const auto& input : inputs) {
            const auto block = input.shape()[plan.axis] * plan.inner;
            const auto copy = [&](const auto& source, auto& target) {
              if (!block) return;
              for (Dim pos = 0; pos < input.size(); pos += block)
                std::memcpy(target.data() + (pos / block) * output_block + offset,
                            source.data() + pos, static_cast<std::size_t>(block) * sizeof(source[0]));
            };
            if (dtype == DType::kFloat32) copy(input.float_data(), result.mutable_float_data());
            else if (dtype == DType::kBool) copy(input.bool_data(), result.mutable_bool_data());
            else copy(input.int32_data(), result.mutable_int32_data());
            offset += block;
          }
        }
        execution.outputs[0] = to_core_tensor(result);
        return Status::Ok();
      }
      case OpKind::kMatmul: {
        const CpuTensor lhs = from_core_tensor(execution.inputs[0]);
        const CpuTensor rhs = from_core_tensor(execution.inputs[1]);
        execution.outputs[0] = to_core_tensor(matmul(lhs, rhs));
        return Status::Ok();
      }
      default:
        return invalid_argument_status("unsupported CPU backend execution operation");
    }
  } catch (const std::invalid_argument& error) {
    return Status(StatusCode::kInvalidArgument, error.what());
  } catch (const std::exception& error) {
    return Status(StatusCode::kInternal, error.what());
  }
}

Tensor to_core_tensor(const CpuTensor& tensor) {
  return Tensor{
      tensor.dtype(),
      tensor.shape(),
      tensor.strides(),
      tensor.device(),
      std::static_pointer_cast<Buffer>(tensor.buffer()),
      0};
}

CpuTensor from_core_tensor(const Tensor& tensor) {
  if (tensor.device.type != "cpu") {
    throw std::invalid_argument("CPU tensor metadata requires device='cpu'");
  }
  if (tensor.device.index != 0) {
    throw std::invalid_argument("CPU tensor metadata requires device index 0");
  }
  if (tensor.offset != 0) {
    throw std::invalid_argument("CPU tensor metadata with non-zero offset is unsupported");
  }
  if (tensor.strides != contiguous_strides(tensor.shape)) {
    throw std::invalid_argument("CPU tensor metadata must be contiguous");
  }
  auto buffer = std::dynamic_pointer_cast<CpuBuffer>(tensor.buffer);
  if (!buffer) {
    throw std::invalid_argument("CPU tensor metadata requires a CPU buffer");
  }
  return CpuTensor(tensor.dtype, tensor.shape, std::move(buffer));
}

Status contract_smoke_test() {
  CpuBackend backend;
  if (backend.name() != "cpu") {
    return Status(StatusCode::kInternal, "CPU backend contract smoke test failed: backend name");
  }

  CpuTensor lhs(Shape{2}, std::vector<float>{1.0F, 2.0F});
  CpuTensor rhs(Shape{2}, std::vector<float>{3.0F, 4.0F});
  std::array<Tensor, 2> inputs{to_core_tensor(lhs), to_core_tensor(rhs)};
  std::array<Tensor, 1> outputs{};
  const BackendExecution valid_add{
      BackendOpClass::kPrimitive,
      OpDesc{OpKind::kAdd},
      std::span<const Tensor>(inputs.data(), inputs.size()),
      std::span<Tensor>(outputs.data(), outputs.size()),
      std::nullopt,
      std::nullopt,
      std::span<const KernelArgument>(),
  };

  if (Status status =
          expect_status_code("valid add", backend.execute(valid_add), StatusCode::kOk);
      !status.ok()) {
    return status;
  }
  const CpuTensor add_result = from_core_tensor(outputs[0]);
  if (add_result.float_data() != std::vector<float>{4.0F, 6.0F}) {
    return Status(StatusCode::kInternal, "CPU backend contract smoke test failed: add result");
  }

  BackendExecution valid_multiply = valid_add;
  valid_multiply.op = OpDesc{OpKind::kMultiply};
  outputs = {};
  valid_multiply.outputs = std::span<Tensor>(outputs.data(), outputs.size());
  if (Status status =
          expect_status_code("valid multiply", backend.execute(valid_multiply), StatusCode::kOk);
      !status.ok()) {
    return status;
  }
  const CpuTensor multiply_result = from_core_tensor(outputs[0]);
  if (multiply_result.float_data() != std::vector<float>{3.0F, 8.0F}) {
    return Status(
        StatusCode::kInternal,
        "CPU backend contract smoke test failed: multiply result");
  }

  const Shape fill_shape{3};
  std::array<Tensor, 1> fill_outputs{Tensor{
      DType::kInt32,
      fill_shape,
      contiguous_strides(fill_shape),
      Device{"cpu", 0},
      nullptr,
      0,
  }};
  const BackendExecution valid_fill{
      BackendOpClass::kPrimitive,
      OpDesc{OpKind::kFill, 0, 1.0e-5, 7.0},
      std::span<const Tensor>(),
      std::span<Tensor>(fill_outputs.data(), fill_outputs.size()),
      std::nullopt,
      std::nullopt,
      std::span<const KernelArgument>(),
  };
  if (Status status =
          expect_status_code("valid fill", backend.execute(valid_fill), StatusCode::kOk);
      !status.ok()) {
    return status;
  }
  const CpuTensor fill_result = from_core_tensor(fill_outputs[0]);
  if (fill_result.int32_data() != std::vector<std::int32_t>{7, 7, 7}) {
    return Status(StatusCode::kInternal, "CPU backend contract smoke test failed: fill result");
  }

  std::array<Tensor, 1> preallocated_fill_outputs{to_core_tensor(lhs)};
  BackendExecution fill_with_preallocated_output = valid_fill;
  fill_with_preallocated_output.outputs =
      std::span<Tensor>(preallocated_fill_outputs.data(), preallocated_fill_outputs.size());
  if (Status status = expect_status_code(
          "fill with preallocated output",
          backend.execute(fill_with_preallocated_output),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  std::array<Tensor, 1> wrong_device_fill_outputs{Tensor{
      DType::kFloat32,
      fill_shape,
      contiguous_strides(fill_shape),
      Device{"metal", 0},
      nullptr,
      0,
  }};
  BackendExecution fill_with_wrong_device = valid_fill;
  fill_with_wrong_device.outputs =
      std::span<Tensor>(wrong_device_fill_outputs.data(), wrong_device_fill_outputs.size());
  if (Status status = expect_status_code(
          "fill with wrong device descriptor",
          backend.execute(fill_with_wrong_device),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  CpuTensor unary_input(Shape{2}, std::vector<float>{0.0F, 1.0F});
  std::array<Tensor, 1> unary_inputs{to_core_tensor(unary_input)};
  std::array<Tensor, 1> unary_outputs{};
  const BackendExecution valid_exp{
      BackendOpClass::kPrimitive,
      OpDesc{OpKind::kExp},
      std::span<const Tensor>(unary_inputs.data(), unary_inputs.size()),
      std::span<Tensor>(unary_outputs.data(), unary_outputs.size()),
      std::nullopt,
      std::nullopt,
      std::span<const KernelArgument>(),
  };

  if (Status status =
          expect_status_code("valid exp", backend.execute(valid_exp), StatusCode::kOk);
      !status.ok()) {
    return status;
  }
  const CpuTensor exp_result = from_core_tensor(unary_outputs[0]);
  const auto& exp_data = exp_result.float_data();
  if (exp_data.size() != 2 || std::abs(exp_data[0] - 1.0F) > 1e-6F ||
      std::abs(exp_data[1] - std::exp(1.0F)) > 1e-6F) {
    return Status(
        StatusCode::kInternal,
        "CPU backend contract smoke test failed: exp result");
  }

  BackendExecution unary_wrong_input_count = valid_exp;
  unary_wrong_input_count.inputs = std::span<const Tensor>(inputs.data(), inputs.size());
  if (Status status = expect_status_code(
          "unary wrong input count",
          backend.execute(unary_wrong_input_count),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  CpuTensor axis_input(Shape{2}, std::vector<float>{1.0F, 2.0F});
  std::array<Tensor, 1> axis_inputs{to_core_tensor(axis_input)};
  std::array<Tensor, 1> axis_outputs{};
  const BackendExecution valid_softmax{
      BackendOpClass::kPrimitive,
      OpDesc{OpKind::kSoftmax, 0},
      std::span<const Tensor>(axis_inputs.data(), axis_inputs.size()),
      std::span<Tensor>(axis_outputs.data(), axis_outputs.size()),
      std::nullopt,
      std::nullopt,
      std::span<const KernelArgument>(),
  };
  if (Status status =
          expect_status_code("valid softmax", backend.execute(valid_softmax), StatusCode::kOk);
      !status.ok()) {
    return status;
  }
  const CpuTensor softmax_result = from_core_tensor(axis_outputs[0]);
  const auto& softmax_data = softmax_result.float_data();
  if (softmax_data.size() != 2 ||
      std::abs((softmax_data[0] + softmax_data[1]) - 1.0F) > 1e-6F) {
    return Status(
        StatusCode::kInternal,
        "CPU backend contract smoke test failed: softmax result");
  }

  CpuTensor reduction_input(
      Shape{2, 3},
      std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  std::array<Tensor, 1> reduction_inputs{to_core_tensor(reduction_input)};
  std::array<Tensor, 1> reduction_outputs{};
  const BackendExecution valid_sum{
      BackendOpClass::kPrimitive,
      OpDesc{OpKind::kSum, 1},
      std::span<const Tensor>(reduction_inputs.data(), reduction_inputs.size()),
      std::span<Tensor>(reduction_outputs.data(), reduction_outputs.size()),
      std::nullopt,
      std::nullopt,
      std::span<const KernelArgument>(),
  };
  if (Status status =
          expect_status_code("valid sum", backend.execute(valid_sum), StatusCode::kOk);
      !status.ok()) {
    return status;
  }
  const CpuTensor sum_result = from_core_tensor(reduction_outputs[0]);
  if (sum_result.shape() != Shape{2} ||
      sum_result.float_data() != std::vector<float>{6.0F, 15.0F}) {
    return Status(
        StatusCode::kInternal,
        "CPU backend contract smoke test failed: sum result");
  }

  CpuTensor matmul_lhs(Shape{2, 2}, std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F});
  CpuTensor matmul_rhs(Shape{2, 2}, std::vector<float>{5.0F, 6.0F, 7.0F, 8.0F});
  std::array<Tensor, 2> matmul_inputs{
      to_core_tensor(matmul_lhs),
      to_core_tensor(matmul_rhs)};
  std::array<Tensor, 1> matmul_outputs{};
  const BackendExecution valid_matmul{
      BackendOpClass::kPrimitive,
      OpDesc{OpKind::kMatmul},
      std::span<const Tensor>(matmul_inputs.data(), matmul_inputs.size()),
      std::span<Tensor>(matmul_outputs.data(), matmul_outputs.size()),
      std::nullopt,
      std::nullopt,
      std::span<const KernelArgument>(),
  };
  if (Status status = expect_status_code(
          "valid matmul",
          backend.execute(valid_matmul),
          StatusCode::kOk);
      !status.ok()) {
    return status;
  }
  const CpuTensor matmul_result = from_core_tensor(matmul_outputs[0]);
  if (matmul_result.shape() != Shape{2, 2} ||
      matmul_result.float_data() != std::vector<float>{19.0F, 22.0F, 43.0F, 50.0F}) {
    return Status(
        StatusCode::kInternal,
        "CPU backend contract smoke test failed: matmul result");
  }

  BackendExecution kernel_op = valid_add;
  kernel_op.op_class = BackendOpClass::kKernel;
  if (Status status =
          expect_status_code("kernel op class", backend.execute(kernel_op), StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  BackendExecution primitive_with_launch = valid_add;
  primitive_with_launch.launch = LaunchConfig{1, 1, 1, 1, 1, 1};
  if (Status status = expect_status_code(
          "primitive with launch metadata",
          backend.execute(primitive_with_launch),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  BackendExecution primitive_with_target = valid_add;
  primitive_with_target.compilation_target =
      CompilationTarget{KernelArtifactKind::kStaticLibrary, "noop_library", "noop"};
  if (Status status = expect_status_code(
          "primitive with compilation target",
          backend.execute(primitive_with_target),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  BackendExecution missing_output = valid_add;
  missing_output.outputs = {};
  if (Status status = expect_status_code(
          "missing output",
          backend.execute(missing_output),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  std::array<Tensor, 2> two_outputs{};
  BackendExecution too_many_outputs = valid_add;
  too_many_outputs.outputs = std::span<Tensor>(two_outputs.data(), two_outputs.size());
  if (Status status = expect_status_code(
          "too many outputs",
          backend.execute(too_many_outputs),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  BackendExecution wrong_input_count = valid_add;
  wrong_input_count.inputs = std::span<const Tensor>(inputs.data(), 1);
  if (Status status = expect_status_code(
          "wrong input count",
          backend.execute(wrong_input_count),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  BackendExecution fill_with_input = valid_fill;
  fill_with_input.inputs = std::span<const Tensor>(inputs.data(), 1);
  fill_outputs[0] = Tensor{
      DType::kFloat32,
      fill_shape,
      contiguous_strides(fill_shape),
      Device{"cpu", 0},
      nullptr,
      0,
  };
  fill_with_input.outputs = std::span<Tensor>(fill_outputs.data(), fill_outputs.size());
  if (Status status = expect_status_code(
          "fill with input",
          backend.execute(fill_with_input),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  std::array<Tensor, 2> non_cpu_inputs{inputs[0], inputs[1]};
  non_cpu_inputs[0].device.type = "metal";
  BackendExecution non_cpu_device = valid_add;
  non_cpu_device.inputs = std::span<const Tensor>(non_cpu_inputs.data(), non_cpu_inputs.size());
  if (Status status = expect_status_code(
          "non-CPU device",
          backend.execute(non_cpu_device),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  std::array<Tensor, 2> non_zero_device_index_inputs{inputs[0], inputs[1]};
  non_zero_device_index_inputs[0].device.index = 1;
  BackendExecution non_zero_device_index = valid_add;
  non_zero_device_index.inputs =
      std::span<const Tensor>(
          non_zero_device_index_inputs.data(),
          non_zero_device_index_inputs.size());
  if (Status status = expect_status_code(
          "non-zero CPU device index",
          backend.execute(non_zero_device_index),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  std::array<Tensor, 2> non_zero_offset_inputs{inputs[0], inputs[1]};
  non_zero_offset_inputs[0].offset = 1;
  BackendExecution non_zero_offset = valid_add;
  non_zero_offset.inputs =
      std::span<const Tensor>(non_zero_offset_inputs.data(), non_zero_offset_inputs.size());
  if (Status status = expect_status_code(
          "non-zero offset",
          backend.execute(non_zero_offset),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  std::array<Tensor, 2> non_contiguous_inputs{inputs[0], inputs[1]};
  non_contiguous_inputs[0].strides = Shape{2};
  BackendExecution non_contiguous = valid_add;
  non_contiguous.inputs =
      std::span<const Tensor>(non_contiguous_inputs.data(), non_contiguous_inputs.size());
  if (Status status = expect_status_code(
          "non-contiguous tensor",
          backend.execute(non_contiguous),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  std::array<Tensor, 2> null_buffer_inputs{inputs[0], inputs[1]};
  null_buffer_inputs[0].buffer.reset();
  BackendExecution null_buffer = valid_add;
  null_buffer.inputs = std::span<const Tensor>(null_buffer_inputs.data(), null_buffer_inputs.size());
  if (Status status = expect_status_code(
          "null CPU buffer",
          backend.execute(null_buffer),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  return Status::Ok();
}

CpuTensor empty(Shape shape, DType dtype) {
  return CpuTensor(dtype, std::move(shape));
}

CpuTensor fill(Shape shape, DType dtype, double value) {
  if (dtype == DType::kInt32 && !is_int32_representable(value)) {
    throw std::invalid_argument("fill value is out of range for int32");
  }

  CpuTensor result(dtype, std::move(shape));
  switch (dtype) {
    case DType::kBool: {
      auto& data = result.mutable_bool_data();
      std::fill(data.begin(), data.end(), static_cast<std::uint8_t>(value != 0));
      break;
    }
    case DType::kFloat32: {
      auto& data = result.mutable_float_data();
      std::fill(data.begin(), data.end(), static_cast<float>(value));
      break;
    }
    case DType::kInt32: {
      auto& data = result.mutable_int32_data();
      std::fill(data.begin(), data.end(), static_cast<std::int32_t>(value));
      break;
    }
  }
  return result;
}

CpuTensor execute_unary(const OpDesc& op, const CpuTensor& input) {
  if (input.device().type != "cpu") {
    throw std::invalid_argument("CPU operations require CPU tensors");
  }
  if (op.kind == OpKind::kTranspose || op.kind == OpKind::kSlice) {
    const auto plan = op.kind == OpKind::kTranspose
        ? make_transpose_plan(input.shape(), op.axes)
        : make_slice_plan(input.shape(), op.slice_starts, op.slice_steps, op.slice_shape);
    CpuTensor result(input.dtype(), plan.output_shape);
    const auto permute = [&](const auto& values, auto& output) {
      for (std::size_t i = 0; i < output.size(); ++i) {
        auto remaining = i;
        Dim source = plan.offset;
        for (std::size_t axis = plan.output_shape.size(); axis-- > 0;) {
          const auto extent = static_cast<std::size_t>(plan.output_shape[axis]);
          source += static_cast<Dim>(remaining % extent) * plan.input_strides[axis];
          remaining /= extent;
        }
        // Reordering is bit-preserving, including NaN payloads and signed zero.
        std::memcpy(&output[i], &values[source], sizeof(output[i]));
      }
    };
    if (input.dtype() == DType::kFloat32) {
      permute(input.float_data(), result.mutable_float_data());
    } else if (input.dtype() == DType::kBool) {
      permute(input.bool_data(), result.mutable_bool_data());
    } else {
      permute(input.int32_data(), result.mutable_int32_data());
    }
    return result;
  }
  if (op.kind == OpKind::kCast) {
    if (input.dtype() == DType::kBool && op.target_dtype == DType::kBool)
      return CpuTensor(input.shape(), input.bool_data());
    if (input.dtype() == DType::kBool || op.target_dtype == DType::kBool) {
      CpuTensor result(op.target_dtype, input.shape());
      const auto convert = [&](const auto& source) {
        for (std::size_t i = 0; i < source.size(); ++i) {
          const bool value = source[i] != 0;
          if (op.target_dtype == DType::kBool) result.mutable_bool_data()[i] = value;
          else if (op.target_dtype == DType::kFloat32) result.mutable_float_data()[i] = value;
          else result.mutable_int32_data()[i] = value;
        }
      };
      if (input.dtype() == DType::kFloat32) convert(input.float_data());
      else if (input.dtype() == DType::kInt32) convert(input.int32_data());
      else convert(input.bool_data());
      return result;
    }
    if (op.target_dtype != DType::kFloat32 && op.target_dtype != DType::kInt32) {
      throw std::invalid_argument("unsupported cast target dtype");
    }
    if (input.dtype() == op.target_dtype) {
      if (input.dtype() == DType::kFloat32) return CpuTensor(input.shape(), input.float_data());
      return CpuTensor(input.shape(), input.int32_data());
    }
    if (op.target_dtype == DType::kInt32) {
      const auto& values = input.float_data();
      // Validate every source before converting: out-of-range floating-to-int
      // conversion is undefined, including NaN and infinity.
      for (const float value : values) {
        if (!std::isfinite(value) || value < -2147483648.0F || value >= 2147483648.0F) {
          throw std::invalid_argument("float32 value is out of range for int32 cast");
        }
      }
      CpuTensor result(op.target_dtype, input.shape());
      auto& out = result.mutable_int32_data();
      for (std::size_t i = 0; i < out.size(); ++i) out[i] = static_cast<std::int32_t>(values[i]);
      return result;
    }
    CpuTensor result(op.target_dtype, input.shape());
    const auto& values = input.int32_data();
    auto& out = result.mutable_float_data();
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = static_cast<float>(values[i]);
    return result;
  }
  if (input.dtype() == DType::kBool) throw std::invalid_argument("numeric operations do not support bool");
  const bool scalar_op = op.kind == OpKind::kAddScalar || op.kind == OpKind::kSubtractScalar ||
                         op.kind == OpKind::kMultiplyScalar || op.kind == OpKind::kDivideScalar;
  if (op.kind == OpKind::kNegate || scalar_op) {
    if (input.dtype() == DType::kInt32 && scalar_op) {
      if (op.kind == OpKind::kDivideScalar) {
        throw std::invalid_argument("division only supports float32 tensors");
      }
      if (!is_int32_representable(op.scalar_value)) {
        throw std::invalid_argument("arithmetic scalar must be an integer in int32 range");
      }
    }
    CpuTensor result(input.dtype(), input.shape());
    if (input.dtype() == DType::kFloat32) {
      const float scalar = static_cast<float>(op.scalar_value);
      const auto& values = input.float_data();
      auto& out = result.mutable_float_data();
      for (std::size_t i = 0; i < out.size(); ++i) {
        const float lhs = op.scalar_left ? scalar : values[i];
        const float rhs = op.scalar_left ? values[i] : scalar;
        switch (op.kind) {
          case OpKind::kNegate: out[i] = -values[i]; break;
          case OpKind::kAddScalar: out[i] = lhs + rhs; break;
          case OpKind::kSubtractScalar: out[i] = lhs - rhs; break;
          case OpKind::kMultiplyScalar: out[i] = lhs * rhs; break;
          case OpKind::kDivideScalar: out[i] = lhs / rhs; break;
          default: break;
        }
      }
    } else {
      const auto scalar = scalar_op ? static_cast<std::uint32_t>(static_cast<std::int32_t>(op.scalar_value)) : 0U;
      const auto& values = input.int32_data();
      auto& out = result.mutable_int32_data();
      for (std::size_t i = 0; i < out.size(); ++i) {
        const auto value = static_cast<std::uint32_t>(values[i]);
        const auto lhs = op.scalar_left ? scalar : value;
        const auto rhs = op.scalar_left ? value : scalar;
        std::uint32_t bits{};
        switch (op.kind) {
          case OpKind::kNegate: bits = 0U - value; break;
          case OpKind::kAddScalar: bits = lhs + rhs; break;
          case OpKind::kSubtractScalar: bits = lhs - rhs; break;
          case OpKind::kMultiplyScalar: bits = lhs * rhs; break;
          default: break;
        }
        out[i] = static_cast<std::int32_t>(bits);
      }
    }
    return result;
  }
  if (op.kind != OpKind::kExp && op.kind != OpKind::kGelu && op.kind != OpKind::kSilu &&
      op.kind != OpKind::kSoftmax && op.kind != OpKind::kRmsNorm &&
      op.kind != OpKind::kLayerNorm) {
    throw std::invalid_argument("unsupported unary operation");
  }
  if (op.kind == OpKind::kSoftmax) {
    const ReductionDims dims = reduction_dims(input, op.axis);
    if (input.dtype() != DType::kFloat32) {
      throw std::invalid_argument("softmax only supports float32 tensors");
    }
    CpuTensor result(input.dtype(), input.shape());
    if (input.size() != 0) {
      compute_softmax(input, result, dims);
    }
    return result;
  }
  if (op.kind == OpKind::kRmsNorm) {
    const ReductionDims dims = reduction_dims(input, op.axis);
    const float epsilon = checked_epsilon(op.epsilon);
    if (input.dtype() != DType::kFloat32) {
      throw std::invalid_argument("rmsnorm only supports float32 tensors");
    }
    if (input.size() == 0) {
      return CpuTensor(input.dtype(), input.shape());
    }
    CpuTensor result(input.dtype(), input.shape());
    compute_rmsnorm(input, result, dims, epsilon);
    return result;
  }
  if (op.kind == OpKind::kLayerNorm) {
    const ReductionDims dims = reduction_dims(input, op.axis);
    const float epsilon = checked_epsilon(op.epsilon);
    if (input.dtype() != DType::kFloat32) {
      throw std::invalid_argument("layernorm only supports float32 tensors");
    }
    if (input.size() == 0) {
      return CpuTensor(input.dtype(), input.shape());
    }
    CpuTensor result(input.dtype(), input.shape());
    compute_layernorm(input, result, dims, epsilon);
    return result;
  }
  if (input.dtype() != DType::kFloat32) {
    switch (op.kind) {
      case OpKind::kExp:
        throw std::invalid_argument("exp only supports float32 tensors");
      case OpKind::kGelu:
        throw std::invalid_argument("gelu only supports float32 tensors");
      case OpKind::kSilu:
        throw std::invalid_argument("silu only supports float32 tensors");
      case OpKind::kRmsNorm:
        throw std::invalid_argument("rmsnorm only supports float32 tensors");
      case OpKind::kLayerNorm:
        throw std::invalid_argument("layernorm only supports float32 tensors");
      default:
        throw std::invalid_argument("unsupported unary operation");
    }
  }

  CpuTensor result(input.dtype(), input.shape());
  const auto& input_data = input.float_data();
  auto& out = result.mutable_float_data();
  for (std::size_t i = 0; i < out.size(); ++i) {
    const float value = input_data[i];
    switch (op.kind) {
      case OpKind::kExp:
        out[i] = std::exp(value);
        break;
      case OpKind::kGelu: {
        const float inner =
            kGeluTanhCoefficient * (value + kGeluCubicCoefficient * value * value * value);
        out[i] = 0.5F * value * (1.0F + std::tanh(inner));
        break;
      }
      case OpKind::kSilu:
        out[i] = value / (1.0F + std::exp(-value));
        break;
      default:
        throw std::invalid_argument("unsupported unary operation");
    }
  }
  return result;
}

CpuTensor execute_binary(const OpDesc& op, const CpuTensor& lhs, const CpuTensor& rhs) {
  if (lhs.dtype() == DType::kBool || rhs.dtype() == DType::kBool)
    throw std::invalid_argument("arithmetic does not support bool tensors");
  validate_binary_inputs(lhs, rhs);
  if (op.kind != OpKind::kAdd && op.kind != OpKind::kSubtract &&
      op.kind != OpKind::kMultiply && op.kind != OpKind::kDivide) {
    throw std::invalid_argument("unsupported binary operation");
  }
  if (op.kind == OpKind::kDivide && lhs.dtype() != DType::kFloat32) {
    throw std::invalid_argument("division only supports float32 tensors");
  }

  const auto plan = make_broadcast_plan(lhs.shape(), rhs.shape());
  CpuTensor result(lhs.dtype(), plan.output_shape);
  const bool same_shape = lhs.shape() == rhs.shape();
  const auto input_offsets = [&](std::size_t index) {
    if (same_shape) return std::pair{index, index};
    std::size_t left = 0, right = 0;
    for (auto axis = plan.output_shape.size(); axis > 0; --axis) {
      const auto coordinate = index % static_cast<std::size_t>(plan.output_shape[axis - 1]);
      index /= static_cast<std::size_t>(plan.output_shape[axis - 1]);
      left += coordinate * static_cast<std::size_t>(plan.lhs_strides[axis - 1]);
      right += coordinate * static_cast<std::size_t>(plan.rhs_strides[axis - 1]);
    }
    return std::pair{left, right};
  };
  switch (lhs.dtype()) {
    case DType::kFloat32: {
      const auto& lhs_data = lhs.float_data();
      const auto& rhs_data = rhs.float_data();
      auto& out = result.mutable_float_data();
      for (std::size_t i = 0; i < out.size(); ++i) {
        const auto [left, right] = input_offsets(i);
        switch (op.kind) {
          case OpKind::kAdd:
            out[i] = lhs_data[left] + rhs_data[right];
            break;
          case OpKind::kSubtract:
            out[i] = lhs_data[left] - rhs_data[right];
            break;
          case OpKind::kDivide:
            out[i] = lhs_data[left] / rhs_data[right];
            break;
          case OpKind::kMultiply:
            out[i] = lhs_data[left] * rhs_data[right];
            break;
          default:
            throw std::invalid_argument("unsupported binary float32 operation");
        }
      }
      break;
    }
    case DType::kInt32: {
      const auto& lhs_data = lhs.int32_data();
      const auto& rhs_data = rhs.int32_data();
      auto& out = result.mutable_int32_data();
      for (std::size_t i = 0; i < out.size(); ++i) {
        const auto [left, right] = input_offsets(i);
        // Compute in uint32 and cast back so int32 overflow is defined
        // two's-complement wraparound (matching NumPy), not signed-overflow UB.
        // This keeps the §12.3 exact-equality contract identical to the Metal
        // add_i32/mul_i32 kernels, which use the same uint round-trip.
        const auto a = static_cast<std::uint32_t>(lhs_data[left]);
        const auto b = static_cast<std::uint32_t>(rhs_data[right]);
        switch (op.kind) {
          case OpKind::kAdd:
            out[i] = static_cast<std::int32_t>(a + b);
            break;
          case OpKind::kSubtract:
            out[i] = static_cast<std::int32_t>(a - b);
            break;
          case OpKind::kMultiply:
            out[i] = static_cast<std::int32_t>(a * b);
            break;
          default:
            throw std::invalid_argument("unsupported binary int32 operation");
        }
      }
      break;
    }
  }
  return result;
}

CpuTensor reduce(const OpDesc& op, const CpuTensor& input) {
  if (input.dtype() == DType::kBool) throw std::invalid_argument("numeric reductions do not support bool tensors");
  if (op.reduction_axes) {
    if (op.kind != OpKind::kSum && op.kind != OpKind::kMax && op.kind != OpKind::kMean)
      throw std::invalid_argument("unsupported CPU reduction operation");
    const auto plan = make_reduction_plan(input.shape(), *op.reduction_axes);
    if (op.kind == OpKind::kMean && input.dtype() != DType::kFloat32)
      throw std::invalid_argument("mean only supports float32 tensors");
    if (op.kind == OpKind::kMax && plan.reduction_size == 0)
      throw std::invalid_argument("max reduction requires non-empty axes");
    if (op.reduction_axes->empty()) {
      OpDesc copy{OpKind::kCast}; copy.target_dtype = input.dtype();
      return execute_unary(copy, input);
    }
    CpuTensor result(input.dtype(), plan.output_shape);
    const auto offset = [&](Dim index, std::size_t begin, std::size_t end) {
      Dim value = 0;
      for (auto axis = end; axis-- > begin;) {
        const auto extent = plan.index_metadata[2 * axis];
        value += (index % extent) * plan.index_metadata[2 * axis + 1];
        index /= extent;
      }
      return static_cast<std::size_t>(value);
    };
    const auto outer_rank = plan.output_shape.size(), rank = input.shape().size();
    for (Dim group = 0; group < result.size(); ++group) {
      const auto base = offset(group, 0, outer_rank);
      if (input.dtype() == DType::kFloat32) {
        float value = op.kind == OpKind::kMax ? -std::numeric_limits<float>::infinity() : 0.0F;
        for (Dim r = 0; r < plan.reduction_size; ++r) {
          const auto item = input.float_data()[base + offset(r, outer_rank, rank)];
          if (op.kind == OpKind::kMax) {
            if (std::isnan(item)) { value = item; break; }
            value = std::max(value, item);
          } else value += item;
        }
        if (op.kind == OpKind::kMean)
          value = plan.reduction_size ? value / static_cast<float>(plan.reduction_size)
                                      : std::numeric_limits<float>::quiet_NaN();
        result.mutable_float_data()[static_cast<std::size_t>(group)] = value;
      } else {
        std::uint32_t sum = 0;
        std::int32_t maximum = std::numeric_limits<std::int32_t>::min();
        for (Dim r = 0; r < plan.reduction_size; ++r) {
          const auto item = input.int32_data()[base + offset(r, outer_rank, rank)];
          if (op.kind == OpKind::kMax) maximum = std::max(maximum, item);
          else sum += static_cast<std::uint32_t>(item);
        }
        result.mutable_int32_data()[static_cast<std::size_t>(group)] =
            op.kind == OpKind::kMax ? maximum : static_cast<std::int32_t>(sum);
      }
    }
    return result;
  }
  const ReductionDims dims = reduction_dims(input, op.axis);
  if (op.kind == OpKind::kMax && dims.reduce == 0) {
    throw std::invalid_argument("max reduction requires a non-empty axis");
  }
  if (op.kind == OpKind::kMean && input.dtype() != DType::kFloat32) {
    throw std::invalid_argument("mean only supports float32 tensors");
  }

  CpuTensor result(input.dtype(), dims.output_shape);
  switch (input.dtype()) {
    case DType::kFloat32: {
      const auto& input_data = input.float_data();
      auto& out = result.mutable_float_data();
      for (std::int64_t id = 0; id < result.size(); ++id) {
        const auto outer_index = id / dims.inner;
        const auto inner_index = id % dims.inner;
        const auto base = outer_index * dims.reduce * dims.inner + inner_index;
        switch (op.kind) {
          case OpKind::kSum: {
            float sum = 0.0F;
            for (std::int64_t reduce_index = 0; reduce_index < dims.reduce; ++reduce_index) {
              sum += input_data[static_cast<std::size_t>(
                  base + reduce_index * dims.inner)];
            }
            out[static_cast<std::size_t>(id)] = sum;
            break;
          }
          case OpKind::kMax: {
            float max_value = -std::numeric_limits<float>::infinity();
            for (std::int64_t reduce_index = 0; reduce_index < dims.reduce; ++reduce_index) {
              const float value =
                  input_data[static_cast<std::size_t>(base + reduce_index * dims.inner)];
              if (std::isnan(value)) {
                max_value = value;
                break;
              }
              max_value = std::max(max_value, value);
            }
            out[static_cast<std::size_t>(id)] = max_value;
            break;
          }
          case OpKind::kMean: {
            if (dims.reduce == 0) {
              out[static_cast<std::size_t>(id)] = std::numeric_limits<float>::quiet_NaN();
              break;
            }
            float sum = 0.0F;
            for (std::int64_t reduce_index = 0; reduce_index < dims.reduce; ++reduce_index) {
              sum += input_data[static_cast<std::size_t>(
                  base + reduce_index * dims.inner)];
            }
            out[static_cast<std::size_t>(id)] = sum / static_cast<float>(dims.reduce);
            break;
          }
          default:
            throw std::invalid_argument("unsupported float32 reduction operation");
        }
      }
      break;
    }
    case DType::kInt32: {
      const auto& input_data = input.int32_data();
      auto& out = result.mutable_int32_data();
      for (std::int64_t id = 0; id < result.size(); ++id) {
        const auto outer_index = id / dims.inner;
        const auto inner_index = id % dims.inner;
        const auto base = outer_index * dims.reduce * dims.inner + inner_index;
        switch (op.kind) {
          case OpKind::kSum: {
            std::uint32_t sum = 0;
            for (std::int64_t reduce_index = 0; reduce_index < dims.reduce; ++reduce_index) {
              sum += static_cast<std::uint32_t>(
                  input_data[static_cast<std::size_t>(base + reduce_index * dims.inner)]);
            }
            out[static_cast<std::size_t>(id)] = static_cast<std::int32_t>(sum);
            break;
          }
          case OpKind::kMax: {
            std::int32_t max_value = std::numeric_limits<std::int32_t>::min();
            for (std::int64_t reduce_index = 0; reduce_index < dims.reduce; ++reduce_index) {
              max_value = std::max(
                  max_value,
                  input_data[static_cast<std::size_t>(base + reduce_index * dims.inner)]);
            }
            out[static_cast<std::size_t>(id)] = max_value;
            break;
          }
          default:
            throw std::invalid_argument("unsupported int32 reduction operation");
        }
      }
      break;
    }
  }
  return result;
}

CpuTensor matmul(const CpuTensor& lhs, const CpuTensor& rhs) {
  if (lhs.device().type != "cpu" || rhs.device().type != "cpu") {
    throw std::invalid_argument("CPU matmul requires CPU tensors");
  }
  if (lhs.dtype() != DType::kFloat32 || rhs.dtype() != DType::kFloat32) {
    throw std::invalid_argument("matmul only supports float32 tensors");
  }
  const auto plan = make_matmul_plan(lhs.shape(), rhs.shape());
  CpuTensor result(DType::kFloat32, plan.output_shape);
  const auto& a = lhs.float_data();
  const auto& b = rhs.float_data();
  auto& out = result.mutable_float_data();
  for (Dim i = 0; i < result.size(); ++i) {
    const Dim col = i % plan.n, row = (i / plan.n) % plan.m;
    const Dim batch = (i / plan.n) / plan.m;
    const auto left = matmul_batch_offset(batch, plan.batch_shape, plan.lhs_batch_strides);
    const auto right = matmul_batch_offset(batch, plan.batch_shape, plan.rhs_batch_strides);
    float sum = 0.0F;
    for (Dim inner = 0; inner < plan.k; ++inner)
      sum += a[left + row * plan.k + inner] * b[right + inner * plan.n + col];
    out[i] = sum;
  }

  return result;
}

}  // namespace tensorcx::cpu
