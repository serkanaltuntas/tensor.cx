#include "tensorcx/backends/metal/metal_backend.h"

#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "tensorcx/backends/cpu/cpu_backend.h"
#include "tensorcx/backends/metal/metal_buffer.h"
#include "tensorcx/backends/metal/metal_context.h"
#include "tensorcx/backends/metal/metal_kernels.h"
#include "tensorcx/backends/metal/metal_kernels_data.h"
#include "tensorcx/backends/metal/metal_library.h"
#if TENSORCX_ENABLE_MPSGRAPH
#include "tensorcx/backends/metal/metal_mpsgraph.h"
#endif
#include "tensorcx/core/shape.h"
#include "tensorcx/core/predicate.h"

namespace tensorcx::metal {
namespace {

Status invalid_argument_status(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Expected<MetalTensor> execute_matmul(
    const OpDesc& op,
    const MetalTensor& lhs,
    const MetalTensor& rhs) {
  switch (op.matmul_preference) {
    case MatmulPreference::kAuto:
#if TENSORCX_ENABLE_MPSGRAPH
      // MPSNDArray is limited to 16 dimensions; custom kernels map any rank.
      if (lhs.shape().size() > 16 || rhs.shape().size() > 16) return matmul_custom(lhs, rhs);
      return matmul_mpsgraph(lhs, rhs);
#else
      return matmul_custom(lhs, rhs);
#endif
    case MatmulPreference::kCustom:
      return matmul_custom(lhs, rhs);
    case MatmulPreference::kOptimized:
#if TENSORCX_ENABLE_MPSGRAPH
      return matmul_mpsgraph(lhs, rhs);
#else
      return invalid_argument_status("optimized Metal matmul backend is not available");
#endif
  }
  return invalid_argument_status("unsupported Metal matmul preference");
}

std::vector<std::uint8_t> artifact_bytes(const std::string& artifact) {
  const auto* first = reinterpret_cast<const std::uint8_t*>(artifact.data());
  return std::vector<std::uint8_t>(first, first + artifact.size());
}

Status validate_metal_kernel_execution(const BackendExecution& execution) {
  const CompilationTarget& target = *execution.compilation_target;
  if (target.artifact_kind != KernelArtifactKind::kBinary &&
      target.artifact_kind != KernelArtifactKind::kStaticLibrary) {
    return invalid_argument_status(
        "Metal kernel execution requires a binary or static library artifact");
  }

  const LaunchConfig& launch = *execution.launch;
  if (launch.grid_y != 1 || launch.grid_z != 1 || launch.threads_per_group_y != 1 ||
      launch.threads_per_group_z != 1) {
    return invalid_argument_status("experimental Metal kernel execution is 1D only");
  }

  for (const Tensor& output : execution.outputs) {
    static_cast<void>(from_core_tensor(output));
  }
  return Status::Ok();
}

Expected<std::string> execute_kernel(const BackendExecution& execution) {
  if (Status status = validate_metal_kernel_execution(execution); !status.ok()) {
    return status;
  }

  std::size_t tensor_argument_count = 0;
  for (const tensorcx::KernelArgument& argument : execution.kernel_arguments) {
    if (argument.kind == KernelArgumentKind::kTensor) {
      ++tensor_argument_count;
    }
  }

  std::vector<MetalTensor> tensor_storage;
  tensor_storage.reserve(tensor_argument_count);
  std::vector<tensorcx::metal::KernelArgument> metal_arguments;
  metal_arguments.reserve(execution.kernel_arguments.size());

  for (const tensorcx::KernelArgument& argument : execution.kernel_arguments) {
    switch (argument.kind) {
      case KernelArgumentKind::kTensor:
        tensor_storage.push_back(from_core_tensor(*argument.tensor));
        metal_arguments.push_back(tensorcx::metal::KernelArgument{
            tensorcx::metal::KernelArgument::Kind::kTensor,
            &tensor_storage.back(),
            0,
        });
        break;
      case KernelArgumentKind::kUInt32:
        metal_arguments.push_back(tensorcx::metal::KernelArgument{
            tensorcx::metal::KernelArgument::Kind::kUInt32,
            nullptr,
            argument.uint32_value,
        });
        break;
    }
  }

  const LaunchConfig& launch = *execution.launch;
  return launch_library_function(
      artifact_bytes(execution.compilation_target->artifact),
      execution.compilation_target->entry_point,
      metal_arguments,
      launch.grid_x,
      launch.threads_per_group_x);
}

Status expect_status_code(
    const char* scenario,
    const Status& status,
    StatusCode expected) {
  if (status.code() == expected) {
    return Status::Ok();
  }
  return Status(
      StatusCode::kInternal,
      std::string("Metal backend contract smoke test failed: ") + scenario);
}

Status expect_float_data_close(
    const char* scenario,
    const cpu::CpuTensor& actual,
    const cpu::CpuTensor& expected) {
  if (actual.shape() != expected.shape()) {
    return Status(
        StatusCode::kInternal,
        std::string("Metal backend contract smoke test failed: ") + scenario);
  }
  const auto& actual_data = actual.float_data();
  const auto& expected_data = expected.float_data();
  if (actual_data.size() != expected_data.size()) {
    return Status(
        StatusCode::kInternal,
        std::string("Metal backend contract smoke test failed: ") + scenario);
  }
  for (std::size_t i = 0; i < actual_data.size(); ++i) {
    if (std::fabs(actual_data[i] - expected_data[i]) > 1.0e-5F) {
      return Status(
          StatusCode::kInternal,
          std::string("Metal backend contract smoke test failed: ") + scenario);
    }
  }
  return Status::Ok();
}

}  // namespace

std::string MetalBackend::name() const { return "metal"; }

Status MetalBackend::execute(const BackendExecution& execution) {
  try {
    if (execution.op_class == BackendOpClass::kKernel) {
      const Status contract = validate_kernel_execution_contract(execution);
      if (!contract.ok()) {
        return contract;
      }
      auto result = execute_kernel(execution);
      if (!result) {
        return result.status();
      }
      return Status::Ok();
    }

    const Status contract = validate_primitive_execution_contract(execution, "metal");
    if (!contract.ok()) {
      return contract;
    }

    if (is_predicate_elementwise(execution.op.kind) || execution.op.kind == OpKind::kAny ||
        execution.op.kind == OpKind::kAll || execution.op.kind == OpKind::kMaskedSelect) {
      std::vector<MetalTensor> inputs;
      for (const auto& input : execution.inputs) inputs.push_back(from_core_tensor(input));
      auto result = execute_predicate(execution.op, inputs);
      if (!result) return result.status();
      execution.outputs[0] = to_core_tensor(result.move_value());
      return Status::Ok();
    }
    switch (execution.op.kind) {
      case OpKind::kAdd:
      case OpKind::kSubtract:
      case OpKind::kMultiply:
      case OpKind::kDivide: {
        const MetalTensor lhs = from_core_tensor(execution.inputs[0]);
        const MetalTensor rhs = from_core_tensor(execution.inputs[1]);
        auto result = execute_binary(execution.op, lhs, rhs);
        if (!result) {
          return result.status();
        }
        execution.outputs[0] = to_core_tensor(result.move_value());
        return Status::Ok();
      }
      case OpKind::kConcat: {
        std::vector<MetalTensor> inputs;
        for (const auto& input : execution.inputs) inputs.push_back(from_core_tensor(input));
        auto result = execute_concat(execution.op, inputs);
        if (!result) return result.status();
        execution.outputs[0] = to_core_tensor(result.move_value());
        return Status::Ok();
      }
      case OpKind::kMatmul: {
        const MetalTensor lhs = from_core_tensor(execution.inputs[0]);
        const MetalTensor rhs = from_core_tensor(execution.inputs[1]);
        auto result = execute_matmul(execution.op, lhs, rhs);
        if (!result) {
          return result.status();
        }
        execution.outputs[0] = to_core_tensor(result.move_value());
        return Status::Ok();
      }
      case OpKind::kCast:
      case OpKind::kTranspose:
      case OpKind::kSlice:
      case OpKind::kNegate:
      case OpKind::kAddScalar:
      case OpKind::kSubtractScalar:
      case OpKind::kMultiplyScalar:
      case OpKind::kDivideScalar:
      case OpKind::kExp:
      case OpKind::kGelu:
      case OpKind::kSilu:
      case OpKind::kSoftmax:
      case OpKind::kRmsNorm:
      case OpKind::kLayerNorm: {
        const MetalTensor input = from_core_tensor(execution.inputs[0]);
        auto result = execute_unary(execution.op, input);
        if (!result) {
          return result.status();
        }
        execution.outputs[0] = to_core_tensor(result.move_value());
        return Status::Ok();
      }
      case OpKind::kSum:
      case OpKind::kMax:
      case OpKind::kMean: {
        const MetalTensor input = from_core_tensor(execution.inputs[0]);
        auto result = reduce(execution.op, input);
        if (!result) {
          return result.status();
        }
        execution.outputs[0] = to_core_tensor(result.move_value());
        return Status::Ok();
      }
      case OpKind::kFill: {
        const Tensor descriptor = execution.outputs[0];
        auto result = fill(
            execution.op,
            descriptor.shape,
            descriptor.dtype,
            execution.op.scalar_value);
        if (!result) {
          return result.status();
        }
        execution.outputs[0] = to_core_tensor(result.move_value());
        return Status::Ok();
      }
      default:
        return invalid_argument_status("unsupported Metal backend execution operation");
    }
  } catch (const std::invalid_argument& error) {
    return Status(StatusCode::kInvalidArgument, error.what());
  } catch (const std::exception& error) {
    return Status(StatusCode::kInternal, error.what());
  }
}

bool available() {
  return default_context().ready();
}

std::vector<std::string> devices() {
  if (!available()) {
    return {};
  }
  auto& context = default_context();
  if (!context.ready()) {
    return {};
  }
  return {context.device_name()};
}

Expected<MetalTensor> from_cpu(const cpu::CpuTensor& tensor) {
  auto buffer_result = MetalBuffer::create(
      tensor.dtype(), static_cast<std::size_t>(tensor.size()));
  if (!buffer_result) {
    return buffer_result.status();
  }
  auto buffer = buffer_result.move_value();

  const auto status = buffer->copy_from_host(tensor.data(), buffer->nbytes());
  if (!status.ok()) return status;

  return MetalTensor(tensor.dtype(), tensor.shape(), std::move(buffer));
}

Expected<cpu::CpuTensor> to_cpu(const MetalTensor& tensor) {
  cpu::CpuTensor result(tensor.dtype(), tensor.shape());
  const auto status = tensor.buffer()->copy_to_host(result.mutable_data(), tensor.nbytes());
  if (!status.ok()) return status;
  return result;
}

Tensor to_core_tensor(const MetalTensor& tensor) {
  return Tensor{
      tensor.dtype(),
      tensor.shape(),
      tensor.strides(),
      tensor.device(),
      std::static_pointer_cast<Buffer>(tensor.buffer()),
      0};
}

MetalTensor from_core_tensor(const Tensor& tensor) {
  if (tensor.device.type != "metal") {
    throw std::invalid_argument("Metal tensor metadata requires device='metal'");
  }
  if (tensor.device.index != 0) {
    throw std::invalid_argument("Metal tensor metadata requires device index 0");
  }
  if (tensor.offset != 0) {
    throw std::invalid_argument("Metal tensor metadata with non-zero offset is unsupported");
  }
  if (tensor.strides != contiguous_strides(tensor.shape)) {
    throw std::invalid_argument("Metal tensor metadata must be contiguous");
  }
  auto buffer = std::dynamic_pointer_cast<MetalBuffer>(tensor.buffer);
  if (!buffer) {
    throw std::invalid_argument("Metal tensor metadata requires a Metal buffer");
  }
  return MetalTensor(tensor.dtype, tensor.shape, std::move(buffer));
}

Status contract_smoke_test() {
  // An element count whose byte size cannot be represented must fail with a
  // clear status; a wrapped multiplication here would otherwise allocate a
  // tiny buffer that later copies would overrun. The guard fires before any
  // device access, so this case runs (and protects) even on hosts without a
  // usable Metal device, ahead of the availability skip below.
  auto overflow_buffer_result = MetalBuffer::create(
      DType::kFloat32, std::numeric_limits<std::size_t>::max());
  if (overflow_buffer_result) {
    return Status(
        StatusCode::kInternal,
        "Metal backend contract smoke test failed: buffer size overflow accepted");
  }
  if (Status status = expect_status_code(
          "buffer size overflow",
          overflow_buffer_result.status(),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  auto& context = default_context();
  if (!context.ready()) {
    return context.status();
  }

  MetalBackend backend;
  if (backend.name() != "metal") {
    return Status(StatusCode::kInternal, "Metal backend contract smoke test failed: backend name");
  }

  cpu::CpuTensor lhs_cpu(Shape{2}, std::vector<float>{1.0F, 2.0F});
  cpu::CpuTensor rhs_cpu(Shape{2}, std::vector<float>{3.0F, 4.0F});
  auto lhs_metal_result = from_cpu(lhs_cpu);
  if (!lhs_metal_result) {
    return lhs_metal_result.status();
  }
  auto rhs_metal_result = from_cpu(rhs_cpu);
  if (!rhs_metal_result) {
    return rhs_metal_result.status();
  }
  const MetalTensor lhs_metal = lhs_metal_result.move_value();
  const MetalTensor rhs_metal = rhs_metal_result.move_value();
  std::array<Tensor, 2> inputs{to_core_tensor(lhs_metal), to_core_tensor(rhs_metal)};
  std::array<Tensor, 1> outputs{};
  const BackendExecution valid_add{
      BackendOpClass::kPrimitive,
      OpDesc{OpKind::kAdd},
      std::span<const Tensor>(inputs.data(), inputs.size()),
      std::span<Tensor>(outputs.data(), outputs.size()),
      std::nullopt,
      std::nullopt,
      std::span<const tensorcx::KernelArgument>(),
  };
  if (Status status =
          expect_status_code("valid add", backend.execute(valid_add), StatusCode::kOk);
      !status.ok()) {
    return status;
  }
  const auto add_cpu_result = to_cpu(from_core_tensor(outputs[0]));
  if (!add_cpu_result) {
    return add_cpu_result.status();
  }
  if (add_cpu_result.value().float_data() != std::vector<float>{4.0F, 6.0F}) {
    return Status(StatusCode::kInternal, "Metal backend contract smoke test failed: add result");
  }

  auto kernel_output_buffer_result = MetalBuffer::create(DType::kFloat32, 2);
  if (!kernel_output_buffer_result) {
    return kernel_output_buffer_result.status();
  }
  MetalTensor kernel_output_metal(
      DType::kFloat32,
      Shape{2},
      kernel_output_buffer_result.move_value());
  std::array<Tensor, 1> kernel_outputs{to_core_tensor(kernel_output_metal)};
  std::array<Tensor, 3> kernel_argument_tensors{
      inputs[0],
      inputs[1],
      kernel_outputs[0],
  };
  std::array<tensorcx::KernelArgument, 4> kernel_arguments{{
      tensorcx::KernelArgument{
          tensorcx::KernelArgumentKind::kTensor,
          &kernel_argument_tensors[0],
          0},
      tensorcx::KernelArgument{
          tensorcx::KernelArgumentKind::kTensor,
          &kernel_argument_tensors[1],
          0},
      tensorcx::KernelArgument{
          tensorcx::KernelArgumentKind::kTensor,
          &kernel_argument_tensors[2],
          0},
      tensorcx::KernelArgument{tensorcx::KernelArgumentKind::kUInt32, nullptr, 2},
  }};
  const std::string kernel_artifact(
      reinterpret_cast<const char*>(kElementwiseMetallib),
      kElementwiseMetallibSize);
  const BackendExecution valid_kernel{
      BackendOpClass::kKernel,
      OpDesc{},
      std::span<const Tensor>(),
      std::span<Tensor>(kernel_outputs.data(), kernel_outputs.size()),
      LaunchConfig{2, 1, 1, 2, 1, 1},
      CompilationTarget{KernelArtifactKind::kStaticLibrary, kernel_artifact, "add_f32"},
      std::span<const tensorcx::KernelArgument>(kernel_arguments.data(), kernel_arguments.size()),
  };
  if (Status status =
          expect_status_code("valid kernel", backend.execute(valid_kernel), StatusCode::kOk);
      !status.ok()) {
    return status;
  }
  const auto kernel_cpu_result = to_cpu(from_core_tensor(kernel_outputs[0]));
  if (!kernel_cpu_result) {
    return kernel_cpu_result.status();
  }
  if (kernel_cpu_result.value().float_data() != std::vector<float>{4.0F, 6.0F}) {
    return Status(
        StatusCode::kInternal,
        "Metal backend contract smoke test failed: kernel result");
  }

  BackendExecution non_1d_kernel = valid_kernel;
  non_1d_kernel.launch = LaunchConfig{2, 2, 1, 2, 1, 1};
  if (Status status = expect_status_code(
          "non-1D kernel launch",
          backend.execute(non_1d_kernel),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  BackendExecution unsupported_kernel_artifact = valid_kernel;
  unsupported_kernel_artifact.compilation_target =
      CompilationTarget{KernelArtifactKind::kSource, kernel_artifact, "add_f32"};
  if (Status status = expect_status_code(
          "unsupported kernel artifact kind",
          backend.execute(unsupported_kernel_artifact),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  std::array<Tensor, 1> wrong_device_kernel_outputs{kernel_outputs[0]};
  wrong_device_kernel_outputs[0].device.type = "cpu";
  BackendExecution wrong_device_kernel_output = valid_kernel;
  wrong_device_kernel_output.outputs =
      std::span<Tensor>(wrong_device_kernel_outputs.data(), wrong_device_kernel_outputs.size());
  if (Status status = expect_status_code(
          "wrong-device kernel output",
          backend.execute(wrong_device_kernel_output),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
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
  const auto multiply_cpu_result = to_cpu(from_core_tensor(outputs[0]));
  if (!multiply_cpu_result) {
    return multiply_cpu_result.status();
  }
  if (multiply_cpu_result.value().float_data() != std::vector<float>{3.0F, 8.0F}) {
    return Status(
        StatusCode::kInternal,
        "Metal backend contract smoke test failed: multiply result");
  }

  std::array<Tensor, 2> wrong_device_inputs{inputs[0], inputs[1]};
  wrong_device_inputs[0].device.type = "cpu";
  BackendExecution wrong_device_add = valid_add;
  wrong_device_add.inputs =
      std::span<const Tensor>(wrong_device_inputs.data(), wrong_device_inputs.size());
  if (Status status = expect_status_code(
          "add with wrong device input",
          backend.execute(wrong_device_add),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  std::array<Tensor, 1> unary_inputs{inputs[0]};
  std::array<Tensor, 1> unary_outputs{};
  auto expect_unary_matches_cpu = [&](OpKind kind, const char* scenario) -> Status {
    unary_outputs = {};
    const BackendExecution valid_unary{
        BackendOpClass::kPrimitive,
        OpDesc{kind},
        std::span<const Tensor>(unary_inputs.data(), unary_inputs.size()),
        std::span<Tensor>(unary_outputs.data(), unary_outputs.size()),
        std::nullopt,
        std::nullopt,
        std::span<const tensorcx::KernelArgument>(),
    };
    if (Status status =
            expect_status_code(scenario, backend.execute(valid_unary), StatusCode::kOk);
        !status.ok()) {
      return status;
    }
    const auto actual_result = to_cpu(from_core_tensor(unary_outputs[0]));
    if (!actual_result) {
      return actual_result.status();
    }
    const cpu::CpuTensor expected = cpu::execute_unary(OpDesc{kind}, lhs_cpu);
    return expect_float_data_close(scenario, actual_result.value(), expected);
  };

  if (Status status = expect_unary_matches_cpu(OpKind::kExp, "valid exp"); !status.ok()) {
    return status;
  }
  if (Status status = expect_unary_matches_cpu(OpKind::kGelu, "valid gelu"); !status.ok()) {
    return status;
  }
  if (Status status = expect_unary_matches_cpu(OpKind::kSilu, "valid silu"); !status.ok()) {
    return status;
  }

  auto expect_axis_transform_matches_cpu =
      [&](const OpDesc& op, const char* scenario) -> Status {
    unary_outputs = {};
    const BackendExecution valid_transform{
        BackendOpClass::kPrimitive,
        op,
        std::span<const Tensor>(unary_inputs.data(), unary_inputs.size()),
        std::span<Tensor>(unary_outputs.data(), unary_outputs.size()),
        std::nullopt,
        std::nullopt,
        std::span<const tensorcx::KernelArgument>(),
    };
    if (Status status =
            expect_status_code(scenario, backend.execute(valid_transform), StatusCode::kOk);
        !status.ok()) {
      return status;
    }
    const auto actual_result = to_cpu(from_core_tensor(unary_outputs[0]));
    if (!actual_result) {
      return actual_result.status();
    }
    const cpu::CpuTensor expected = cpu::execute_unary(op, lhs_cpu);
    return expect_float_data_close(scenario, actual_result.value(), expected);
  };

  if (Status status =
          expect_axis_transform_matches_cpu(OpDesc{OpKind::kSoftmax, 0}, "valid softmax");
      !status.ok()) {
    return status;
  }
  if (Status status =
          expect_axis_transform_matches_cpu(OpDesc{OpKind::kRmsNorm, 0}, "valid rmsnorm");
      !status.ok()) {
    return status;
  }
  if (Status status = expect_axis_transform_matches_cpu(
          OpDesc{OpKind::kLayerNorm, 0},
          "valid layernorm");
      !status.ok()) {
    return status;
  }

  auto expect_reduction_matches_cpu = [&](const OpDesc& op, const char* scenario) -> Status {
    unary_outputs = {};
    const BackendExecution valid_reduction{
        BackendOpClass::kPrimitive,
        op,
        std::span<const Tensor>(unary_inputs.data(), unary_inputs.size()),
        std::span<Tensor>(unary_outputs.data(), unary_outputs.size()),
        std::nullopt,
        std::nullopt,
        std::span<const tensorcx::KernelArgument>(),
    };
    if (Status status =
            expect_status_code(scenario, backend.execute(valid_reduction), StatusCode::kOk);
        !status.ok()) {
      return status;
    }
    const auto actual_result = to_cpu(from_core_tensor(unary_outputs[0]));
    if (!actual_result) {
      return actual_result.status();
    }
    const cpu::CpuTensor expected = cpu::reduce(op, lhs_cpu);
    return expect_float_data_close(scenario, actual_result.value(), expected);
  };

  if (Status status = expect_reduction_matches_cpu(OpDesc{OpKind::kSum, 0}, "valid sum");
      !status.ok()) {
    return status;
  }
  if (Status status = expect_reduction_matches_cpu(OpDesc{OpKind::kMax, 0}, "valid max");
      !status.ok()) {
    return status;
  }
  if (Status status = expect_reduction_matches_cpu(OpDesc{OpKind::kMean, 0}, "valid mean");
      !status.ok()) {
    return status;
  }

  cpu::CpuTensor matmul_lhs_cpu(Shape{2, 2}, std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F});
  cpu::CpuTensor matmul_rhs_cpu(Shape{2, 2}, std::vector<float>{5.0F, 6.0F, 7.0F, 8.0F});
  auto matmul_lhs_metal_result = from_cpu(matmul_lhs_cpu);
  if (!matmul_lhs_metal_result) {
    return matmul_lhs_metal_result.status();
  }
  auto matmul_rhs_metal_result = from_cpu(matmul_rhs_cpu);
  if (!matmul_rhs_metal_result) {
    return matmul_rhs_metal_result.status();
  }
  const MetalTensor matmul_lhs_metal = matmul_lhs_metal_result.move_value();
  const MetalTensor matmul_rhs_metal = matmul_rhs_metal_result.move_value();
  std::array<Tensor, 2> matmul_inputs{
      to_core_tensor(matmul_lhs_metal),
      to_core_tensor(matmul_rhs_metal)};
  std::array<Tensor, 1> matmul_outputs{};
  auto expect_matmul_matches_cpu =
      [&](MatmulPreference preference, const char* scenario) -> Status {
    matmul_outputs = {};
    OpDesc op{OpKind::kMatmul};
    op.matmul_preference = preference;
    const BackendExecution valid_matmul{
        BackendOpClass::kPrimitive,
        op,
        std::span<const Tensor>(matmul_inputs.data(), matmul_inputs.size()),
        std::span<Tensor>(matmul_outputs.data(), matmul_outputs.size()),
        std::nullopt,
        std::nullopt,
        std::span<const tensorcx::KernelArgument>(),
    };
    if (Status status =
            expect_status_code(scenario, backend.execute(valid_matmul), StatusCode::kOk);
        !status.ok()) {
      return status;
    }
    const auto actual_result = to_cpu(from_core_tensor(matmul_outputs[0]));
    if (!actual_result) {
      return actual_result.status();
    }
    const cpu::CpuTensor expected = cpu::matmul(matmul_lhs_cpu, matmul_rhs_cpu);
    return expect_float_data_close(scenario, actual_result.value(), expected);
  };

  if (Status status = expect_matmul_matches_cpu(MatmulPreference::kAuto, "valid matmul auto");
      !status.ok()) {
    return status;
  }
  if (Status status =
          expect_matmul_matches_cpu(MatmulPreference::kCustom, "valid matmul custom");
      !status.ok()) {
    return status;
  }
#if TENSORCX_ENABLE_MPSGRAPH
  if (Status status =
          expect_matmul_matches_cpu(MatmulPreference::kOptimized, "valid matmul optimized");
      !status.ok()) {
    return status;
  }
#else
  matmul_outputs = {};
  OpDesc optimized_matmul{OpKind::kMatmul};
  optimized_matmul.matmul_preference = MatmulPreference::kOptimized;
  const BackendExecution unavailable_optimized_matmul{
      BackendOpClass::kPrimitive,
      optimized_matmul,
      std::span<const Tensor>(matmul_inputs.data(), matmul_inputs.size()),
      std::span<Tensor>(matmul_outputs.data(), matmul_outputs.size()),
      std::nullopt,
      std::nullopt,
  };
  if (Status status = expect_status_code(
          "unavailable optimized matmul",
          backend.execute(unavailable_optimized_matmul),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }
#endif

  const Shape fill_shape{3};
  std::array<Tensor, 1> fill_outputs{Tensor{
      DType::kFloat32,
      fill_shape,
      contiguous_strides(fill_shape),
      Device{"metal", 0},
      nullptr,
      0,
  }};
  const BackendExecution valid_fill{
      BackendOpClass::kPrimitive,
      OpDesc{OpKind::kFill, 0, 1.0e-5, 2.5},
      std::span<const Tensor>(),
      std::span<Tensor>(fill_outputs.data(), fill_outputs.size()),
      std::nullopt,
      std::nullopt,
      std::span<const tensorcx::KernelArgument>(),
  };
  if (Status status =
          expect_status_code("valid fill", backend.execute(valid_fill), StatusCode::kOk);
      !status.ok()) {
    return status;
  }

  const auto fill_cpu_result = to_cpu(from_core_tensor(fill_outputs[0]));
  if (!fill_cpu_result) {
    return fill_cpu_result.status();
  }
  const auto fill_cpu = fill_cpu_result.value();
  if (fill_cpu.float_data() != std::vector<float>{2.5F, 2.5F, 2.5F}) {
    return Status(StatusCode::kInternal, "Metal backend contract smoke test failed: fill result");
  }

  std::array<Tensor, 1> wrong_device_outputs{Tensor{
      DType::kFloat32,
      fill_shape,
      contiguous_strides(fill_shape),
      Device{"cpu", 0},
      nullptr,
      0,
  }};
  BackendExecution wrong_device_fill = valid_fill;
  wrong_device_fill.outputs =
      std::span<Tensor>(wrong_device_outputs.data(), wrong_device_outputs.size());
  if (Status status = expect_status_code(
          "fill with wrong device descriptor",
          backend.execute(wrong_device_fill),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  std::array<Tensor, 1> preallocated_outputs{fill_outputs[0]};
  BackendExecution preallocated_fill = valid_fill;
  preallocated_fill.outputs =
      std::span<Tensor>(preallocated_outputs.data(), preallocated_outputs.size());
  if (Status status = expect_status_code(
          "fill with preallocated output",
          backend.execute(preallocated_fill),
          StatusCode::kInvalidArgument);
      !status.ok()) {
    return status;
  }

  return Status::Ok();
}

}  // namespace tensorcx::metal
