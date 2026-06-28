#include "cortex/backends/metal/metal_backend.h"

#include <array>
#include <exception>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "cortex/backends/metal/metal_buffer.h"
#include "cortex/backends/metal/metal_context.h"
#include "cortex/core/shape.h"

namespace cortex::metal {
namespace {

Status invalid_argument_status(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
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

}  // namespace

std::string MetalBackend::name() const { return "metal"; }

Status MetalBackend::execute(const BackendExecution& execution) {
  try {
    const Status contract = validate_primitive_execution_contract(execution, "metal");
    if (!contract.ok()) {
      return contract;
    }

    switch (execution.op.kind) {
      case OpKind::kAdd:
      case OpKind::kMultiply: {
        const MetalTensor lhs = from_core_tensor(execution.inputs[0]);
        const MetalTensor rhs = from_core_tensor(execution.inputs[1]);
        auto result = execute_binary(execution.op, lhs, rhs);
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

  switch (tensor.dtype()) {
    case DType::kFloat32: {
      const Status status = buffer->copy_from_host(tensor.float_data().data(), buffer->nbytes());
      if (!status.ok()) {
        return status;
      }
      break;
    }
    case DType::kInt32: {
      const Status status = buffer->copy_from_host(tensor.int32_data().data(), buffer->nbytes());
      if (!status.ok()) {
        return status;
      }
      break;
    }
  }

  return MetalTensor(tensor.dtype(), tensor.shape(), std::move(buffer));
}

Expected<cpu::CpuTensor> to_cpu(const MetalTensor& tensor) {
  switch (tensor.dtype()) {
    case DType::kFloat32: {
      std::vector<float> values(static_cast<std::size_t>(tensor.size()));
      const Status status = tensor.buffer()->copy_to_host(values.data(), tensor.nbytes());
      if (!status.ok()) {
        return status;
      }
      return cpu::CpuTensor(tensor.shape(), std::move(values));
    }
    case DType::kInt32: {
      std::vector<std::int32_t> values(static_cast<std::size_t>(tensor.size()));
      const Status status = tensor.buffer()->copy_to_host(values.data(), tensor.nbytes());
      if (!status.ok()) {
        return status;
      }
      return cpu::CpuTensor(tensor.shape(), std::move(values));
    }
  }
  return Status(StatusCode::kInvalidArgument, "unsupported Metal tensor dtype");
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

}  // namespace cortex::metal
