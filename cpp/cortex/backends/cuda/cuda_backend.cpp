#include "cortex/backends/cuda/cuda_backend.h"

#include <cuda_runtime_api.h>

#include <limits>
#include <stdexcept>
#include <utility>

#include "cortex/backends/cuda/cuda_kernels.h"

namespace cortex::cuda {
namespace {
Status invalid(std::string message) {
  return {StatusCode::kInvalidArgument, std::move(message)};
}
Status cuda_status(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) return Status::Ok();
  const auto code = (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
                     error == cudaErrorInitializationError)
                        ? StatusCode::kUnavailable : StatusCode::kInternal;
  return {code, std::string(operation) + ": " + cudaGetErrorString(error)};
}

// The CUDA current device is thread-local. Every entry point selects our sole
// supported device and restores the caller's selection, including on failure.
class DeviceScope {
 public:
  DeviceScope() {
    error_ = cudaGetDevice(&previous_);
    if (error_ == cudaSuccess) error_ = cudaSetDevice(0);
  }
  ~DeviceScope() {
    if (error_ == cudaSuccess && previous_ != 0) (void)cudaSetDevice(previous_);
  }
  Status status() const { return cuda_status(error_, "select CUDA device 0"); }
  bool ready() const noexcept { return error_ == cudaSuccess; }
 private:
  int previous_{0};
  cudaError_t error_;
};

struct DeviceDeleter {
  void operator()(void* pointer) const noexcept {
    if (!pointer) return;
    DeviceScope device;
    // Destructors cannot report a lost device. Never free on a wrong device.
    if (device.ready()) (void)cudaFree(pointer);
  }
};

Expected<std::size_t> byte_size(DType dtype, const Shape& shape) {
  if (dtype != DType::kFloat32 && dtype != DType::kInt32) return invalid("unsupported CUDA dtype");
  const auto count = static_cast<std::uint64_t>(numel(shape));
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    return invalid("CUDA buffer size overflow");
  }
  return static_cast<std::size_t>(count) * sizeof(float);
}
}  // namespace

class CudaBuffer final : public Buffer {
 public:
  static Expected<std::shared_ptr<CudaBuffer>> create(DType dtype, const Shape& shape) {
    auto bytes = byte_size(dtype, shape);
    if (!bytes) return bytes.status();
    DeviceScope device;
    if (auto status = device.status(); !status.ok()) return status;
    auto buffer = std::make_shared<CudaBuffer>();
    buffer->bytes_ = bytes.value();
    buffer->dtype_ = dtype;
    if (buffer->bytes_) {
      void* pointer = nullptr;
      const auto error = cudaMalloc(&pointer, buffer->bytes_);
      if (error != cudaSuccess) return cuda_status(error, "CUDA allocation");
      buffer->data_.reset(pointer);
    }
    return buffer;
  }
  std::size_t nbytes() const override { return bytes_; }
  DType dtype() const { return dtype_; }
  void* data() const { return data_.get(); }
 private:
  DType dtype_{DType::kFloat32};
  std::size_t bytes_{0};
  std::unique_ptr<void, DeviceDeleter> data_;
};

CudaTensor::CudaTensor(DType dtype, Shape shape, std::shared_ptr<CudaBuffer> buffer)
    : dtype_(dtype), shape_(std::move(shape)), strides_(contiguous_strides(shape_)),
      buffer_(std::move(buffer)) {
  auto bytes = byte_size(dtype_, shape_);
  if (!bytes) throw std::invalid_argument(bytes.status().message());
  if (!buffer_ || buffer_->dtype() != dtype_ || buffer_->nbytes() != bytes.value()) {
    throw std::invalid_argument("CUDA tensor buffer dtype/size mismatch");
  }
}
std::size_t CudaTensor::nbytes() const { return buffer_->nbytes(); }

bool available() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return false;
  DeviceScope device;
  return device.status().ok();
}

Expected<std::string> device_name() {
  DeviceScope device;
  if (auto status = device.status(); !status.ok()) return status;
  cudaDeviceProp properties{};
  const auto error = cudaGetDeviceProperties(&properties, 0);
  if (error != cudaSuccess) return cuda_status(error, "CUDA device discovery");
  return std::string(properties.name);
}

Expected<CudaTensor> from_cpu(const cpu::CpuTensor& tensor) {
  try {
    DeviceScope device;
    if (auto status = device.status(); !status.ok()) return status;
    auto buffer = CudaBuffer::create(tensor.dtype(), tensor.shape());
    if (!buffer) return buffer.status();
    if (buffer.value()->nbytes()) {
      const void* source = tensor.dtype() == DType::kFloat32
                               ? static_cast<const void*>(tensor.float_data().data())
                               : static_cast<const void*>(tensor.int32_data().data());
      const auto error = cudaMemcpy(buffer.value()->data(), source,
                                    buffer.value()->nbytes(), cudaMemcpyHostToDevice);
      if (error != cudaSuccess) return cuda_status(error, "CPU to CUDA copy");
      // Host-to-device pageable copies may return before the DMA completes.
      if (auto status = cuda_status(cudaDeviceSynchronize(), "CPU to CUDA synchronize");
          !status.ok()) return status;
    }
    return CudaTensor(tensor.dtype(), tensor.shape(), buffer.move_value());
  } catch (const std::invalid_argument& error) {
    return invalid(error.what());
  } catch (const std::exception& error) {
    return Status(StatusCode::kInternal, error.what());
  }
}

Expected<cpu::CpuTensor> to_cpu(const CudaTensor& tensor) {
  try {
    DeviceScope device;
    if (auto status = device.status(); !status.ok()) return status;
    cpu::CpuTensor result(tensor.dtype(), tensor.shape());
    if (tensor.nbytes()) {
      void* destination = tensor.dtype() == DType::kFloat32
                              ? static_cast<void*>(result.mutable_float_data().data())
                              : static_cast<void*>(result.mutable_int32_data().data());
      const auto error = cudaMemcpy(destination, tensor.buffer()->data(), tensor.nbytes(),
                                    cudaMemcpyDeviceToHost);
      if (error != cudaSuccess) return cuda_status(error, "CUDA to CPU copy");
    }
    return result;
  } catch (const std::invalid_argument& error) {
    return invalid(error.what());
  } catch (const std::exception& error) {
    return Status(StatusCode::kInternal, error.what());
  }
}

Tensor to_core_tensor(const CudaTensor& tensor) {
  return {tensor.dtype(), tensor.shape(), tensor.strides(), {"cuda", 0}, tensor.buffer(), 0};
}

Expected<CudaTensor> from_core_tensor(const Tensor& tensor) {
  try {
    if (tensor.device.type != "cuda" || tensor.device.index != 0) {
      return invalid("CUDA tensor metadata requires device='cuda', index 0");
    }
    if (tensor.offset != 0 || tensor.strides != contiguous_strides(tensor.shape)) {
      return invalid("CUDA tensor metadata must be contiguous with offset 0");
    }
    auto buffer = std::dynamic_pointer_cast<CudaBuffer>(tensor.buffer);
    if (!buffer) return invalid("CUDA tensor metadata requires a CUDA buffer");
    return CudaTensor(tensor.dtype, tensor.shape, std::move(buffer));
  } catch (const std::invalid_argument& error) {
    return invalid(error.what());
  } catch (const std::exception& error) {
    return Status(StatusCode::kInternal, error.what());
  }
}

Status CudaBackend::execute(const BackendExecution& execution) {
  try {
    if (execution.op_class == BackendOpClass::kKernel) {
      const auto status = validate_kernel_execution_contract(execution);
      return status.ok() ? invalid("CUDA generated kernels are not supported") : status;
    }
    if (auto status = validate_primitive_execution_contract(execution, "cuda");
        !status.ok()) return status;
    const auto kind = execution.op.kind;
    if (kind != OpKind::kFill && kind != OpKind::kAdd && kind != OpKind::kMultiply) {
      return invalid("CUDA prototype only supports fill, add, and multiply");
    }
    Shape shape;
    if (kind == OpKind::kFill) {
      if (execution.outputs[0].dtype != DType::kFloat32) {
        return invalid("CUDA fill only supports float32");
      }
      shape = execution.outputs[0].shape;
    } else {
      auto lhs = from_core_tensor(execution.inputs[0]);
      if (!lhs) return lhs.status();
      auto rhs = from_core_tensor(execution.inputs[1]);
      if (!rhs) return rhs.status();
      if (lhs.value().dtype() != rhs.value().dtype()) return invalid("dtype mismatch for binary operation");
      if (lhs.value().shape() != rhs.value().shape()) return invalid("shape mismatch for binary operation");
      if (lhs.value().dtype() != DType::kFloat32) return invalid("CUDA binary operations only support float32");
      shape = lhs.value().shape();
    }
    // Validate all metadata before touching CUDA or allocating output memory.
    DeviceScope device;
    if (auto status = device.status(); !status.ok()) return status;
    auto buffer = CudaBuffer::create(DType::kFloat32, shape);
    if (!buffer) return buffer.status();
    auto* output = static_cast<float*>(buffer.value()->data());
    const auto count = static_cast<std::size_t>(numel(shape));
    cudaError_t error;
    if (kind == OpKind::kFill) {
      error = launch_fill(output, count, static_cast<float>(execution.op.scalar_value));
    } else {
      const auto lhs = std::static_pointer_cast<CudaBuffer>(execution.inputs[0].buffer);
      const auto rhs = std::static_pointer_cast<CudaBuffer>(execution.inputs[1].buffer);
      error = launch_binary(static_cast<const float*>(lhs->data()),
                            static_cast<const float*>(rhs->data()), output, count,
                            kind == OpKind::kMultiply);
    }
    if (auto status = cuda_status(error, "CUDA kernel execution"); !status.ok()) return status;
    // Publish only on success; failed execution must leave result slots intact.
    execution.outputs[0] = to_core_tensor(CudaTensor(DType::kFloat32, std::move(shape), buffer.move_value()));
    return Status::Ok();
  } catch (const std::invalid_argument& error) {
    return invalid(error.what());
  } catch (const std::exception& error) {
    return {StatusCode::kInternal, error.what()};
  }
}
}  // namespace cortex::cuda
