#include "tensorcx/backends/cuda/cuda_backend.h"

#include <cuda_runtime_api.h>

#include <limits>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "tensorcx/backends/cuda/cuda_kernels.h"
#include "tensorcx/backends/cuda/cuda_buffer.h"
#include "tensorcx/backends/cuda/cuda_kernel.h"

namespace tensorcx::cuda {
namespace {
Status invalid(std::string message) {
  return {StatusCode::kInvalidArgument, std::move(message)};
}
Expected<std::size_t> byte_size(DType dtype, const Shape& shape) {
  if (dtype != DType::kFloat32 && dtype != DType::kInt32) return invalid("unsupported CUDA dtype");
  const auto count = static_cast<std::uint64_t>(numel(shape));
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    return invalid("CUDA buffer size overflow");
  }
  return static_cast<std::size_t>(count) * sizeof(float);
}
}  // namespace

void DeviceDeleter::operator()(void* pointer) const noexcept {
  if (!pointer) return;
  ContextScope context(owner);
  if (context.ready()) (void)cudaFree(pointer);
}

Expected<std::shared_ptr<CudaBuffer>> CudaBuffer::create(DType dtype, const Shape& shape) {
  auto bytes = byte_size(dtype, shape);
  if (!bytes) return bytes.status();
  ContextScope context;
  if (!context.ready()) return context.status();
  auto buffer = std::make_shared<CudaBuffer>();
  buffer->bytes_ = bytes.value();
  buffer->dtype_ = dtype;
  buffer->data_.get_deleter().owner = context.owner();
  if (buffer->bytes_) {
    void* pointer = nullptr;
    auto status = runtime_status(cudaMalloc(&pointer, buffer->bytes_), "CUDA allocation");
    if (!status.ok()) return status;
    buffer->data_.reset(pointer);
  }
  return buffer;
}

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
  ContextScope device;
  return device.status().ok();
}

Expected<std::string> device_name() {
  ContextScope device;
  if (auto status = device.status(); !status.ok()) return status;
  cudaDeviceProp properties{};
  const auto error = cudaGetDeviceProperties(&properties, 0);
  if (error != cudaSuccess) return runtime_status(error, "CUDA device discovery");
  return std::string(properties.name);
}

Expected<CudaTensor> from_cpu(const cpu::CpuTensor& tensor) {
  try {
    ContextScope device;
    if (auto status = device.status(); !status.ok()) return status;
    auto buffer = CudaBuffer::create(tensor.dtype(), tensor.shape());
    if (!buffer) return buffer.status();
    if (buffer.value()->nbytes()) {
      const void* source = tensor.dtype() == DType::kFloat32
                               ? static_cast<const void*>(tensor.float_data().data())
                               : static_cast<const void*>(tensor.int32_data().data());
      const auto error = cudaMemcpy(buffer.value()->data(), source,
                                    buffer.value()->nbytes(), cudaMemcpyHostToDevice);
      if (error != cudaSuccess) return runtime_status(error, "CPU to CUDA copy");
      // Host-to-device pageable copies may return before the DMA completes.
      if (auto status = runtime_status(cudaDeviceSynchronize(), "CPU to CUDA synchronize");
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
    ContextScope device;
    if (auto status = device.status(); !status.ok()) return status;
    cpu::CpuTensor result(tensor.dtype(), tensor.shape());
    if (tensor.nbytes()) {
      void* destination = tensor.dtype() == DType::kFloat32
                              ? static_cast<void*>(result.mutable_float_data().data())
                              : static_cast<void*>(result.mutable_int32_data().data());
      const auto error = cudaMemcpy(destination, tensor.buffer()->data(), tensor.nbytes(),
                                    cudaMemcpyDeviceToHost);
      if (error != cudaSuccess) return runtime_status(error, "CUDA to CPU copy");
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
      return execute_compiled_kernel(execution);
    }
    if (auto status = validate_primitive_execution_contract(execution, "cuda");
        !status.ok()) return status;
    const auto kind = execution.op.kind;
    Shape shape;
    std::size_t outer=1, reduce=1, inner=1;
    float epsilon=0;
    for (const auto& input : execution.inputs) {
      auto tensor=from_core_tensor(input);
      if (!tensor) return tensor.status();
    }
    if ((kind==OpKind::kAdd || kind==OpKind::kMultiply) &&
        execution.inputs[0].dtype!=execution.inputs[1].dtype) return invalid("dtype mismatch for binary operation");
    for (const auto& input : execution.inputs) {
      if (input.dtype!=DType::kFloat32) return invalid("CUDA operations only support float32");
    }
    if (kind==OpKind::kFill) {
      if(execution.outputs[0].dtype!=DType::kFloat32) return invalid("CUDA fill only supports float32");
      shape=execution.outputs[0].shape;
    } else {
      shape=execution.inputs[0].shape;
      if (kind==OpKind::kAdd || kind==OpKind::kMultiply) {
        if(shape!=execution.inputs[1].shape) return invalid("shape mismatch for binary operation");
      } else if(kind==OpKind::kMatmul) {
        const auto& right=execution.inputs[1].shape;
        if(shape.size()!=2 || right.size()!=2) return invalid("matmul requires rank-2 tensors");
        if(shape[1]!=right[0]) return invalid("matmul shape mismatch");
        if(execution.op.matmul_preference!=MatmulPreference::kAuto &&
           execution.op.matmul_preference!=MatmulPreference::kCustom) return invalid("CUDA matmul supports auto or custom");
        shape={shape[0],right[1]};
      } else if(kind!=OpKind::kExp && kind!=OpKind::kGelu && kind!=OpKind::kSilu) {
        auto axis=execution.op.axis;
        const auto rank=static_cast<std::int64_t>(shape.size());
        if(!rank) {
          if(axis!=0 && axis!=-1) return invalid("reduction axis is out of range");
        } else {
          if(axis<0)axis+=rank;
          if(axis<0 || axis>=rank)return invalid("reduction axis is out of range");
          Shape before(shape.begin(),shape.begin()+axis),after(shape.begin()+axis+1,shape.end());
          outer=static_cast<std::size_t>(numel(before));
          inner=static_cast<std::size_t>(numel(after));
          reduce=static_cast<std::size_t>(shape[axis]);
          if(kind==OpKind::kSum || kind==OpKind::kMax || kind==OpKind::kMean)shape.erase(shape.begin()+axis);
        }
        if(kind==OpKind::kMax && !reduce)return invalid("max reduction requires a non-empty axis");
        if(kind==OpKind::kRmsNorm || kind==OpKind::kLayerNorm) {
          const double eps=execution.op.epsilon;
          if(!std::isfinite(eps) || eps<0 || eps>std::numeric_limits<float>::max() ||
             (eps>0 && static_cast<float>(eps)==0))return invalid("epsilon must be finite and non-negative and representable as float32");
          epsilon=static_cast<float>(eps);
        }
      }
    }
    // Validate all metadata before touching CUDA or allocating output memory.
    ContextScope device;
    if (auto status = device.status(); !status.ok()) return status;
    auto buffer = CudaBuffer::create(DType::kFloat32, shape);
    if (!buffer) return buffer.status();
    auto* output = static_cast<float*>(buffer.value()->data());
    const auto count = static_cast<std::size_t>(numel(shape));
    auto data=[&](std::size_t index) {
      return static_cast<const float*>(std::static_pointer_cast<CudaBuffer>(execution.inputs[index].buffer)->data());
    };
    cudaError_t error;
    if (kind==OpKind::kFill) {
      error=launch_fill(output,count,static_cast<float>(execution.op.scalar_value));
    } else if(kind==OpKind::kAdd || kind==OpKind::kMultiply) {
      error=launch_binary(data(0),data(1),output,count,kind==OpKind::kMultiply);
    } else if(kind==OpKind::kMatmul) {
      error=launch_matmul(data(0),data(1),output,shape[0],shape[1],execution.inputs[0].shape[1]);
    } else if(kind==OpKind::kExp || kind==OpKind::kGelu || kind==OpKind::kSilu) {
      error=launch_unary(data(0),output,count,kind);
    } else {
      // Output allocation and validated input shapes bound the nonempty product.
      if (outer && inner>std::numeric_limits<std::size_t>::max()/outer)return invalid("CUDA axis size overflow");
      error=launch_axis(data(0),output,outer*inner,reduce,inner,kind,epsilon);
    }
    if (auto status = runtime_status(error, "CUDA kernel execution"); !status.ok()) return status;
    // Publish only on success; failed execution must leave result slots intact.
    execution.outputs[0] = to_core_tensor(CudaTensor(DType::kFloat32, std::move(shape), buffer.move_value()));
    return Status::Ok();
  } catch (const std::invalid_argument& error) {
    return invalid(error.what());
  } catch (const std::exception& error) {
    return {StatusCode::kInternal, error.what()};
  }
}
}  // namespace tensorcx::cuda
