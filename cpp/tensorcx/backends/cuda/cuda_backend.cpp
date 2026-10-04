#include "tensorcx/backends/cuda/cuda_backend.h"

#include <cuda_runtime_api.h>

#include <limits>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "tensorcx/backends/cuda/cuda_kernels.h"
#include "tensorcx/backends/cuda/cuda_predicate.h"
#include "tensorcx/core/predicate.h"
#include "tensorcx/backends/cuda/cuda_buffer.h"
#include "tensorcx/backends/cuda/cuda_kernel.h"

namespace tensorcx::cuda {
namespace {
Status invalid(std::string message) {
  return {StatusCode::kInvalidArgument, std::move(message)};
}
bool is_binary_arithmetic(OpKind kind) {
  return kind == OpKind::kAdd || kind == OpKind::kSubtract ||
         kind == OpKind::kMultiply || kind == OpKind::kDivide;
}
bool is_scalar_arithmetic(OpKind kind) {
  return kind == OpKind::kAddScalar || kind == OpKind::kSubtractScalar ||
         kind == OpKind::kMultiplyScalar || kind == OpKind::kDivideScalar;
}
bool is_elementwise_unary(OpKind kind) {
  return kind == OpKind::kExp || kind == OpKind::kGelu ||
         kind == OpKind::kSilu || kind == OpKind::kNegate;
}
Expected<std::size_t> byte_size(DType dtype, const Shape& shape) {
  if (dtype != DType::kFloat32 && dtype != DType::kInt32 && dtype != DType::kBool) return invalid("unsupported CUDA dtype");
  const auto count = static_cast<std::uint64_t>(numel(shape));
  if (count > std::numeric_limits<std::size_t>::max() / dtype_size(dtype)) {
    return invalid("CUDA buffer size overflow");
  }
  return static_cast<std::size_t>(count) * dtype_size(dtype);
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
      const void* source = tensor.data();
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
      void* destination = result.mutable_data();
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
    if (is_predicate_elementwise(kind) || kind == OpKind::kAny || kind == OpKind::kAll || kind == OpKind::kMaskedSelect)
      return execute_predicate(execution);
    if (kind == OpKind::kFill && execution.outputs[0].dtype == DType::kBool) {
      const auto shape = execution.outputs[0].shape;
      ContextScope device;
      if (!device.ready()) return device.status();
      auto buffer = CudaBuffer::create(DType::kBool, shape);
      if (!buffer) return buffer.status();
      if (buffer.value()->nbytes()) {
        auto status = runtime_status(cudaMemset(buffer.value()->data(), execution.op.scalar_value != 0,
                                               buffer.value()->nbytes()), "CUDA bool fill");
        if (!status.ok()) return status;
        status = runtime_status(cudaDeviceSynchronize(), "CUDA bool fill synchronize");
        if (!status.ok()) return status;
      }
      execution.outputs[0] = to_core_tensor(CudaTensor(DType::kBool, shape, buffer.move_value()));
      return Status::Ok();
    }
    if (kind == OpKind::kTranspose || kind == OpKind::kSlice) {
      const auto& input = execution.inputs[0];
      const auto plan = kind == OpKind::kTranspose
          ? make_transpose_plan(input.shape, execution.op.axes)
          : make_slice_plan(input.shape, execution.op.slice_starts,
                            execution.op.slice_steps, execution.op.slice_shape);
      const auto rank = plan.output_shape.size();
      if (rank > std::numeric_limits<std::size_t>::max() / (2 * sizeof(Dim)))
        return invalid("CUDA transpose metadata size overflow");
      ContextScope device;
      if (!device.ready()) return device.status();
      const auto source = std::static_pointer_cast<CudaBuffer>(input.buffer);
      auto result = CudaBuffer::create(input.dtype, plan.output_shape);
      if (!result) return result.status();
      const auto count = static_cast<std::size_t>(numel(plan.output_shape));
      if (count) {
        std::unique_ptr<void, DeviceDeleter> device_metadata(nullptr, DeviceDeleter{device.owner()});
        if (rank) {
          Shape metadata = plan.output_shape;
          metadata.insert(metadata.end(), plan.input_strides.begin(), plan.input_strides.end());
          void* allocation = nullptr;
          auto status = runtime_status(cudaMalloc(&allocation, metadata.size() * sizeof(Dim)),
                                       "CUDA transpose metadata allocation");
          if (!status.ok()) return status;
          device_metadata.reset(allocation);
          status = runtime_status(cudaMemcpy(allocation, metadata.data(), metadata.size() * sizeof(Dim),
                                             cudaMemcpyHostToDevice), "CUDA transpose metadata copy");
          if (!status.ok()) return status;
        }
        // Rank-zero kernels copy one word without reading the null metadata.
        const auto status = runtime_status(launch_transpose(source->data(), result.value()->data(), count,
            static_cast<const Dim*>(device_metadata.get()), rank, plan.offset, input.dtype), "CUDA index copy");
        if (!status.ok()) return status;
      }
      execution.outputs[0] = to_core_tensor(CudaTensor(input.dtype, plan.output_shape, result.move_value()));
      return Status::Ok();
    }
    if (kind == OpKind::kConcat) {
      const auto dtype = execution.inputs[0].dtype;
      std::vector<Shape> shapes;
      for (const auto& input : execution.inputs) {
        if (input.dtype != dtype) return invalid("concat dtypes must match");
        shapes.push_back(input.shape);
      }
      const auto plan = make_concat_plan(shapes, execution.op.axis);
      ContextScope device;
      if (!device.ready()) return device.status();
      auto result = CudaBuffer::create(dtype, plan.output_shape);
      if (!result) return result.status();
      if (numel(plan.output_shape)) {
        const auto output_block = plan.output_shape[plan.axis] * plan.inner;
        Dim offset = 0;
        for (const auto& input : execution.inputs) {
          const auto block = input.shape[plan.axis] * plan.inner;
          const auto source = std::static_pointer_cast<CudaBuffer>(input.buffer);
          auto status = runtime_status(launch_concat(source->data(), result.value()->data(),
              static_cast<std::size_t>(numel(input.shape)), block, output_block, offset, dtype), "CUDA concat");
          if (!status.ok()) return status;
          offset += block;
        }
      }
      execution.outputs[0] = to_core_tensor(CudaTensor(dtype, plan.output_shape, result.move_value()));
      return Status::Ok();
    }
    if (kind == OpKind::kCast) {
      const auto& input = execution.inputs[0];
      const auto target = execution.op.target_dtype;
      if (target != DType::kFloat32 && target != DType::kInt32 && target != DType::kBool)
        return invalid("unsupported cast target dtype");
      ContextScope device;
      if (!device.ready()) return device.status();
      const auto source = std::static_pointer_cast<CudaBuffer>(input.buffer);
      const auto count = static_cast<std::size_t>(numel(input.shape));
      if (input.dtype == DType::kFloat32 && target == DType::kInt32 && count) {
        auto flag = CudaBuffer::create(DType::kInt32, {1});
        if (!flag) return flag.status();
        auto* invalid_flag = static_cast<int*>(flag.value()->data());
        auto status = runtime_status(cudaMemset(invalid_flag, 0, sizeof(int)), "CUDA cast validation flag");
        if (!status.ok()) return status;
        status = runtime_status(launch_validate_int32_cast(
            static_cast<const float*>(source->data()), count, invalid_flag), "CUDA cast validation");
        if (!status.ok()) return status;
        int invalid_value = 0;
        status = runtime_status(cudaMemcpy(&invalid_value, invalid_flag, sizeof(int), cudaMemcpyDeviceToHost),
                                "CUDA cast validation readback");
        if (!status.ok()) return status;
        if (invalid_value)
          return invalid("float32 to int32 cast requires finite values in [-2147483648, 2147483648)");
      }
      auto result = CudaBuffer::create(target, input.shape);
      if (!result) return result.status();
      if (count) {
        const auto error = input.dtype == target
            ? cudaMemcpy(result.value()->data(), source->data(), source->nbytes(), cudaMemcpyDeviceToDevice)
            : input.dtype == DType::kBool || target == DType::kBool
                ? launch_bool_cast(source->data(), result.value()->data(), count, input.dtype, target)
                : launch_cast(source->data(), result.value()->data(), count, input.dtype);
        auto status = runtime_status(error, "CUDA tensor cast");
        if (!status.ok()) return status;
        // Device-to-device copies can return before completion. Cast kernels
        // synchronize inside launch_cast; same-dtype copies have the same API.
        if (input.dtype == target) {
          status = runtime_status(cudaDeviceSynchronize(), "CUDA tensor cast synchronize");
          if (!status.ok()) return status;
        }
      }
      execution.outputs[0] = to_core_tensor(CudaTensor(target, input.shape, result.move_value()));
      return Status::Ok();
    }
    if (is_binary_arithmetic(kind) &&
        execution.inputs[0].dtype!=execution.inputs[1].dtype) return invalid("dtype mismatch for binary operation");
    for (const auto& input : execution.inputs) {
      if (input.dtype!=DType::kFloat32) return invalid("CUDA operations only support float32");
    }
    if ((kind == OpKind::kSum || kind == OpKind::kMax || kind == OpKind::kMean) &&
        execution.op.reduction_axes) {
      const auto& input = execution.inputs[0];
      const auto plan = make_reduction_plan(input.shape, *execution.op.reduction_axes);
      if (kind == OpKind::kMax && plan.reduction_size == 0)
        return invalid("max reduction requires non-empty axes");
      if (execution.op.reduction_axes->empty()) {
        auto copy = execution; copy.op.kind = OpKind::kCast; copy.op.target_dtype = input.dtype;
        return execute(copy);
      }
      if (plan.index_metadata.size() > std::numeric_limits<std::size_t>::max() / sizeof(Dim))
        return invalid("CUDA reduction metadata size overflow");
      ContextScope device;
      if (!device.ready()) return device.status();
      auto result = CudaBuffer::create(input.dtype, plan.output_shape);
      if (!result) return result.status();
      const auto count = static_cast<std::size_t>(numel(plan.output_shape));
      if (count) {
        void* allocation = nullptr;
        const auto bytes = plan.index_metadata.size() * sizeof(Dim);
        auto status = runtime_status(cudaMalloc(&allocation, bytes), "CUDA reduction metadata allocation");
        if (!status.ok()) return status;
        std::unique_ptr<void, DeviceDeleter> metadata(allocation, DeviceDeleter{device.owner()});
        status = runtime_status(cudaMemcpy(allocation, plan.index_metadata.data(), bytes, cudaMemcpyHostToDevice),
                                "CUDA reduction metadata copy");
        if (!status.ok()) return status;
        const auto source = std::static_pointer_cast<CudaBuffer>(input.buffer);
        status = runtime_status(launch_reduce_axes(static_cast<const float*>(source->data()),
            static_cast<float*>(result.value()->data()), count, static_cast<std::size_t>(plan.reduction_size),
            static_cast<const Dim*>(metadata.get()), plan.output_shape.size(), input.shape.size(), kind),
            "CUDA multi-axis reduction");
        if (!status.ok()) return status;
      }
      execution.outputs[0] = to_core_tensor(CudaTensor(input.dtype, plan.output_shape, result.move_value()));
      return Status::Ok();
    }
    std::optional<BroadcastPlan> broadcast;
    if (kind==OpKind::kFill) {
      if(execution.outputs[0].dtype!=DType::kFloat32) return invalid("CUDA fill only supports float32");
      shape=execution.outputs[0].shape;
    } else {
      shape=execution.inputs[0].shape;
      if (is_binary_arithmetic(kind)) {
        if (shape != execution.inputs[1].shape) {
          broadcast = make_broadcast_plan(shape, execution.inputs[1].shape);
          shape = broadcast->output_shape;
        }
      } else if(kind==OpKind::kMatmul) {
        const auto& right=execution.inputs[1].shape;
        if(shape.size()!=2 || right.size()!=2) return invalid("matmul requires rank-2 tensors");
        if(shape[1]!=right[0]) return invalid("matmul shape mismatch");
        if(execution.op.matmul_preference!=MatmulPreference::kAuto &&
           execution.op.matmul_preference!=MatmulPreference::kCustom) return invalid("CUDA matmul supports auto or custom");
        shape={shape[0],right[1]};
      } else if(!is_elementwise_unary(kind) && !is_scalar_arithmetic(kind)) {
        auto axis=execution.op.axis;
        const auto rank=static_cast<std::int64_t>(shape.size());
        if(!rank) {
          if(axis!=0 && axis!=-1) return invalid("reduction axis is out of range");
        } else {
          if(axis<0)axis+=rank;
          if(axis<0 || axis>=rank)return invalid("reduction axis is out of range");
          Shape before(shape.begin(),shape.begin()+axis);
          outer=static_cast<std::size_t>(numel(before));
          // Metadata validation already checked the contiguous suffix product.
          inner=static_cast<std::size_t>(execution.inputs[0].strides[axis]);
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
    } else if(is_binary_arithmetic(kind)) {
      if (broadcast && count) {
        // Transfer only O(rank) metadata. Inputs remain in their original
        // buffers; zero strides implement broadcast dimensions in the kernel.
        const auto rank = shape.size();
        if (rank > std::numeric_limits<std::size_t>::max() / (3 * sizeof(Dim)))
          return invalid("CUDA broadcast metadata size overflow");
        Shape metadata;
        metadata.reserve(3 * rank);
        metadata.insert(metadata.end(), shape.begin(), shape.end());
        metadata.insert(metadata.end(), broadcast->lhs_strides.begin(), broadcast->lhs_strides.end());
        metadata.insert(metadata.end(), broadcast->rhs_strides.begin(), broadcast->rhs_strides.end());
        void* allocation = nullptr;
        auto status = runtime_status(cudaMalloc(&allocation, metadata.size() * sizeof(Dim)),
                                     "CUDA broadcast metadata allocation");
        if (!status.ok()) return status;
        std::unique_ptr<void, DeviceDeleter> device_metadata(allocation, DeviceDeleter{device.owner()});
        status = runtime_status(cudaMemcpy(device_metadata.get(), metadata.data(),
                                           metadata.size() * sizeof(Dim), cudaMemcpyHostToDevice),
                                "CUDA broadcast metadata copy");
        if (!status.ok()) return status;
        error = launch_broadcast_binary(data(0), data(1), output, count, kind,
                                        static_cast<const Dim*>(device_metadata.get()), rank);
      } else {
        error=launch_binary(data(0),data(1),output,count,kind);
      }
    } else if(is_scalar_arithmetic(kind)) {
      error=launch_scalar(data(0),output,count,kind,
                          static_cast<float>(execution.op.scalar_value),execution.op.scalar_left);
    } else if(kind==OpKind::kMatmul) {
      error=launch_matmul(data(0),data(1),output,shape[0],shape[1],execution.inputs[0].shape[1]);
    } else if(is_elementwise_unary(kind)) {
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
