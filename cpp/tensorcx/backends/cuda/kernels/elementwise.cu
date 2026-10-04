#include "tensorcx/backends/cuda/cuda_kernels.h"

#include <algorithm>
#include <cuda_runtime.h>

namespace tensorcx::cuda {
namespace {
constexpr unsigned kThreads = 256;

__global__ void fill_f32(float* output, std::size_t count, float value) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    output[i] = value;
  }
}

__device__ float arithmetic(float lhs, float rhs, OpKind op) {
  if (op == OpKind::kAdd || op == OpKind::kAddScalar) return __fadd_rn(lhs, rhs);
  if (op == OpKind::kSubtract || op == OpKind::kSubtractScalar) return __fsub_rn(lhs, rhs);
  if (op == OpKind::kMultiply || op == OpKind::kMultiplyScalar) return __fmul_rn(lhs, rhs);
  return __fdiv_rn(lhs, rhs);
}

__global__ void binary_f32(const float* lhs, const float* rhs, float* output,
                           std::size_t count, OpKind op) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    output[i] = arithmetic(lhs[i], rhs[i], op);
  }
}

__global__ void broadcast_binary_f32(const float* lhs, const float* rhs, float* output,
                                     std::size_t count, OpKind op,
                                     const Dim* metadata, std::size_t rank) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    std::size_t remaining = i, left_offset = 0, right_offset = 0;
    for (std::size_t axis = rank; axis-- > 0;) {
      const auto dim = static_cast<std::size_t>(metadata[axis]);
      const auto coordinate = remaining % dim;
      remaining /= dim;
      left_offset += coordinate * static_cast<std::size_t>(metadata[rank + axis]);
      right_offset += coordinate * static_cast<std::size_t>(metadata[2 * rank + axis]);
    }
    output[i] = arithmetic(lhs[left_offset], rhs[right_offset], op);
  }
}

__global__ void transpose_bits(const std::uint32_t* input, std::uint32_t* output,
                               std::size_t count, const Dim* metadata, std::size_t rank) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    std::size_t remaining = i, source = 0;
    for (std::size_t axis = rank; axis-- > 0;) {
      const auto extent = static_cast<std::size_t>(metadata[axis]);
      source += (remaining % extent) * static_cast<std::size_t>(metadata[rank + axis]);
      remaining /= extent;
    }
    output[i] = input[source];
  }
}

__global__ void validate_int32_cast(const float* input, std::size_t count, int* invalid) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    const float value = input[i];
    // Negated comparisons reject NaNs as well as both infinities. The upper
    // bound is exclusive because INT32_MAX itself rounds to 2**31 in float32.
    if (!(value >= -2147483648.0F && value < 2147483648.0F)) atomicExch(invalid, 1);
  }
}

__global__ void cast_f32_i32(const float* input, std::int32_t* output, std::size_t count) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    output[i] = __float2int_rz(input[i]);
  }
}

__global__ void cast_i32_f32(const std::int32_t* input, float* output, std::size_t count) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    output[i] = __int2float_rn(input[i]);
  }
}

__global__ void scalar_f32(const float* input, float* output, std::size_t count,
                           OpKind op, float scalar, bool scalar_left) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    const float value = input[i];
    output[i] = scalar_left ? arithmetic(scalar, value, op) : arithmetic(value, scalar, op);
  }
}

unsigned blocks(std::size_t count) {
  return static_cast<unsigned>(std::min<std::size_t>(1 + (count - 1) / kThreads, 65535));
}

cudaError_t finish_launch() {
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? cudaDeviceSynchronize() : error;
}
}  // namespace

cudaError_t launch_fill(float* output, std::size_t count, float value) {
  if (count == 0) return cudaSuccess;
  fill_f32<<<blocks(count), kThreads>>>(output, count, value);
  return finish_launch();
}

cudaError_t launch_binary(const float* lhs, const float* rhs, float* output,
                          std::size_t count, OpKind op) {
  if (count == 0) return cudaSuccess;
  binary_f32<<<blocks(count), kThreads>>>(lhs, rhs, output, count, op);
  return finish_launch();
}
cudaError_t launch_broadcast_binary(const float* lhs, const float* rhs, float* output,
                                    std::size_t count, OpKind op,
                                    const Dim* metadata, std::size_t rank) {
  if (count == 0) return cudaSuccess;
  broadcast_binary_f32<<<blocks(count), kThreads>>>(lhs, rhs, output, count, op, metadata, rank);
  return finish_launch();
}

cudaError_t launch_transpose(const void* input, void* output, std::size_t count,
                             const Dim* metadata, std::size_t rank) {
  if (count == 0) return cudaSuccess;
  transpose_bits<<<blocks(count), kThreads>>>(static_cast<const std::uint32_t*>(input),
      static_cast<std::uint32_t*>(output), count, metadata, rank);
  return finish_launch();
}

cudaError_t launch_validate_int32_cast(const float* input, std::size_t count, int* invalid) {
  if (count == 0) return cudaSuccess;
  validate_int32_cast<<<blocks(count), kThreads>>>(input, count, invalid);
  return finish_launch();
}

cudaError_t launch_cast(const void* input, void* output, std::size_t count, DType input_dtype) {
  if (count == 0) return cudaSuccess;
  if (input_dtype == DType::kFloat32) {
    cast_f32_i32<<<blocks(count), kThreads>>>(static_cast<const float*>(input),
                                            static_cast<std::int32_t*>(output), count);
  } else {
    cast_i32_f32<<<blocks(count), kThreads>>>(static_cast<const std::int32_t*>(input),
                                            static_cast<float*>(output), count);
  }
  return finish_launch();
}
cudaError_t launch_scalar(const float* input, float* output, std::size_t count,
                          OpKind op, float scalar, bool scalar_left) {
  if (count == 0) return cudaSuccess;
  scalar_f32<<<blocks(count), kThreads>>>(input, output, count, op, scalar, scalar_left);
  return finish_launch();
}
}  // namespace tensorcx::cuda
