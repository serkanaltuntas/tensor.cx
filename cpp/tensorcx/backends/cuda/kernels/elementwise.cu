#include "tensorcx/backends/cuda/cuda_kernels.h"

#include <algorithm>
#include <cuda_runtime.h>

namespace tensorcx::cuda {
namespace {
constexpr unsigned kThreads = 256;

template<typename T>
__global__ void fill_numeric(T* output, std::size_t count, T value) {
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

// Read/store int32 through the corresponding unsigned type. Every operation
// is modulo 2^32, including INT_MIN negation, with no signed-overflow UB.
__device__ std::uint32_t arithmetic(std::uint32_t lhs, std::uint32_t rhs, OpKind op) {
  if (op == OpKind::kAdd || op == OpKind::kAddScalar) return lhs + rhs;
  if (op == OpKind::kSubtract || op == OpKind::kSubtractScalar) return lhs - rhs;
  return lhs * rhs;
}

template<typename T>
__global__ void binary_numeric(const T* lhs, const T* rhs, T* output,
                           std::size_t count, OpKind op) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    output[i] = arithmetic(lhs[i], rhs[i], op);
  }
}

template<typename T>
__global__ void broadcast_binary_numeric(const T* lhs, const T* rhs, T* output,
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

template<typename Word>
__global__ void transpose_bits(const Word* input, Word* output,
                               std::size_t count, const Dim* metadata, std::size_t rank, Dim offset) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    std::size_t remaining = i;
    Dim source = offset;
    for (std::size_t axis = rank; axis-- > 0;) {
      const auto extent = static_cast<std::size_t>(metadata[axis]);
      source += static_cast<Dim>(remaining % extent) * metadata[rank + axis];
      remaining /= extent;
    }
    output[i] = input[source];
  }
}

template<typename Word>
__global__ void concat_bits(const Word* input, Word* output,
                            std::size_t count, Dim block, Dim output_block, Dim offset) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x)
    output[(i / block) * output_block + offset + i % block] = input[i];
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

template<typename T>
__global__ void scalar_numeric(const T* input, T* output, std::size_t count,
                               OpKind op, T scalar, bool scalar_left) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    const T value = input[i];
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

cudaError_t launch_fill(void* output, std::size_t count, double value, DType dtype) {
  if (count == 0) return cudaSuccess;
  if (dtype == DType::kInt32)
    fill_numeric<<<blocks(count), kThreads>>>(static_cast<std::int32_t*>(output), count,
                                             static_cast<std::int32_t>(value));
  else
    fill_numeric<<<blocks(count), kThreads>>>(static_cast<float*>(output), count, static_cast<float>(value));
  return finish_launch();
}

cudaError_t launch_binary(const void* lhs, const void* rhs, void* output,
                          std::size_t count, OpKind op, DType dtype) {
  if (count == 0) return cudaSuccess;
  if (dtype == DType::kInt32)
    binary_numeric<<<blocks(count), kThreads>>>(static_cast<const std::uint32_t*>(lhs),
        static_cast<const std::uint32_t*>(rhs), static_cast<std::uint32_t*>(output), count, op);
  else
    binary_numeric<<<blocks(count), kThreads>>>(static_cast<const float*>(lhs),
        static_cast<const float*>(rhs), static_cast<float*>(output), count, op);
  return finish_launch();
}
cudaError_t launch_broadcast_binary(const void* lhs, const void* rhs, void* output,
                                    std::size_t count, OpKind op,
                                    const Dim* metadata, std::size_t rank, DType dtype) {
  if (count == 0) return cudaSuccess;
  if (dtype == DType::kInt32)
    broadcast_binary_numeric<<<blocks(count), kThreads>>>(static_cast<const std::uint32_t*>(lhs),
        static_cast<const std::uint32_t*>(rhs), static_cast<std::uint32_t*>(output), count, op, metadata, rank);
  else
    broadcast_binary_numeric<<<blocks(count), kThreads>>>(static_cast<const float*>(lhs),
        static_cast<const float*>(rhs), static_cast<float*>(output), count, op, metadata, rank);
  return finish_launch();
}

cudaError_t launch_transpose(const void* input, void* output, std::size_t count,
                             const Dim* metadata, std::size_t rank, Dim offset, DType dtype) {
  if (count == 0) return cudaSuccess;
  if (dtype == DType::kBool) {
    transpose_bits<<<blocks(count), kThreads>>>(static_cast<const std::uint8_t*>(input),
        static_cast<std::uint8_t*>(output), count, metadata, rank, offset);
    return finish_launch();
  }
  transpose_bits<<<blocks(count), kThreads>>>(static_cast<const std::uint32_t*>(input),
      static_cast<std::uint32_t*>(output), count, metadata, rank, offset);
  return finish_launch();
}

cudaError_t launch_concat(const void* input, void* output, std::size_t count,
                          Dim block, Dim output_block, Dim offset, DType dtype) {
  if (count == 0) return cudaSuccess;
  if (dtype == DType::kBool) {
    concat_bits<<<blocks(count), kThreads>>>(static_cast<const std::uint8_t*>(input),
        static_cast<std::uint8_t*>(output), count, block, output_block, offset);
    return finish_launch();
  }
  concat_bits<<<blocks(count), kThreads>>>(static_cast<const std::uint32_t*>(input),
      static_cast<std::uint32_t*>(output), count, block, output_block, offset);
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
cudaError_t launch_scalar(const void* input, void* output, std::size_t count,
                          OpKind op, double scalar, bool scalar_left, DType dtype) {
  if (count == 0) return cudaSuccess;
  if (dtype == DType::kInt32)
    scalar_numeric<<<blocks(count), kThreads>>>(static_cast<const std::uint32_t*>(input),
        static_cast<std::uint32_t*>(output), count, op,
        static_cast<std::uint32_t>(static_cast<std::int32_t>(scalar)), scalar_left);
  else
    scalar_numeric<<<blocks(count), kThreads>>>(static_cast<const float*>(input),
        static_cast<float*>(output), count, op, static_cast<float>(scalar), scalar_left);
  return finish_launch();
}
}  // namespace tensorcx::cuda
