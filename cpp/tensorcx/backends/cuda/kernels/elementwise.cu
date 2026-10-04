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
cudaError_t launch_scalar(const float* input, float* output, std::size_t count,
                          OpKind op, float scalar, bool scalar_left) {
  if (count == 0) return cudaSuccess;
  scalar_f32<<<blocks(count), kThreads>>>(input, output, count, op, scalar, scalar_left);
  return finish_launch();
}
}  // namespace tensorcx::cuda
