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

__global__ void binary_f32(const float* lhs, const float* rhs, float* output,
                           std::size_t count, bool multiply) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    output[i] = multiply ? lhs[i] * rhs[i] : lhs[i] + rhs[i];
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
                          std::size_t count, bool multiply) {
  if (count == 0) return cudaSuccess;
  binary_f32<<<blocks(count), kThreads>>>(lhs, rhs, output, count, multiply);
  return finish_launch();
}
}  // namespace tensorcx::cuda
