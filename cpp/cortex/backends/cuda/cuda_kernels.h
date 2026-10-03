#pragma once

#include <cstddef>
#include "cortex/core/operation.h"
#include <cuda_runtime_api.h>

namespace cortex::cuda {

// Called with device 0 current. Returns only after the kernel has completed.
cudaError_t launch_fill(float* output, std::size_t count, float value);
cudaError_t launch_binary(const float* lhs, const float* rhs, float* output,
                          std::size_t count, bool multiply);

cudaError_t launch_unary(const float* input, float* output, std::size_t count, OpKind op);
cudaError_t launch_axis(const float* input, float* output, std::size_t groups,
                        std::size_t reduce, std::size_t inner, OpKind op, float epsilon);
cudaError_t launch_matmul(const float* lhs, const float* rhs, float* output,
                          std::size_t m, std::size_t n, std::size_t k);

}  // namespace cortex::cuda
