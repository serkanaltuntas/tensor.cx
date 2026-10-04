#pragma once

#include <cstddef>
#include "tensorcx/core/dtype.h"
#include "tensorcx/core/operation.h"
#include "tensorcx/core/shape.h"
#include <cuda_runtime_api.h>

namespace tensorcx::cuda {

// Called with device 0 current. Returns only after the kernel has completed.
cudaError_t launch_fill(float* output, std::size_t count, float value);
cudaError_t launch_binary(const float* lhs, const float* rhs, float* output,
                          std::size_t count, OpKind op);
// Metadata contains output dimensions, then lhs/rhs element strides; expanded
// and padded dimensions have stride zero. Metadata is already on the device.
cudaError_t launch_broadcast_binary(const float* lhs, const float* rhs, float* output,
                                    std::size_t count, OpKind op,
                                    const Dim* metadata, std::size_t rank);
cudaError_t launch_validate_int32_cast(const float* input, std::size_t count, int* invalid);
cudaError_t launch_cast(const void* input, void* output, std::size_t count, DType input_dtype);
// Metadata holds output dimensions followed by mapped input element strides.
// Both supported dtypes are copied as 32-bit words, preserving their bits.
cudaError_t launch_transpose(const void* input, void* output, std::size_t count,
                             const Dim* metadata, std::size_t rank, Dim offset);
cudaError_t launch_concat(const void* input, void* output, std::size_t count,
                          Dim block, Dim output_block, Dim offset);
cudaError_t launch_scalar(const float* input, float* output, std::size_t count,
                          OpKind op, float scalar, bool scalar_left);

cudaError_t launch_unary(const float* input, float* output, std::size_t count, OpKind op);
cudaError_t launch_axis(const float* input, float* output, std::size_t groups,
                        std::size_t reduce, std::size_t inner, OpKind op, float epsilon);
cudaError_t launch_reduce_axes(const float* input, float* output, std::size_t groups,
                               std::size_t reduce, const Dim* metadata,
                               std::size_t output_rank, std::size_t rank, OpKind op);
cudaError_t launch_matmul(const float* lhs, const float* rhs, float* output,
                          std::size_t m, std::size_t n, std::size_t k);

}  // namespace tensorcx::cuda
