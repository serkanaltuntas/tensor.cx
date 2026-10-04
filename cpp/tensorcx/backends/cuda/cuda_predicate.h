#pragma once
#include "tensorcx/core/backend.h"
#include "tensorcx/backends/cuda/cuda_kernels.h"

namespace tensorcx::cuda {
Status execute_predicate(const BackendExecution& execution);
cudaError_t launch_predicate(const void* a, const void* b, const void* c, void* output,
                            std::size_t count, const Dim* metadata, std::size_t rank,
                            OpKind kind, DType value_dtype);
cudaError_t launch_bool_reduce(const std::uint8_t* input, std::uint8_t* output,
                              std::size_t count, Dim reduce, const Dim* metadata,
                              std::size_t rank, bool all);
cudaError_t launch_bool_cast(const void* input, void* output, std::size_t count,
                            DType source, DType target);
cudaError_t launch_mask_prefix(const std::uint8_t* mask, Dim* output, std::size_t count);
cudaError_t launch_prefix_step(const Dim* input, Dim* output, std::size_t count, Dim step);
cudaError_t launch_mask_copy(const void* input, const std::uint8_t* mask, const Dim* prefix,
                            void* output, std::size_t count, Dim block, DType dtype);
}
