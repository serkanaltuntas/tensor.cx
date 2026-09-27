#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

namespace cortex::cuda {

// Called with device 0 current. Returns only after the kernel has completed.
cudaError_t launch_fill(float* output, std::size_t count, float value);
cudaError_t launch_binary(const float* lhs, const float* rhs, float* output,
                          std::size_t count, bool multiply);

}  // namespace cortex::cuda
