#pragma once

#include <cuda_runtime_api.h>
#include "tensorcx/core/profiling.h"

namespace tensorcx::cuda {

inline cudaError_t profiled_malloc(void** pointer, std::size_t bytes) {
  ProfileScope scope(ProfileCategory::kAllocation, "cudaMalloc", "cuda", bytes);
  return cudaMalloc(pointer, bytes);
}
inline cudaError_t profiled_free(void* pointer) {
  ProfileScope scope(ProfileCategory::kDeallocation, "cudaFree", "cuda");
  return cudaFree(pointer);
}
inline cudaError_t profiled_memcpy(void* target, const void* source, std::size_t bytes, cudaMemcpyKind kind) {
  const char* name = kind == cudaMemcpyHostToDevice ? "host_to_device" :
                     kind == cudaMemcpyDeviceToHost ? "device_to_host" : "device_to_device";
  ProfileScope scope(ProfileCategory::kCopy, name, "cuda", bytes);
  return cudaMemcpy(target, source, bytes, kind);
}
inline cudaError_t profiled_memset(void* target, int value, std::size_t bytes) {
  ProfileScope scope(ProfileCategory::kMemset, "cudaMemset", "cuda", bytes);
  return cudaMemset(target, value, bytes);
}
inline cudaError_t profiled_synchronize() {
  ProfileScope scope(ProfileCategory::kSynchronize, "cudaDeviceSynchronize", "cuda");
  return cudaDeviceSynchronize();
}

}  // namespace tensorcx::cuda
