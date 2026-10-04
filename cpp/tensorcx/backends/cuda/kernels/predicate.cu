#include "tensorcx/backends/cuda/cuda_predicate.h"
#include <algorithm>
#include <cuda_runtime.h>

namespace tensorcx::cuda {
namespace {
constexpr unsigned kThreads = 256;
unsigned blocks(std::size_t count) { return static_cast<unsigned>(std::min<std::size_t>((count - 1) / kThreads + 1, 65535)); }
cudaError_t finish() {
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? cudaDeviceSynchronize() : error;
}
__device__ Dim mapped(Dim i, const Dim* metadata, std::size_t rank, std::size_t arg) {
  Dim result = 0;
  for (std::size_t axis = rank; axis-- > 0;) {
    result += (i % metadata[axis]) * metadata[(arg + 1) * rank + axis];
    i /= metadata[axis];
  }
  return result;
}
template<typename T>
__device__ bool comparison(T a, T b, OpKind kind) {
  switch (kind) {
    case OpKind::kEqual: return a == b;
    case OpKind::kNotEqual: return a != b;
    case OpKind::kLess: return a < b;
    case OpKind::kLessEqual: return a <= b;
    case OpKind::kGreater: return a > b;
    case OpKind::kGreaterEqual: return a >= b;
    default: return false;
  }
}
template<typename T>
__global__ void predicate_kernel(const void* a, const void* b, const void* c, void* output,
                                 std::size_t count, const Dim* metadata, std::size_t rank, OpKind kind) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    const auto ia = mapped(i, metadata, rank, 0);
    if (kind == OpKind::kWhere) {
      const bool chosen = static_cast<const std::uint8_t*>(a)[ia] != 0;
      const auto source = mapped(i, metadata, rank, chosen ? 1 : 2);
      // Select stored words, without floating arithmetic or NaN conversion.
      if constexpr (sizeof(T) == 1)
        static_cast<std::uint8_t*>(output)[i] = static_cast<const std::uint8_t*>(chosen ? b : c)[source];
      else static_cast<std::uint32_t*>(output)[i] = static_cast<const std::uint32_t*>(chosen ? b : c)[source];
      continue;
    }
    auto* out = static_cast<std::uint8_t*>(output);
    if (kind == OpKind::kLogicalNot) { out[i] = !static_cast<const std::uint8_t*>(a)[ia]; continue; }
    const auto ib = mapped(i, metadata, rank, 1);
    if (kind == OpKind::kLogicalAnd || kind == OpKind::kLogicalOr || kind == OpKind::kLogicalXor) {
      const bool left = static_cast<const std::uint8_t*>(a)[ia] != 0;
      const bool right = static_cast<const std::uint8_t*>(b)[ib] != 0;
      out[i] = kind == OpKind::kLogicalAnd ? left && right : kind == OpKind::kLogicalOr ? left || right : left != right;
    } else if constexpr (sizeof(T) == 1) {
      out[i] = comparison(static_cast<const T*>(a)[ia] != 0, static_cast<const T*>(b)[ib] != 0, kind);
    } else out[i] = comparison(static_cast<const T*>(a)[ia], static_cast<const T*>(b)[ib], kind);
  }
}
__global__ void bool_reduce(const std::uint8_t* input, std::uint8_t* output, std::size_t count,
                            Dim reduce, const Dim* metadata, std::size_t rank, bool all) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    bool value = all;
    for (Dim j = 0; j < reduce; ++j) {
      Dim remaining = static_cast<Dim>(i) * reduce + j, source = 0;
      for (std::size_t axis = rank; axis-- > 0;) {
        source += remaining % metadata[2 * axis] * metadata[2 * axis + 1];
        remaining /= metadata[2 * axis];
      }
      value = all ? value && input[source] != 0 : value || input[source] != 0;
    }
    output[i] = value;
  }
}
__global__ void bool_cast(const void* input, void* output, std::size_t count, DType source, DType target) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    const bool value = source == DType::kFloat32 ? static_cast<const float*>(input)[i] != 0 :
                       source == DType::kInt32 ? static_cast<const std::int32_t*>(input)[i] != 0 :
                       static_cast<const std::uint8_t*>(input)[i] != 0;
    if (target == DType::kBool) static_cast<std::uint8_t*>(output)[i] = value;
    else if (target == DType::kFloat32) static_cast<float*>(output)[i] = value;
    else static_cast<std::int32_t*>(output)[i] = value;
  }
}
__global__ void mask_prefix(const std::uint8_t* mask, Dim* output, std::size_t count) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) output[i] = mask[i] != 0;
}
__global__ void prefix_step(const Dim* input, Dim* output, std::size_t count, Dim step) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x)
    output[i] = input[i] + (i >= static_cast<std::size_t>(step) ? input[i - step] : 0);
}
template<typename T>
__global__ void mask_copy(const T* input, const std::uint8_t* mask, const Dim* prefix,
                         T* output, std::size_t count, Dim block) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    const auto m = i / block;
    if (mask[m]) output[(prefix[m] - 1) * block + i % block] = input[i];
  }
}
}

cudaError_t launch_predicate(const void* a, const void* b, const void* c, void* output,
                            std::size_t count, const Dim* metadata, std::size_t rank,
                            OpKind kind, DType dtype) {
  if (!count) return cudaSuccess;
  if (dtype == DType::kFloat32) predicate_kernel<float><<<blocks(count), kThreads>>>(a,b,c,output,count,metadata,rank,kind);
  else if (dtype == DType::kInt32) predicate_kernel<std::int32_t><<<blocks(count), kThreads>>>(a,b,c,output,count,metadata,rank,kind);
  else predicate_kernel<std::uint8_t><<<blocks(count), kThreads>>>(a,b,c,output,count,metadata,rank,kind);
  return finish();
}
cudaError_t launch_bool_reduce(const std::uint8_t* input, std::uint8_t* output,
                              std::size_t count, Dim reduce, const Dim* metadata, std::size_t rank, bool all) {
  if (!count) return cudaSuccess;
  bool_reduce<<<blocks(count), kThreads>>>(input,output,count,reduce,metadata,rank,all);
  return finish();
}
cudaError_t launch_bool_cast(const void* input, void* output, std::size_t count, DType source, DType target) {
  if (!count) return cudaSuccess;
  bool_cast<<<blocks(count), kThreads>>>(input,output,count,source,target);
  return finish();
}
cudaError_t launch_mask_prefix(const std::uint8_t* mask, Dim* output, std::size_t count) {
  if (!count) return cudaSuccess;
  mask_prefix<<<blocks(count), kThreads>>>(mask,output,count);
  return finish();
}
cudaError_t launch_prefix_step(const Dim* input, Dim* output, std::size_t count, Dim step) {
  if (!count) return cudaSuccess;
  prefix_step<<<blocks(count), kThreads>>>(input,output,count,step);
  return finish();
}
cudaError_t launch_mask_copy(const void* input, const std::uint8_t* mask, const Dim* prefix,
                            void* output, std::size_t count, Dim block, DType dtype) {
  if (!count) return cudaSuccess;
  if (dtype == DType::kBool)
    mask_copy<<<blocks(count), kThreads>>>(static_cast<const std::uint8_t*>(input), mask, prefix,
                                         static_cast<std::uint8_t*>(output), count, block);
  else mask_copy<<<blocks(count), kThreads>>>(static_cast<const std::uint32_t*>(input), mask, prefix,
                                             static_cast<std::uint32_t*>(output), count, block);
  return finish();
}
}  // namespace tensorcx::cuda
