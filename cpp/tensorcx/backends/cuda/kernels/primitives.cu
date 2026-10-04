#include "tensorcx/backends/cuda/cuda_kernels.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace tensorcx::cuda {
namespace {
constexpr unsigned threads = 256;
unsigned blocks(std::size_t n) {
  return static_cast<unsigned>(std::min<std::size_t>(1 + (n - 1) / threads, 65535));
}
cudaError_t finish() {
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? cudaDeviceSynchronize() : error;
}
__global__ void unary(const float* x, float* out, std::size_t n, OpKind op) {
  for (std::size_t i = std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
       i<n; i+=std::size_t(blockDim.x)*gridDim.x) {
    const float v=x[i];
    if (op==OpKind::kNegate) out[i]=-v;
    else if (op==OpKind::kExp) out[i]=expf(v);
    else if (op==OpKind::kSilu) out[i]=v/(1.0F+expf(-v));
    else {
      const float cubic=__fmul_rn(__fmul_rn(__fmul_rn(0.044715F,v),v),v);
      const float inner=__fmul_rn(0.7978845608028654F,__fadd_rn(v,cubic));
      out[i]=__fmul_rn(__fmul_rn(0.5F,v),__fadd_rn(1.0F,tanhf(inner)));
    }
  }
}
// One lane per independent slice preserves the CPU accumulation order. Keep
// this path for short rows and strided axes, where neighboring lanes coalesce.
__global__ void axis_kernel(const float* x,float* out,std::size_t groups,
    std::size_t reduce,std::size_t inner,OpKind op,float eps) {
  for (std::size_t group=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
       group<groups;group+=std::size_t(blockDim.x)*gridDim.x) {
    const std::size_t base=(group/inner)*reduce*inner+group%inner;
    if (op==OpKind::kSum || op==OpKind::kMean) {
      float sum=0;
      for(std::size_t r=0;r<reduce;++r) sum=__fadd_rn(sum,x[base+r*inner]);
      out[group]=op==OpKind::kSum?sum:(reduce?sum/static_cast<float>(reduce):nanf(""));
      continue;
    }
    if (op==OpKind::kMax || op==OpKind::kSoftmax) {
      float maximum=-CUDART_INF_F;
      for(std::size_t r=0;r<reduce;++r) {
        const float v=x[base+r*inner];
        if(isnan(v)){maximum=v;break;}
        if(maximum<v)maximum=v;
      }
      if(op==OpKind::kMax){out[group]=maximum;continue;}
      float denom=0;
      for(std::size_t r=0;r<reduce;++r) denom=__fadd_rn(denom,expf(x[base+r*inner]-maximum));
      for(std::size_t r=0;r<reduce;++r) out[base+r*inner]=expf(x[base+r*inner]-maximum)/denom;
      continue;
    }
    float sum=0,mean=0;
    bool equal=true;
    const float first=x[base]; // Normalizations with an empty axis never launch.
    for(std::size_t r=0;r<reduce;++r) {
      const float v=x[base+r*inner];
      sum=__fadd_rn(sum,op==OpKind::kRmsNorm?__fmul_rn(v,v):v);
      equal=equal && v==first;
    }
    if(op==OpKind::kLayerNorm) {
      if(equal && isfinite(first)) {
        for(std::size_t r=0;r<reduce;++r) out[base+r*inner]=eps==0?nanf(""):0.0F;
        continue;
      }
      mean=sum/static_cast<float>(reduce);sum=0;
      for(std::size_t r=0;r<reduce;++r) {
        const float d=x[base+r*inner]-mean;
        sum=__fadd_rn(sum,__fmul_rn(d,d));
      }
    }
    const float denom=__fadd_rn(sum/static_cast<float>(reduce),eps);
    const float scale=1.0F/sqrtf(denom);
    for(std::size_t r=0;r<reduce;++r) {
      const auto i=base+r*inner;
      out[i]=(op==OpKind::kLayerNorm && denom==0)?nanf(""):
        __fmul_rn(op==OpKind::kLayerNorm?x[i]-mean:x[i],scale);
    }
  }
}
enum class Transform { identity, square, centered_square, exponential };

// Stage coalesced loads and parallel transforms, then accumulate in the same
// left-to-right float32 order as the CPU. A tree sum would change cancellation,
// overflow and normalization behavior. Only lane zero's return value is used.
__device__ float staged_sum(const float* x, float* out, std::size_t n,
    float* tile, Transform transform, float center, bool& equal) {
  float sum = 0;
  for (std::size_t start = 0; start < n; start += threads) {
    const auto index = start + threadIdx.x;
    float value = 0;
    if (index < n) {
      value = x[index];
      equal = equal && value == center;
      if (transform == Transform::square) value = __fmul_rn(value, value);
      else if (transform == Transform::centered_square) {
        const float difference = value - center;
        value = __fmul_rn(difference, difference);
      } else if (transform == Transform::exponential) {
        value = expf(value - center);
        out[index] = value;
      }
    }
    tile[threadIdx.x] = value;
    __syncthreads();
    if (threadIdx.x == 0) {
      const auto count = n - start < threads ? n - start : threads;
      for (std::size_t j = 0; j < count; ++j) sum = __fadd_rn(sum, tile[j]);
    }
    __syncthreads(); // Do not replace a tile while lane zero still reads it.
  }
  return sum;
}

// One block per contiguous row. Long rows benefit from coalesced global
// memory access and parallel transcendental/output work without reassociation.
__global__ void staged_axis(const float* x, float* out, std::size_t groups,
    std::size_t reduce, OpKind op, float eps) {
  __shared__ float tile[threads], center, denominator;
  for (std::size_t group = blockIdx.x; group < groups; group += gridDim.x) {
    const float* row = x + group * reduce;
    float* result = out;
    if (op == OpKind::kSoftmax || op == OpKind::kRmsNorm || op == OpKind::kLayerNorm)
      result += group * reduce;
    bool equal = true;
    if (op == OpKind::kMax || op == OpKind::kSoftmax) {
      float maximum = -CUDART_INF_F;
      for (std::size_t start = 0; start < reduce; start += threads) {
        const auto index = start + threadIdx.x;
        tile[threadIdx.x] = index < reduce ? row[index] : -CUDART_INF_F;
        __syncthreads();
        if (threadIdx.x == 0 && !isnan(maximum)) {
          const auto count = reduce - start < threads ? reduce - start : threads;
          for (std::size_t j = 0; j < count; ++j) {
            const float value = tile[j];
            if (isnan(value)) { maximum = value; break; }
            if (maximum < value) maximum = value;
          }
        }
        __syncthreads();
      }
      if (threadIdx.x == 0) center = maximum;
      __syncthreads();
      if (op == OpKind::kMax) {
        if (threadIdx.x == 0) out[group] = center;
      } else {
        const float sum = staged_sum(row, result, reduce, tile, Transform::exponential, center, equal);
        if (threadIdx.x == 0) denominator = sum;
        __syncthreads();
        for (std::size_t r = threadIdx.x; r < reduce; r += threads) result[r] /= denominator;
      }
    } else {
      const float first = row[0];
      const auto transform = op == OpKind::kRmsNorm ? Transform::square : Transform::identity;
      float sum = staged_sum(row, result, reduce, tile, transform, first, equal);
      if (op == OpKind::kSum || op == OpKind::kMean) {
        if (threadIdx.x == 0) out[group] = op == OpKind::kSum ? sum : sum / static_cast<float>(reduce);
      } else {
        const bool constant = __syncthreads_and(equal) && isfinite(first);
        if (op == OpKind::kLayerNorm && constant) {
          for (std::size_t r = threadIdx.x; r < reduce; r += threads) result[r] = eps == 0 ? nanf("") : 0.0F;
        } else {
          if (threadIdx.x == 0) center = op == OpKind::kLayerNorm ? sum / static_cast<float>(reduce) : 0;
          __syncthreads();
          if (op == OpKind::kLayerNorm)
            sum = staged_sum(row, result, reduce, tile, Transform::centered_square, center, equal);
          if (threadIdx.x == 0) denominator = __fadd_rn(sum / static_cast<float>(reduce), eps);
          __syncthreads();
          const float scale = 1.0F / sqrtf(denominator);
          for (std::size_t r = threadIdx.x; r < reduce; r += threads)
            result[r] = (op == OpKind::kLayerNorm && denominator == 0) ? nanf("") :
              __fmul_rn(op == OpKind::kLayerNorm ? row[r] - center : row[r], scale);
        }
      }
    }
    __syncthreads(); // All lanes must finish before reusing shared row state.
  }
}
__device__ std::size_t reduction_offset(std::size_t index, const Dim* metadata,
                                       std::size_t begin, std::size_t end) {
  std::size_t offset = 0;
  for (auto axis = end; axis-- > begin;) {
    const auto extent = static_cast<std::size_t>(metadata[2 * axis]);
    offset += (index % extent) * static_cast<std::size_t>(metadata[2 * axis + 1]);
    index /= extent;
  }
  return offset;
}

__global__ void reduce_axes(const float* input, float* output, std::size_t groups,
                            std::size_t reduce, const Dim* metadata,
                            std::size_t output_rank, std::size_t rank, OpKind op) {
  for (std::size_t group = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       group < groups; group += std::size_t(blockDim.x) * gridDim.x) {
    const auto base = reduction_offset(group, metadata, 0, output_rank);
    float value = op == OpKind::kMax ? -CUDART_INF_F : 0.0F;
    for (std::size_t r = 0; r < reduce; ++r) {
      const float item = input[base + reduction_offset(r, metadata, output_rank, rank)];
      if (op == OpKind::kMax) {
        if (isnan(item)) { value = item; break; }
        if (value < item) value = item;
      } else value = __fadd_rn(value, item);
    }
    if (op == OpKind::kMean) value = reduce ? value / static_cast<float>(reduce) : nanf("");
    output[group] = value;
  }
}

__global__ void matmul(const float* a,const float* b,float* out,
    std::size_t m,std::size_t n,std::size_t k,std::size_t tile_count) {
  __shared__ float left[16][16],right[16][16];
  const unsigned x=threadIdx.x,y=threadIdx.y;
  const std::size_t columns=(n+15)/16;
  for(std::size_t tile=blockIdx.x;tile<tile_count;tile+=gridDim.x) {
    const std::size_t row=(tile/columns)*16+y,col=(tile%columns)*16+x;
    float sum=0;
    for(std::size_t base=0;base<k;base+=16) {
      left[y][x]=(row<m && base+x<k)?a[row*k+base+x]:0;
      right[y][x]=(base+y<k && col<n)?b[(base+y)*n+col]:0;
      __syncthreads();
      for(unsigned j=0;j<16 && base+j<k;++j) sum=__fadd_rn(sum,__fmul_rn(left[y][j],right[j][x]));
      __syncthreads();
    }
    if(row<m && col<n)out[row*n+col]=sum;
    __syncthreads();
  }
}
}
cudaError_t launch_unary(const float* x,float* out,std::size_t n,OpKind op) {
  if(!n)return cudaSuccess;
  unary<<<blocks(n),threads>>>(x,out,n,op);return finish();
}
cudaError_t launch_axis(const float* x,float* out,std::size_t groups,std::size_t reduce,
    std::size_t inner,OpKind op,float eps) {
  if(!groups || (!reduce && (op==OpKind::kSoftmax || op==OpKind::kRmsNorm || op==OpKind::kLayerNorm)))return cudaSuccess;
  if (inner == 1 && reduce >= threads) {
    staged_axis<<<static_cast<unsigned>(std::min<std::size_t>(groups,65535)),threads>>>(x,out,groups,reduce,op,eps);
    return finish();
  }
  axis_kernel<<<blocks(groups),threads>>>(x,out,groups,reduce,inner,op,eps);return finish();
}
cudaError_t launch_reduce_axes(const float* input, float* output, std::size_t groups,
                               std::size_t reduce, const Dim* metadata,
                               std::size_t output_rank, std::size_t rank, OpKind op) {
  if (!groups) return cudaSuccess;
  reduce_axes<<<blocks(groups), threads>>>(input, output, groups, reduce, metadata, output_rank, rank, op);
  return finish();
}

cudaError_t launch_matmul(const float* a,const float* b,float* out,std::size_t m,std::size_t n,std::size_t k) {
  if(!m || !n)return cudaSuccess;
  const auto tiles=((m+15)/16)*((n+15)/16);
  matmul<<<static_cast<unsigned>(std::min<std::size_t>(tiles,65535)),dim3(16,16)>>>(a,b,out,m,n,k,tiles);
  return finish();
}
} // namespace tensorcx::cuda
