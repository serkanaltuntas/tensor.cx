#include "cortex/backends/cuda/cuda_kernels.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace cortex::cuda {
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
    if (op==OpKind::kExp) out[i]=expf(v);
    else if (op==OpKind::kSilu) out[i]=v/(1.0F+expf(-v));
    else {
      const float cubic=__fmul_rn(__fmul_rn(__fmul_rn(0.044715F,v),v),v);
      const float inner=__fmul_rn(0.7978845608028654F,__fadd_rn(v,cubic));
      out[i]=__fmul_rn(__fmul_rn(0.5F,v),__fadd_rn(1.0F,tanhf(inner)));
    }
  }
}
// One lane per independent slice preserves the CPU accumulation order. This
// supports arbitrary axes; parallel reduction of long slices is a later tuning step.
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
  axis_kernel<<<blocks(groups),threads>>>(x,out,groups,reduce,inner,op,eps);return finish();
}
cudaError_t launch_matmul(const float* a,const float* b,float* out,std::size_t m,std::size_t n,std::size_t k) {
  if(!m || !n)return cudaSuccess;
  const auto tiles=((m+15)/16)*((n+15)/16);
  matmul<<<static_cast<unsigned>(std::min<std::size_t>(tiles,65535)),dim3(16,16)>>>(a,b,out,m,n,k,tiles);
  return finish();
}
} // namespace cortex::cuda
