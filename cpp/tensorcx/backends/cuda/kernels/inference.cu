#include "tensorcx/backends/cuda/cuda_inference.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>
namespace tensorcx::cuda {
namespace {
unsigned blocks(Dim n){return static_cast<unsigned>(std::min<Dim>((n-1)/256+1,65535));}
cudaError_t finish(){auto error=cudaGetLastError();return error==cudaSuccess?cudaDeviceSynchronize():error;}
__global__ void embedding_validate(const std::int32_t* indices,Dim count,Dim vocabulary,int* invalid) {
  for(Dim i=Dim(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=Dim(blockDim.x)*gridDim.x)
    if(indices[i]<0 || Dim(indices[i])>=vocabulary)atomicExch(invalid,1);
}
__global__ void embedding_gather(const std::int32_t* indices,const unsigned* weight,unsigned* output,Dim count,Dim width) {
  for(Dim i=Dim(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=Dim(blockDim.x)*gridDim.x)
    output[i]=weight[Dim(indices[i/width])*width+i%width];
}
__device__ Dim mapped(Dim i,const Dim* m,std::size_t rank,std::size_t arg) {
  Dim offset=0;for(std::size_t a=rank;a-->0;){offset+=i%m[a]*m[(arg+1)*rank+a];i/=m[a];}return offset;
}
__device__ float attention_score(const float* scores,const void* mask,const Dim* m,std::size_t rank,
                                Dim i,Dim query,Dim col,float scale,unsigned mask_kind,bool causal) {
  if(causal && col>query)return -CUDART_INF_F;
  const Dim offset=mask_kind?mapped(i,m,rank,1):0;
  if(mask_kind==1 && !static_cast<const unsigned char*>(mask)[offset])return -CUDART_INF_F;
  const float bias=mask_kind==2?static_cast<const float*>(mask)[offset]:0.0F;
  if(bias==-CUDART_INF_F)return bias;
  return __fadd_rn(__fmul_rn(scores[mapped(i,m,rank,0)],scale),bias);
}
__global__ void attention_softmax(const float* scores,const void* mask,float* output,const Dim* m,
                                  Dim rows,Dim columns,std::size_t rank,float scale,unsigned mask_kind,bool causal) {
  for(Dim row=Dim(blockIdx.x)*blockDim.x+threadIdx.x;row<rows;row+=Dim(blockDim.x)*gridDim.x) {
    const Dim query=row%m[rank-2];float maximum=-CUDART_INF_F;
    for(Dim col=0;col<columns;++col){float value=attention_score(scores,mask,m,rank,row*columns+col,query,col,scale,mask_kind,causal);if(isnan(value)){maximum=value;break;}maximum=fmaxf(maximum,value);}
    if(maximum==-CUDART_INF_F){for(Dim col=0;col<columns;++col)output[row*columns+col]=0.0F;continue;}
    float total=0;
    for(Dim col=0;col<columns;++col)total+=expf(attention_score(scores,mask,m,rank,row*columns+col,query,col,scale,mask_kind,causal)-maximum);
    for(Dim col=0;col<columns;++col)output[row*columns+col]=expf(attention_score(scores,mask,m,rank,row*columns+col,query,col,scale,mask_kind,causal)-maximum)/total;
  }
}
}
cudaError_t launch_embedding_validate(const std::int32_t* indices,Dim count,Dim vocabulary,int* invalid) {
  if(!count)return cudaSuccess;embedding_validate<<<blocks(count),256>>>(indices,count,vocabulary,invalid);return finish();
}
cudaError_t launch_embedding_gather(const std::int32_t* indices,const void* weight,void* output,Dim count,Dim width) {
  if(!count)return cudaSuccess;embedding_gather<<<blocks(count),256>>>(indices,static_cast<const unsigned*>(weight),static_cast<unsigned*>(output),count,width);return finish();
}
cudaError_t launch_attention_softmax(const float* scores,const void* mask,float* output,const Dim* metadata,
                                     Dim rows,Dim columns,std::size_t rank,float scale,unsigned mask_kind,bool causal) {
  if(!rows)return cudaSuccess;attention_softmax<<<blocks(rows),256>>>(scores,mask,output,metadata,rows,columns,rank,scale,mask_kind,causal);return finish();
}
}
