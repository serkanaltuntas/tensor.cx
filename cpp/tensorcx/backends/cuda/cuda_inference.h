#pragma once
#include "tensorcx/core/backend.h"
#include <cuda_runtime_api.h>
namespace tensorcx::cuda {
Status execute_inference_primitive(const BackendExecution& execution);
cudaError_t launch_embedding_validate(const std::int32_t* indices,Dim count,Dim vocabulary,int* invalid);
cudaError_t launch_embedding_gather(const std::int32_t* indices,const void* weight,void* output,Dim count,Dim width);
cudaError_t launch_attention_softmax(const float* scores,const void* mask,float* output,const Dim* metadata,
                                     Dim rows,Dim columns,std::size_t rank,float scale,unsigned mask_kind,bool causal);
}
