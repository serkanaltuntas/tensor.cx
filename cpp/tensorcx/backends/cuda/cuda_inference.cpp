#include "tensorcx/backends/cuda/cuda_inference.h"
#include "tensorcx/backends/cuda/cuda_backend.h"
#include "tensorcx/backends/cuda/cuda_buffer.h"
#include "tensorcx/core/inference.h"
#include <limits>
namespace tensorcx::cuda {
Status execute_inference_primitive(const BackendExecution& e) {
  ContextScope device;if(!device.ready())return device.status();
  auto ptr=[&](std::size_t i){return std::static_pointer_cast<CudaBuffer>(e.inputs[i].buffer)->data();};
  if(e.op.kind==OpKind::kEmbedding) {
    const auto p=make_embedding_plan(e.inputs);
    if(p.indices) {
      auto flag=CudaBuffer::create(DType::kInt32,{1});if(!flag)return flag.status();
      auto status=runtime_status(cudaMemset(flag.value()->data(),0,sizeof(int)),"embedding validation initialization");if(!status.ok())return status;
      status=runtime_status(launch_embedding_validate(static_cast<const std::int32_t*>(ptr(0)),p.indices,p.vocabulary,static_cast<int*>(flag.value()->data())),"embedding validation");if(!status.ok())return status;
      int invalid=0;status=runtime_status(cudaMemcpy(&invalid,flag.value()->data(),sizeof(int),cudaMemcpyDeviceToHost),"embedding validation result");if(!status.ok())return status;
      if(invalid)return {StatusCode::kInvalidArgument,"embedding index out of range"};
    }
    auto output=CudaBuffer::create(DType::kFloat32,p.shape);if(!output)return output.status();
    auto status=runtime_status(launch_embedding_gather(static_cast<const std::int32_t*>(ptr(0)),ptr(1),output.value()->data(),numel(p.shape),p.width),"embedding gather");if(!status.ok())return status;
    e.outputs[0]=to_core_tensor(CudaTensor(DType::kFloat32,p.shape,output.move_value()));return Status::Ok();
  }
  const auto p=make_attention_softmax_plan(e.op,e.inputs);
  auto output=CudaBuffer::create(DType::kFloat32,p.shape);if(!output)return output.status();
  if(p.rows) {
    if(p.metadata.size()>static_cast<std::size_t>(std::numeric_limits<Dim>::max()/2))return {StatusCode::kInvalidArgument,"attention metadata size overflow"};
    auto metadata=CudaBuffer::create(DType::kInt32,{static_cast<Dim>(p.metadata.size()*2)});if(!metadata)return metadata.status();
    auto status=runtime_status(cudaMemcpy(metadata.value()->data(),p.metadata.data(),p.metadata.size()*sizeof(Dim),cudaMemcpyHostToDevice),"attention metadata upload");if(!status.ok())return status;
    status=runtime_status(launch_attention_softmax(static_cast<const float*>(ptr(0)),ptr(e.inputs.size()==2?1:0),static_cast<float*>(output.value()->data()),static_cast<const Dim*>(metadata.value()->data()),p.rows,p.columns,p.shape.size(),p.scale,p.mask_kind,e.op.causal),"attention softmax");if(!status.ok())return status;
  }
  e.outputs[0]=to_core_tensor(CudaTensor(DType::kFloat32,p.shape,output.move_value()));return Status::Ok();
}
}
