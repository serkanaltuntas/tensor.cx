#include "tensorcx/backends/cuda/cuda_math.h"
#include "tensorcx/backends/cuda/cuda_backend.h"
#include "tensorcx/backends/cuda/cuda_buffer.h"
#include "tensorcx/core/math.h"
#include <limits>
namespace tensorcx::cuda {
Status execute_math(const BackendExecution& e) {
  const auto p=make_math_plan(e.op,e.inputs);
  ContextScope device; if(!device.ready()) return device.status();
  auto output=CudaBuffer::create(p.dtype,p.shape); if(!output) return output.status();
  std::shared_ptr<CudaBuffer> indices;
  if(p.code==6) {auto buffer=CudaBuffer::create(DType::kInt32,p.shape);if(!buffer)return buffer.status();indices=buffer.move_value();}
  if(p.groups) {
    Shape metadata{p.groups,p.reduce,p.inner,p.k,static_cast<Dim>(p.rank),p.code,
                   e.inputs[0].dtype==DType::kFloat32,e.op.largest,e.op.sorted};
    metadata.insert(metadata.end(),p.metadata.begin(),p.metadata.end());
    if(metadata.size()>static_cast<std::size_t>(std::numeric_limits<Dim>::max()/2))
      return {StatusCode::kInvalidArgument,"math metadata size overflow"};
    auto gpu=CudaBuffer::create(DType::kInt32,{static_cast<Dim>(metadata.size()*2)});if(!gpu)return gpu.status();
    auto status=runtime_status(cudaMemcpy(gpu.value()->data(),metadata.data(),metadata.size()*sizeof(Dim),cudaMemcpyHostToDevice),"math metadata upload");
    if(!status.ok())return status;
    auto ptr=[&](std::size_t i){return std::static_pointer_cast<CudaBuffer>(e.inputs[i<e.inputs.size()?i:0].buffer)->data();};
    status=runtime_status(launch_math(ptr(0),ptr(1),ptr(2),output.value()->data(),indices?indices->data():output.value()->data(),
                                    static_cast<const Dim*>(gpu.value()->data()),p.groups),"CUDA math");
    if(!status.ok())return status;
  }
  auto result=to_core_tensor(CudaTensor(p.dtype,p.shape,output.move_value()));
  if(indices) {auto idx=to_core_tensor(CudaTensor(DType::kInt32,p.shape,indices));e.outputs[0]=std::move(result);e.outputs[1]=std::move(idx);}
  else e.outputs[0]=std::move(result);
  return Status::Ok();
}
}
