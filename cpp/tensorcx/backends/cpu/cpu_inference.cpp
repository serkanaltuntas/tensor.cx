#include "tensorcx/backends/cpu/cpu_backend.h"
#include "tensorcx/core/inference.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace tensorcx::cpu {
Status execute_inference_primitive(const BackendExecution& e) {
  if(e.op.kind==OpKind::kEmbedding) {
    const auto p=make_embedding_plan(e.inputs);
    const auto indices=from_core_tensor(e.inputs[0]),weight=from_core_tensor(e.inputs[1]);
    for(auto index:indices.int32_data())if(index<0 || index>=p.vocabulary)
      throw std::invalid_argument("embedding index out of range");
    CpuTensor output(DType::kFloat32,p.shape);
    for(Dim i=0;i<p.indices && p.width;++i)
      std::memcpy(output.mutable_float_data().data()+i*p.width,weight.float_data().data()+Dim(indices.int32_data()[i])*p.width,p.width*sizeof(float));
    e.outputs[0]=to_core_tensor(output);return Status::Ok();
  }
  const auto p=make_attention_softmax_plan(e.op,e.inputs);
  const auto input=from_core_tensor(e.inputs[0]);
  std::optional<CpuTensor> mask;if(e.inputs.size()==2)mask=from_core_tensor(e.inputs[1]);
  CpuTensor output(DType::kFloat32,p.shape);
  auto offset=[&](Dim i,std::size_t arg){Dim result=0;for(std::size_t a=p.shape.size();a-->0;){result+=i%p.metadata[a]*p.metadata[(arg+1)*p.shape.size()+a];i/=p.metadata[a];}return result;};
  for(Dim row=0;row<p.rows;++row) {
    const Dim query=row%p.shape[p.shape.size()-2];
    auto score=[&](Dim col){
      const Dim i=row*p.columns+col;
      if(e.op.causal && col>query)return -std::numeric_limits<float>::infinity();
      if(p.mask_kind==1 && !mask->bool_data()[offset(i,1)])return -std::numeric_limits<float>::infinity();
      const float bias=p.mask_kind==2?mask->float_data()[offset(i,1)]:0.0F;
      if(bias==-std::numeric_limits<float>::infinity())return bias;
      return input.float_data()[offset(i,0)]*p.scale+bias;
    };
    float maximum=-std::numeric_limits<float>::infinity();
    for(Dim col=0;col<p.columns;++col){const float value=score(col);if(std::isnan(value)){maximum=value;break;}maximum=std::max(maximum,value);}
    if(maximum==-std::numeric_limits<float>::infinity()) {
      for(Dim col=0;col<p.columns;++col)output.mutable_float_data()[row*p.columns+col]=0.0F;
      continue;
    }
    float total=0;
    for(Dim col=0;col<p.columns;++col)total+=std::exp(score(col)-maximum);
    for(Dim col=0;col<p.columns;++col)output.mutable_float_data()[row*p.columns+col]=std::exp(score(col)-maximum)/total;
  }
  e.outputs[0]=to_core_tensor(output);return Status::Ok();
}
}
