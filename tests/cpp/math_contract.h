#pragma once
#include <array>
#include <cmath>
#include <stdexcept>
#include "tensorcx/backends/cpu/cpu_backend.h"

template<typename Upload,typename Download>
void math_contract(tensorcx::Backend& backend,Upload upload,Download download) {
  using namespace tensorcx;
  auto check=[](bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);};
  for(auto dtype:{DType::kFloat32,DType::kInt32}) {
    cpu::CpuTensor x(dtype,{2,3}),lo(dtype,{}),hi(dtype,{1,3});
    for(Dim i=0;i<x.size();++i) {
      if(dtype==DType::kFloat32)x.mutable_float_data()[i]=static_cast<float>(i+1);
      else x.mutable_int32_data()[i]=static_cast<std::int32_t>(i-3);
    }
    if(dtype==DType::kFloat32){lo.mutable_float_data()[0]=2;hi.mutable_float_data()={3,4,5};}
    else {lo.mutable_int32_data()[0]=-2;hi.mutable_int32_data()={0,1,2};}
    for(auto kind:{OpKind::kLog,OpKind::kSqrt,OpKind::kAbs,OpKind::kMin,OpKind::kArgmax,OpKind::kClip,OpKind::kTopK}) {
      std::array<Tensor,3> inputs{upload(x),upload(lo),upload(hi)};
      std::array<Tensor,3> cpu_inputs{cpu::to_core_tensor(x),cpu::to_core_tensor(lo),cpu::to_core_tensor(hi)};
      std::array<Tensor,2> outputs{inputs[0],inputs[0]},reference;
      OpDesc op{kind};op.axis=1;op.k=2;
      auto input_count=kind==OpKind::kClip?3:1,output_count=kind==OpKind::kTopK?2:1;
      BackendExecution e{BackendOpClass::kPrimitive,op,std::span<const Tensor>(inputs).first(input_count),std::span<Tensor>(outputs).first(output_count),std::nullopt,std::nullopt,{}};
      if(dtype==DType::kInt32&&(kind==OpKind::kLog||kind==OpKind::kSqrt)) {
        check(backend.execute(e).code()==StatusCode::kInvalidArgument,"integer log/sqrt accepted");
        check(outputs[0].buffer==inputs[0].buffer,"dtype error replaced output");continue;
      }
      cpu::CpuBackend cpu_backend;
      BackendExecution ce{BackendOpClass::kPrimitive,op,std::span<const Tensor>(cpu_inputs).first(input_count),std::span<Tensor>(reference).first(output_count),std::nullopt,std::nullopt,{}};
      check(cpu_backend.execute(ce).ok(),"math reference failed");
      auto status=backend.execute(e);check(status.ok(),status.message().c_str());
      for(int n=0;n<output_count;++n) {
        const auto actual=download(outputs[n]),expected=cpu::from_core_tensor(reference[n]);
        check(actual.dtype()==expected.dtype()&&actual.shape()==expected.shape(),"math output metadata");
        check(outputs[n].buffer!=inputs[0].buffer,"math aliases input");
        for(Dim i=0;i<actual.size();++i) {
          if(actual.dtype()==DType::kFloat32)check(std::abs(actual.float_data()[i]-expected.float_data()[i])<1e-6F,"math float parity");
          else check(actual.int32_data()[i]==expected.int32_data()[i],"math index/int parity");
        }
      }
      const auto original=inputs[0];const auto saved=outputs;
      for(int scenario=0;scenario<7;++scenario) {
        inputs[0]=original;
        switch(scenario) {
          case 0:inputs[0].buffer.reset();break; case 1:inputs[0].offset=1;break;
          case 2:inputs[0].device.index=1;break; case 3:inputs[0].dtype=DType::kBool;break;
          case 4:inputs[0].shape={-1};break;case 5:inputs[0].strides.clear();break;
          case 6:inputs[0].shape={2,4};inputs[0].strides={4,1};break;
        }
        check(backend.execute(e).code()==StatusCode::kInvalidArgument,"math accepted malformed input");
        for(int n=0;n<2;++n)check(outputs[n].buffer==saved[n].buffer&&outputs[n].shape==saved[n].shape,"math error replaced output slot");
      }
      inputs[0]=original;
      e.outputs=std::span<Tensor>(outputs).first(output_count==2?1:2);
      check(backend.execute(e).code()==StatusCode::kInvalidArgument,"math accepted output arity");
      for(int n=0;n<2;++n)check(outputs[n].buffer==saved[n].buffer,"math arity replaced output slot");
      if(kind==OpKind::kTopK) {
        e.outputs=outputs;e.op.k=4;
        check(backend.execute(e).code()==StatusCode::kInvalidArgument,"topk accepted oversized k");
        for(int n=0;n<2;++n)check(outputs[n].buffer==saved[n].buffer,"topk invalid k replaced output");
      }
    }
  }
}
