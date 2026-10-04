#include "tensorcx/backends/cpu/cpu_backend.h"
#include "tensorcx/core/math.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <numeric>
namespace tensorcx::cpu {
namespace {
using Word = std::uint32_t;
Word load(const CpuTensor& x, Dim i) { Word w; std::memcpy(&w, static_cast<const char*>(x.data())+i*4,4); return w; }
void store(CpuTensor& x, Dim i, Word w) { std::memcpy(static_cast<char*>(x.mutable_data())+i*4,&w,4); }
bool nan(Word a, bool fp) { return fp && (a & 0x7fffffffu) > 0x7f800000u; }
bool less(Word a, Word b, bool fp) {
  if (!fp) return std::bit_cast<std::int32_t>(a) < std::bit_cast<std::int32_t>(b);
  if (!(a & 0x7fffffffu)) a=0; if (!(b & 0x7fffffffu)) b=0;
  return ((a & 0x80000000u) ? ~a : (a ^ 0x80000000u)) <
         ((b & 0x80000000u) ? ~b : (b ^ 0x80000000u));
}
Dim mapped(Dim i, const MathPlan& p) {
  Dim source=0;
  for (std::size_t a=p.rank; a-- > 0;) { source += i%p.metadata[2*a]*p.metadata[2*a+1]; i/=p.metadata[2*a]; }
  return source;
}
}
Status execute_math(const BackendExecution& e) {
  std::vector<CpuTensor> inputs;
  for (const auto& t : e.inputs) inputs.push_back(from_core_tensor(t));
  const auto p=make_math_plan(e.op,e.inputs); const bool fp=inputs[0].dtype()==DType::kFloat32;
  CpuTensor out(p.dtype,p.shape);
  std::optional<CpuTensor> indices;
  if (p.code==6) indices.emplace(DType::kInt32,p.shape);
  const auto& x=inputs[0];
  for (Dim i=0;i<p.groups;++i) {
    if (p.code<3) {
      auto a=load(x,i);
      if (p.code==2) a=fp ? a&0x7fffffffu : ((a&0x80000000u) ? 0u-a : a);
      else { const float f=std::bit_cast<float>(a); a=std::bit_cast<Word>(p.code==0 ? std::log(f) : std::sqrt(f)); }
      store(out,i,a);
    } else if (p.code==5) {
      auto offset=[&](std::size_t arg) { Dim flat=i,result=0; for(std::size_t a=p.rank;a-->0;) {result+=flat%p.metadata[a]*p.metadata[(arg+1)*p.rank+a];flat/=p.metadata[a];}return result;};
      auto a=load(x,offset(0)), lo=load(inputs[1],offset(1)), hi=load(inputs[2],offset(2));
      if (!nan(a,fp)) a=nan(lo,fp)||!less(lo,a,fp) ? lo : a;
      if (!nan(a,fp)) a=nan(hi,fp)||!less(a,hi,fp) ? hi : a;
      store(out,i,a);
    } else if (p.code==3 || p.code==4) {
      Word best=load(x,mapped(i*p.reduce,p)); Dim index=0;
      for (Dim j=1;j<p.reduce;++j) {
        const auto a=load(x,mapped(i*p.reduce+j,p));
        if (!nan(best,fp) && (nan(a,fp)||(p.code==3 ? less(a,best,fp) : less(best,a,fp)))) {best=a;index=j;}
      }
      store(out,i,p.code==4 ? static_cast<Word>(index) : best);
    } else {
      const Dim base=(i/p.inner)*p.reduce*p.inner+i%p.inner;
      std::vector<std::int32_t> order(static_cast<std::size_t>(p.reduce));
      std::iota(order.begin(),order.end(),0);
      auto before=[&](Dim ia,Dim ib) {
        Word a=load(x,base+ia*p.inner),b=load(x,base+ib*p.inner);
        const bool an=nan(a,fp),bn=nan(b,fp);
        if(an!=bn) return e.op.largest ? an : bn;
        if(!an && (less(a,b,fp)||less(b,a,fp))) return e.op.largest ? less(b,a,fp) : less(a,b,fp);
        return ia<ib;
      };
      std::partial_sort(order.begin(),order.begin()+p.k,order.end(),before);
      if(!e.op.sorted) std::sort(order.begin(),order.begin()+p.k);
      for(Dim j=0;j<p.k;++j) {Dim dest=(i/p.inner)*p.k*p.inner+j*p.inner+i%p.inner;store(out,dest,load(x,base+order[j]*p.inner));store(*indices,dest,order[j]);}
    }
  }
  // Publish all outputs only after validation and computation succeed.
  auto result=to_core_tensor(out);
  if(indices) {auto idx=to_core_tensor(*indices);e.outputs[0]=std::move(result);e.outputs[1]=std::move(idx);}
  else e.outputs[0]=std::move(result);
  return Status::Ok();
}
}
