// Included by elementwise.metal; operation codes come from core MathPlan.
namespace tensorcx_math {
using Word=uint;using Index=long;
inline bool math_nan(Word a,bool fp) {return fp && (a&0x7fffffffu)>0x7f800000u;}
inline bool math_less(Word a,Word b,bool fp) {
  if(!fp) return as_type<int>(a)<as_type<int>(b);
  if(!(a&0x7fffffffu))a=0; if(!(b&0x7fffffffu))b=0;
  return ((a&0x80000000u)?~a:(a^0x80000000u))<((b&0x80000000u)?~b:(b^0x80000000u));
}
inline bool math_before(Word a,Index ia,Word b,Index ib,bool fp,bool largest) {
  const bool an=math_nan(a,fp),bn=math_nan(b,fp);
  if(an!=bn)return largest?an:bn;
  if(!an && (math_less(a,b,fp)||math_less(b,a,fp)))return largest?math_less(b,a,fp):math_less(a,b,fp);
  return ia<ib;
}
inline Index math_mapped(Index flat,device const long* m,Index rank) {
  Index result=0;
  for(Index axis=rank;axis>0;--axis){result+=flat%m[2*(axis-1)]*m[2*(axis-1)+1];flat/=m[2*(axis-1)];}
  return result;
}
inline Index math_broadcast(Index flat,device const long* m,Index rank,Index arg) {
  Index result=0;
  for(Index axis=rank;axis>0;--axis){result+=flat%m[axis-1]*m[(arg+1)*rank+axis-1];flat/=m[axis-1];}
  return result;
}
inline Word math_unary(Word a,Index code,bool fp) {
  if(code==2)return fp?a&0x7fffffffu:((a&0x80000000u)?0u-a:a);
  const Word mag=a&0x7fffffffu;
  if(mag>0x7f800000u)return a;
  if(!mag)return code==0?0xff800000u:a;
  if(a&0x80000000u)return 0x7fc00000u;
  if(mag==0x7f800000u)return a;
  // Reconstruct subnormals with integer magnitude before any float operation;
  // this also works on Metal hardware that flushes float subnormal operands.
  const bool sub=mag<0x00800000u;
  const float value=sub?float(mag)*0x1p-125f:as_type<float>(a);
  const float result=code==0 ? log(value)-(sub?24.0f*0.6931471805599453f:0.0f) : sqrt(value)*(sub?0x1p-12f:1.0f);
  return as_type<uint>(result);
}

}
kernel void math_values(device const uint* a [[buffer(0)]],device const uint* b [[buffer(1)]],
                        device const uint* c [[buffer(2)]],device uint* out [[buffer(3)]],
                        device uint* indices [[buffer(4)]],device const long* params [[buffer(5)]],
                        uint tid [[thread_position_in_grid]]) {
  using namespace tensorcx_math;
  const Index id=tid;if(id>=params[0])return;
  const Index reduce=params[1],inner=params[2],k=params[3],rank=params[4],code=params[5];
  const bool fp=params[6]!=0,largest=params[7]!=0,sorted=params[8]!=0;
  device const long* m=params+9;
  if(code<3) {out[id]=math_unary(a[id],code,fp);}
  else if(code==5) {
    Word value=a[math_broadcast(id,m,rank,0)],lo=b[math_broadcast(id,m,rank,1)],hi=c[math_broadcast(id,m,rank,2)];
    if(!math_nan(value,fp))value=math_nan(lo,fp)||!math_less(lo,value,fp)?lo:value;
    if(!math_nan(value,fp))value=math_nan(hi,fp)||!math_less(value,hi,fp)?hi:value;
    out[id]=value;
  } else if(code==3 || code==4) {
    Word best=a[math_mapped(id*reduce,m,rank)];Index index=0;
    for(Index j=1;j<reduce;++j){Word value=a[math_mapped(id*reduce+j,m,rank)];
      if(!math_nan(best,fp)&&(math_nan(value,fp)||(code==3?math_less(value,best,fp):math_less(best,value,fp)))){best=value;index=j;}}
    out[id]=code==4?Word(index):best;
  } else {
    const Index base=(id/inner)*reduce*inner+id%inner,dest=(id/inner)*k*inner+id%inner;
    Word previous=0;Index previous_index=-1;
    // Select the next value in deterministic order. No expanded input or host copy.
    for(Index slot=0;slot<k;++slot) {
      Word best=0;Index best_index=-1;
      for(Index j=0;j<reduce;++j) {
        const Word value=a[base+j*inner];
        if(previous_index>=0&&!math_before(previous,previous_index,value,j,fp,largest))continue;
        if(best_index<0||math_before(value,j,best,best_index,fp,largest)){best=value;best_index=j;}
      }
      previous=best;previous_index=best_index;
      Index position=slot;
      if(!sorted) {while(position>0 && indices[dest+(position-1)*inner]>Word(best_index)) {
        indices[dest+position*inner]=indices[dest+(position-1)*inner];--position;}}
      indices[dest+position*inner]=Word(best_index);
    }
    for(Index slot=0;slot<k;++slot)out[dest+slot*inner]=a[base+Index(indices[dest+slot*inner])*inner];
  }

}
