// Embedding validation precedes gather, even for zero-width tables.
kernel void embedding_validate(device const int* indices [[buffer(0)]],device atomic_uint* invalid [[buffer(1)]],
                               constant ulong& count [[buffer(2)]],constant long& vocabulary [[buffer(3)]],
                               uint id [[thread_position_in_grid]]) {
  if(ulong(id)<count && (indices[id]<0 || long(indices[id])>=vocabulary))atomic_store_explicit(invalid,1u,memory_order_relaxed);
}
kernel void embedding_gather(device const int* indices [[buffer(0)]],device const uint* weight [[buffer(1)]],
                             device uint* output [[buffer(2)]],constant uint& count [[buffer(3)]],
                             constant ulong& width [[buffer(4)]],uint id [[thread_position_in_grid]]) {
  if(id<count)output[id]=weight[ulong(indices[ulong(id)/width])*width+ulong(id)%width];
}
inline float inference_attention_score(device const float* scores,device const uchar* mask,
                                       device const long* m,ulong rank,ulong i,ulong query,ulong col,
                                       float scale,uint kind,bool causal) {
  if(causal && col>query)return -INFINITY;
  const ulong offset=kind?predicate_offset(i,m,rank,1):0;
  if(kind==1 && !mask[offset])return -INFINITY;
  const float bias=kind==2?reinterpret_cast<device const float*>(mask)[offset]:0.0f;
  if(bias==-INFINITY)return bias;
  const float score = scores[predicate_offset(i,m,rank,0)];
  const uint scale_bits = as_type<uint>(scale), magnitude = scale_bits & 0x7fffffffu;
  // Reconstruct subnormal scale operands using integer bits on Metal devices
  // that flush denormals. The enlarged scale cannot overflow a finite score.
  if (magnitude != 0 && magnitude < 0x00800000u) {
    const float enlarged = float(magnitude) * 0x1p-125f * ((scale_bits & 0x80000000u) ? -1.0f : 1.0f);
    return (score * enlarged) * 0x1p-24f + bias;
  }
  return score*scale+bias;
}
kernel void attention_softmax(device const float* scores [[buffer(0)]],device const uchar* mask [[buffer(1)]],
                              device float* output [[buffer(2)]],device const long* m [[buffer(3)]],
                              constant uint& rows [[buffer(4)]],constant ulong& columns [[buffer(5)]],
                              constant ulong& rank [[buffer(6)]],constant float& scale [[buffer(7)]],
                              constant uint& kind [[buffer(8)]],constant uint& causal [[buffer(9)]],
                              uint id [[thread_position_in_grid]]) {
  if(id>=rows)return;const ulong row=id,query=row%ulong(m[rank-2]);float maximum=-INFINITY;
  for(ulong col=0;col<columns;++col){float value=inference_attention_score(scores,mask,m,rank,row*columns+col,query,col,scale,kind,causal!=0);if(isnan(value)){maximum=value;break;}maximum=max(maximum,value);}
  if(maximum==-INFINITY){for(ulong col=0;col<columns;++col)output[row*columns+col]=0.0f;return;}
  float total=0;
  for(ulong col=0;col<columns;++col)total+=exp(inference_attention_score(scores,mask,m,rank,row*columns+col,query,col,scale,kind,causal!=0)-maximum);
  for(ulong col=0;col<columns;++col)output[row*columns+col]=exp(inference_attention_score(scores,mask,m,rank,row*columns+col,query,col,scale,kind,causal!=0)-maximum)/total;
}
