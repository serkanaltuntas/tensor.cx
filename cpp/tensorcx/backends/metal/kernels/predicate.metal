// Included by elementwise.metal. Operation codes match predicate_operation_code.
inline ulong predicate_offset(ulong i, device const long* metadata, ulong rank, ulong arg) {
  ulong offset = 0;
  for (ulong axis = rank; axis > 0; --axis) {
    offset += (i % ulong(metadata[axis - 1])) * ulong(metadata[(arg + 1) * rank + axis - 1]);
    i /= ulong(metadata[axis - 1]);
  }
  return offset;
}
template<typename T> inline bool predicate_compare(T a, T b, uint op) {
  switch (op) {
    case 0: return a == b;
    case 1: return a != b;
    case 2: return a < b;
    case 3: return a <= b;
    case 4: return a > b;
    case 5: return a >= b;
    default: return false;
  }
}
inline bool predicate_compare_float_bits(uint a, uint b, uint op) {
  // Metal floating comparisons can flush subnormals to zero even without
  // fast math. Read the stored IEEE words and compare ordered integer keys.
  const uint magnitude_a = a & 0x7fffffffu, magnitude_b = b & 0x7fffffffu;
  if (magnitude_a > 0x7f800000u || magnitude_b > 0x7f800000u) return op == 1;
  if (magnitude_a == 0) a = 0;  // Both signed zeros compare equal.
  if (magnitude_b == 0) b = 0;
  const uint key_a = (a & 0x80000000u) ? ~a : (a ^ 0x80000000u);
  const uint key_b = (b & 0x80000000u) ? ~b : (b ^ 0x80000000u);
  return predicate_compare(key_a, key_b, op);
}
kernel void predicate_values(device const uchar* a [[buffer(0)]],
                             device const uchar* b [[buffer(1)]],
                             device const uchar* c [[buffer(2)]],
                             device uchar* output [[buffer(3)]],
                             constant uint& count [[buffer(4)]],
                             device const long* metadata [[buffer(5)]],
                             constant ulong& rank [[buffer(6)]],
                             constant uint& op [[buffer(7)]],
                             constant uint& dtype [[buffer(8)]],
                             uint id [[thread_position_in_grid]]) {
  if (id >= count) return;
  const ulong ia = predicate_offset(id, metadata, rank, 0);
  if (op == 10) {
    const bool choose = a[ia] != 0;
    const ulong source = predicate_offset(id, metadata, rank, choose ? 1 : 2);
    device const uchar* branch = choose ? b : c;
    if (dtype == 2) output[id] = branch[source];
    else reinterpret_cast<device uint*>(output)[id] = reinterpret_cast<device const uint*>(branch)[source];
    return;
  }
  if (op == 9) { output[id] = !a[ia]; return; }
  const ulong ib = predicate_offset(id, metadata, rank, 1);
  if (op >= 6) {
    const bool left = a[ia] != 0, right = b[ib] != 0;
    output[id] = op == 6 ? left && right : op == 7 ? left || right : left != right;
  } else if (dtype == 0) output[id] = predicate_compare_float_bits(reinterpret_cast<device const uint*>(a)[ia], reinterpret_cast<device const uint*>(b)[ib], op);
  else if (dtype == 1) output[id] = predicate_compare(reinterpret_cast<device const int*>(a)[ia], reinterpret_cast<device const int*>(b)[ib], op);
  else output[id] = predicate_compare(a[ia] != 0, b[ib] != 0, op);
}
kernel void reduce_bool(device const uchar* input [[buffer(0)]], device uchar* output [[buffer(1)]],
                        constant uint& count [[buffer(2)]], constant long& reduce [[buffer(3)]],
                        device const long* metadata [[buffer(4)]], constant ulong& rank [[buffer(5)]],
                        constant uint& all [[buffer(6)]], uint id [[thread_position_in_grid]]) {
  if (id >= count) return;
  bool value = all != 0;
  for (long j = 0; j < reduce; ++j) {
    long remaining = long(id) * reduce + j, source = 0;
    for (ulong axis = rank; axis > 0; --axis) {
      source += remaining % metadata[2 * (axis - 1)] * metadata[2 * (axis - 1) + 1];
      remaining /= metadata[2 * (axis - 1)];
    }
    value = all ? value && input[source] != 0 : value || input[source] != 0;
  }
  output[id] = value;
}
kernel void cast_bool(device const uchar* input [[buffer(0)]], device uchar* output [[buffer(1)]],
                      constant uint& count [[buffer(2)]], constant uint& source [[buffer(3)]],
                      constant uint& target [[buffer(4)]], uint id [[thread_position_in_grid]]) {
  if (id >= count) return;
  if (source == 2 && target == 2) { output[id] = input[id]; return; }
  const bool value = source == 0 ? (reinterpret_cast<device const uint*>(input)[id] & 0x7fffffffu) != 0 :
                     source == 1 ? reinterpret_cast<device const int*>(input)[id] != 0 : input[id] != 0;
  if (target == 2) output[id] = value;
  else if (target == 0) reinterpret_cast<device float*>(output)[id] = value;
  else reinterpret_cast<device int*>(output)[id] = value;
}
kernel void fill_bool(device uchar* output [[buffer(0)]], constant uchar& value [[buffer(1)]],
                      constant uint& count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
  if (id < count) output[id] = value;
}
kernel void mask_prefix(device const uchar* mask [[buffer(0)]], device long* prefix [[buffer(1)]],
                        constant uint& count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
  if (id < count) prefix[id] = mask[id] != 0;
}
kernel void mask_scan(device const long* input [[buffer(0)]], device long* output [[buffer(1)]],
                      constant uint& count [[buffer(2)]], constant ulong& step [[buffer(3)]],
                      uint id [[thread_position_in_grid]]) {
  if (id < count) output[id] = input[id] + (id >= step ? input[id - step] : 0);
}
kernel void mask_copy(device const uchar* input [[buffer(0)]], device const uchar* mask [[buffer(1)]],
                      device const long* prefix [[buffer(2)]], device uchar* output [[buffer(3)]],
                      constant uint& count [[buffer(4)]], constant ulong& block [[buffer(5)]],
                      constant uint& dtype [[buffer(6)]], uint id [[thread_position_in_grid]]) {
  if (id >= count) return;
  const ulong m = id / block;
  if (mask[m]) {
    const ulong target = ulong(prefix[m] - 1) * block + id % block;
    if (dtype == 2) output[target] = input[id];
    else reinterpret_cast<device uint*>(output)[target] = reinterpret_cast<device const uint*>(input)[id];
  }
}
