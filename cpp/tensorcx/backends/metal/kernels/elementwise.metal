#include <metal_stdlib>

using namespace metal;
#include "predicate.metal"
#include "math.metal"

// Three ulong entries per axis: output extent, left stride, right stride.
// Dynamic metadata avoids imposing an artificial tensor-rank limit. Broadcast
// axes have stride zero; the actual tensor buffers are never expanded.
inline ulong2 broadcast_offsets(uint id, device const ulong* metadata, ulong rank) {
  ulong remaining = id;
  ulong2 offsets(0);
  for (ulong axis = rank; axis > 0; --axis) {
    const ulong entry = (axis - 1) * 3;
    const ulong coordinate = remaining % metadata[entry];
    remaining /= metadata[entry];
    offsets.x += coordinate * metadata[entry + 1];
    offsets.y += coordinate * metadata[entry + 2];
  }
  return offsets;
}

kernel void broadcast_f32(device const float* lhs [[buffer(0)]],
                          device const float* rhs [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          constant uint& n [[buffer(3)]],
                          device const ulong* metadata [[buffer(4)]],
                          constant ulong& rank [[buffer(5)]],
                          constant uint& operation [[buffer(6)]],
                          uint id [[thread_position_in_grid]]) {
  if (id < n) {
    const ulong2 offsets = broadcast_offsets(id, metadata, rank);
    const float a = lhs[offsets.x], b = rhs[offsets.y];
    switch (operation) {
      case 0: out[id] = a + b; break;
      case 1: out[id] = a - b; break;
      case 2: out[id] = a * b; break;
      case 3: out[id] = precise::divide(a, b); break;
    }
  }
}

kernel void broadcast_i32(device const int* lhs [[buffer(0)]],
                          device const int* rhs [[buffer(1)]],
                          device int* out [[buffer(2)]],
                          constant uint& n [[buffer(3)]],
                          device const ulong* metadata [[buffer(4)]],
                          constant ulong& rank [[buffer(5)]],
                          constant uint& operation [[buffer(6)]],
                          uint id [[thread_position_in_grid]]) {
  if (id < n) {
    const ulong2 offsets = broadcast_offsets(id, metadata, rank);
    const uint a = uint(lhs[offsets.x]), b = uint(rhs[offsets.y]);
    switch (operation) {
      case 0: out[id] = as_type<int>(a + b); break;
      case 1: out[id] = as_type<int>(a - b); break;
      case 2: out[id] = as_type<int>(a * b); break;
    }
  }
}

// Two signed long entries per output axis: extent and mapped input element stride.
kernel void transpose_bits(device const uint* input [[buffer(0)]],
                           device uint* out [[buffer(1)]],
                           constant uint& n [[buffer(2)]],
                           device const long* metadata [[buffer(3)]],
                           constant ulong& rank [[buffer(4)]],
                           constant long& offset [[buffer(5)]],
                           uint id [[thread_position_in_grid]]) {
  if (id < n) {
    long remaining = id, source = offset;
    for (ulong axis = rank; axis > 0; --axis) {
      const ulong entry = (axis - 1) * 2;
      source += (remaining % metadata[entry]) * metadata[entry + 1];
      remaining /= metadata[entry];
    }
    out[id] = input[source];
  }
}

kernel void concat_bits(device const uint* input [[buffer(0)]],
                        device uint* out [[buffer(1)]],
                        constant uint& n [[buffer(2)]],
                        constant ulong& block [[buffer(3)]],
                        constant ulong& output_block [[buffer(4)]],
                        constant ulong& offset [[buffer(5)]],
                        uint id [[thread_position_in_grid]]) {
  if (id < n) out[(id / block) * output_block + offset + id % block] = input[id];
}

kernel void copy_bits(device const uint* input [[buffer(0)]],
                      device uint* out [[buffer(1)]],
                      constant uint& n [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
  if (id < n) out[id] = input[id];
}

kernel void cast_i32_f32(device const int* input [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant uint& n [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
  if (id < n) out[id] = float(input[id]);
}

kernel void cast_f32_i32(device const float* input [[buffer(0)]],
                         device int* out [[buffer(1)]],
                         constant uint& n [[buffer(2)]],
                         device atomic_uint* invalid [[buffer(3)]],
                         uint id [[thread_position_in_grid]]) {
  if (id < n) {
    const float value = input[id];
    if (!isfinite(value) || value < -2147483648.0f || value >= 2147483648.0f) {
      atomic_store_explicit(invalid, 1u, memory_order_relaxed);
    } else {
      // MSL floating-to-integer conversion truncates toward zero. Guard first
      // so an invalid conversion never reaches the device instruction.
      out[id] = int(value);
    }
  }
}

kernel void add_f32(device const float* lhs [[buffer(0)]],
                    device const float* rhs [[buffer(1)]],
                    device float* out [[buffer(2)]],
                    constant uint& n [[buffer(3)]],
                    uint id [[thread_position_in_grid]]) {
  if (id < n) {
    out[id] = lhs[id] + rhs[id];
  }
}

kernel void mul_f32(device const float* lhs [[buffer(0)]],
                    device const float* rhs [[buffer(1)]],
                    device float* out [[buffer(2)]],
                    constant uint& n [[buffer(3)]],
                    uint id [[thread_position_in_grid]]) {
  if (id < n) {
    out[id] = lhs[id] * rhs[id];
  }
}

kernel void sub_f32(device const float* lhs [[buffer(0)]],
                    device const float* rhs [[buffer(1)]],
                    device float* out [[buffer(2)]],
                    constant uint& n [[buffer(3)]],
                    uint id [[thread_position_in_grid]]) {
  if (id < n) {
    out[id] = lhs[id] - rhs[id];
  }
}

kernel void div_f32(device const float* lhs [[buffer(0)]],
                    device const float* rhs [[buffer(1)]],
                    device float* out [[buffer(2)]],
                    constant uint& n [[buffer(3)]],
                    uint id [[thread_position_in_grid]]) {
  if (id < n) {
    out[id] = precise::divide(lhs[id], rhs[id]);
  }
}

kernel void neg_f32(device const float* input [[buffer(0)]],
                    device float* out [[buffer(1)]],
                    constant uint& n [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
  if (id < n) {
    // Flip only the sign bit, preserving signed zero and NaN payloads.
    out[id] = as_type<float>(as_type<uint>(input[id]) ^ 0x80000000u);
  }
}

kernel void scalar_f32(device const float* input [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       constant uint& n [[buffer(2)]],
                       constant float& scalar [[buffer(3)]],
                       constant uint& operation [[buffer(4)]],
                       constant uint& scalar_left [[buffer(5)]],
                       uint id [[thread_position_in_grid]]) {
  if (id < n) {
    const float lhs = scalar_left ? scalar : input[id];
    const float rhs = scalar_left ? input[id] : scalar;
    // Codes match arithmetic_operation_code in metal_kernels.cpp.
    switch (operation) {
      case 0: out[id] = lhs + rhs; break;
      case 1: out[id] = lhs - rhs; break;
      case 2: out[id] = lhs * rhs; break;
      case 3: out[id] = precise::divide(lhs, rhs); break;
    }
  }
}

kernel void fill_f32(device float* out [[buffer(0)]],
                     constant float& value [[buffer(1)]],
                     constant uint& n [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
  if (id < n) {
    out[id] = value;
  }
}

kernel void add_i32(device const int* lhs [[buffer(0)]],
                    device const int* rhs [[buffer(1)]],
                    device int* out [[buffer(2)]],
                    constant uint& n [[buffer(3)]],
                    uint id [[thread_position_in_grid]]) {
  if (id < n) {
    // Wrap via uint so int32 overflow is defined two's-complement, matching the
    // CPU reference (uint round-trip) and NumPy for the §12.3 exact contract.
    out[id] = int(uint(lhs[id]) + uint(rhs[id]));
  }
}

kernel void mul_i32(device const int* lhs [[buffer(0)]],
                    device const int* rhs [[buffer(1)]],
                    device int* out [[buffer(2)]],
                    constant uint& n [[buffer(3)]],
                    uint id [[thread_position_in_grid]]) {
  if (id < n) {
    out[id] = int(uint(lhs[id]) * uint(rhs[id]));
  }
}

kernel void sub_i32(device const int* lhs [[buffer(0)]],
                    device const int* rhs [[buffer(1)]],
                    device int* out [[buffer(2)]],
                    constant uint& n [[buffer(3)]],
                    uint id [[thread_position_in_grid]]) {
  if (id < n) {
    out[id] = as_type<int>(uint(lhs[id]) - uint(rhs[id]));
  }
}

kernel void neg_i32(device const int* input [[buffer(0)]],
                    device int* out [[buffer(1)]],
                    constant uint& n [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
  if (id < n) {
    out[id] = as_type<int>(0u - uint(input[id]));
  }
}

kernel void scalar_i32(device const int* input [[buffer(0)]],
                       device int* out [[buffer(1)]],
                       constant uint& n [[buffer(2)]],
                       constant int& scalar [[buffer(3)]],
                       constant uint& operation [[buffer(4)]],
                       constant uint& scalar_left [[buffer(5)]],
                       uint id [[thread_position_in_grid]]) {
  if (id < n) {
    const uint lhs = uint(scalar_left ? scalar : input[id]);
    const uint rhs = uint(scalar_left ? input[id] : scalar);
    // Unsigned arithmetic defines wrapping, including INT_MIN negation.
    switch (operation) {
      case 0: out[id] = as_type<int>(lhs + rhs); break;
      case 1: out[id] = as_type<int>(lhs - rhs); break;
      case 2: out[id] = as_type<int>(lhs * rhs); break;
    }
  }
}

kernel void fill_i32(device int* out [[buffer(0)]],
                     constant int& value [[buffer(1)]],
                     constant uint& n [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
  if (id < n) {
    out[id] = value;
  }
}

kernel void exp_f32(device const float* input [[buffer(0)]],
                    device float* out [[buffer(1)]],
                    constant uint& n [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
  if (id < n) {
    out[id] = exp(input[id]);
  }
}

kernel void gelu_f32(device const float* input [[buffer(0)]],
                     device float* out [[buffer(1)]],
                     constant uint& n [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
  if (id < n) {
    const float x = input[id];
    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    out[id] = 0.5f * x * (1.0f + tanh(inner));
  }
}

kernel void silu_f32(device const float* input [[buffer(0)]],
                     device float* out [[buffer(1)]],
                     constant uint& n [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
  if (id < n) {
    const float x = input[id];
    out[id] = x / (1.0f + exp(-x));
  }
}

kernel void softmax_f32(device const float* input [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& total_n [[buffer(2)]],
                        constant uint& reduce_n [[buffer(3)]],
                        constant uint& inner_n [[buffer(4)]],
                        uint id [[thread_position_in_grid]]) {
  if (id >= total_n) {
    return;
  }

  const uint slice_n = reduce_n * inner_n;
  const uint outer_index = id / slice_n;
  const uint inner_index = id % inner_n;
  const uint base = outer_index * slice_n + inner_index;

  float max_value = -INFINITY;
  for (uint reduce_index = 0; reduce_index < reduce_n; ++reduce_index) {
    const float value = input[base + reduce_index * inner_n];
    if (isnan(value)) {
      max_value = value;
      break;
    }
    max_value = max(max_value, value);
  }

  float denom = 0.0f;
  for (uint reduce_index = 0; reduce_index < reduce_n; ++reduce_index) {
    denom += exp(input[base + reduce_index * inner_n] - max_value);
  }
  out[id] = exp(input[id] - max_value) / denom;
}

kernel void rmsnorm_f32(device const float* input [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& total_n [[buffer(2)]],
                        constant uint& reduce_n [[buffer(3)]],
                        constant uint& inner_n [[buffer(4)]],
                        constant float& epsilon [[buffer(5)]],
                        uint id [[thread_position_in_grid]]) {
  if (id >= total_n) {
    return;
  }

  const uint slice_n = reduce_n * inner_n;
  const uint outer_index = id / slice_n;
  const uint inner_index = id % inner_n;
  const uint base = outer_index * slice_n + inner_index;

  float sum_squares = 0.0f;
  for (uint reduce_index = 0; reduce_index < reduce_n; ++reduce_index) {
    const float value = input[base + reduce_index * inner_n];
    sum_squares += value * value;
  }
  const float scale = rsqrt((sum_squares / float(reduce_n)) + epsilon);
  out[id] = input[id] * scale;
}

kernel void layernorm_f32(device const float* input [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant uint& total_n [[buffer(2)]],
                          constant uint& reduce_n [[buffer(3)]],
                          constant uint& inner_n [[buffer(4)]],
                          constant float& epsilon [[buffer(5)]],
                          uint id [[thread_position_in_grid]]) {
  if (id >= total_n) {
    return;
  }

  const uint slice_n = reduce_n * inner_n;
  const uint outer_index = id / slice_n;
  const uint inner_index = id % inner_n;
  const uint base = outer_index * slice_n + inner_index;

  float sum = 0.0f;
  const float first_value = input[base];
  bool all_equal = true;
  for (uint reduce_index = 0; reduce_index < reduce_n; ++reduce_index) {
    const float value = input[base + reduce_index * inner_n];
    sum += value;
    all_equal = all_equal && value == first_value;
  }
  if (all_equal && isfinite(first_value)) {
    out[id] = epsilon == 0.0f ? as_type<float>(0x7fc00000u) : 0.0f;
    return;
  }
  const float mean = sum / float(reduce_n);

  float sum_squared_diff = 0.0f;
  for (uint reduce_index = 0; reduce_index < reduce_n; ++reduce_index) {
    const float diff = input[base + reduce_index * inner_n] - mean;
    sum_squared_diff += diff * diff;
  }
  const float variance = sum_squared_diff / float(reduce_n);
  const float denom = variance + epsilon;
  if (denom == 0.0f) {
    out[id] = as_type<float>(0x7fc00000u);
    return;
  }

  const float scale = rsqrt(denom);
  out[id] = (input[id] - mean) * scale;
}

kernel void matmul_f32(device const float* lhs [[buffer(0)]],
                       device const float* rhs [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       constant ulong& m [[buffer(3)]],
                       constant ulong& k [[buffer(4)]],
                       constant ulong& n [[buffer(5)]],
                       constant uint& count [[buffer(6)]],
                       device const long* metadata [[buffer(7)]],
                       constant ulong& rank [[buffer(8)]],
                       uint id [[thread_position_in_grid]]) {
  if (id >= count) return;
  const ulong row = (ulong(id) / n) % m, col = ulong(id) % n;
  ulong batch = (ulong(id) / n) / m, left = 0, right = 0;
  for (ulong axis = rank; axis > 0; --axis) {
    const ulong coordinate = batch % ulong(metadata[axis - 1]);
    batch /= ulong(metadata[axis - 1]);
    left += coordinate * ulong(metadata[rank + axis - 1]);
    right += coordinate * ulong(metadata[2 * rank + axis - 1]);
  }
  float sum = 0.0f;
  for (ulong inner = 0; inner < k; ++inner)
    sum += lhs[left + row * k + inner] * rhs[right + inner * n + col];
  out[id] = sum;
}

inline ulong reduction_offset(ulong index, device const ulong* metadata, ulong begin, ulong end) {
  ulong offset = 0;
  for (ulong axis = end; axis > begin; --axis) {
    const ulong entry = (axis - 1) * 2;
    offset += (index % metadata[entry]) * metadata[entry + 1];
    index /= metadata[entry];
  }
  return offset;
}

kernel void reduce_axes_f32(device const float* input [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant uint& output_n [[buffer(2)]],
                            constant uint& reduce_n [[buffer(3)]],
                            device const ulong* metadata [[buffer(4)]],
                            constant ulong& output_rank [[buffer(5)]],
                            constant ulong& rank [[buffer(6)]],
                            constant uint& operation [[buffer(7)]],
                            uint id [[thread_position_in_grid]]) {
  if (id >= output_n) return;
  const ulong base = reduction_offset(id, metadata, 0, output_rank);
  float value = operation == 1 ? -INFINITY : 0.0f;
  for (uint r = 0; r < reduce_n; ++r) {
    const float item = input[base + reduction_offset(r, metadata, output_rank, rank)];
    if (operation == 1) {
      if (isnan(item)) { value = item; break; }
      if (value < item) value = item;
    } else value += item;
  }
  out[id] = operation == 2 ? value / float(reduce_n) : value;
}

kernel void reduce_axes_i32(device const int* input [[buffer(0)]],
                            device int* out [[buffer(1)]],
                            constant uint& output_n [[buffer(2)]],
                            constant uint& reduce_n [[buffer(3)]],
                            device const ulong* metadata [[buffer(4)]],
                            constant ulong& output_rank [[buffer(5)]],
                            constant ulong& rank [[buffer(6)]],
                            constant uint& operation [[buffer(7)]],
                            uint id [[thread_position_in_grid]]) {
  if (id >= output_n) return;
  const ulong base = reduction_offset(id, metadata, 0, output_rank);
  uint sum = 0;
  int maximum = (-2147483647 - 1);
  for (uint r = 0; r < reduce_n; ++r) {
    const int item = input[base + reduction_offset(r, metadata, output_rank, rank)];
    if (operation == 1) maximum = max(maximum, item);
    else sum += uint(item);
  }
  out[id] = operation == 1 ? maximum : as_type<int>(sum);
}

kernel void reduce_sum_f32(device const float* input [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           constant uint& output_n [[buffer(2)]],
                           constant uint& reduce_n [[buffer(3)]],
                           constant uint& inner_n [[buffer(4)]],
                           uint id [[thread_position_in_grid]]) {
  if (id >= output_n) {
    return;
  }

  const uint outer_index = id / inner_n;
  const uint inner_index = id - outer_index * inner_n;
  const uint base = outer_index * reduce_n * inner_n + inner_index;
  float sum = 0.0f;
  for (uint reduce_index = 0; reduce_index < reduce_n; ++reduce_index) {
    sum += input[base + reduce_index * inner_n];
  }
  out[id] = sum;
}

kernel void reduce_max_f32(device const float* input [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           constant uint& output_n [[buffer(2)]],
                           constant uint& reduce_n [[buffer(3)]],
                           constant uint& inner_n [[buffer(4)]],
                           uint id [[thread_position_in_grid]]) {
  if (id >= output_n) {
    return;
  }

  const uint outer_index = id / inner_n;
  const uint inner_index = id - outer_index * inner_n;
  const uint base = outer_index * reduce_n * inner_n + inner_index;
  float max_value = -INFINITY;
  for (uint reduce_index = 0; reduce_index < reduce_n; ++reduce_index) {
    const float value = input[base + reduce_index * inner_n];
    if (isnan(value)) {
      max_value = value;
      break;
    }
    max_value = max(max_value, value);
  }
  out[id] = max_value;
}

kernel void reduce_mean_f32(device const float* input [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant uint& output_n [[buffer(2)]],
                            constant uint& reduce_n [[buffer(3)]],
                            constant uint& inner_n [[buffer(4)]],
                            uint id [[thread_position_in_grid]]) {
  if (id >= output_n) {
    return;
  }

  if (reduce_n == 0) {
    out[id] = NAN;
    return;
  }

  const uint outer_index = id / inner_n;
  const uint inner_index = id - outer_index * inner_n;
  const uint base = outer_index * reduce_n * inner_n + inner_index;
  float sum = 0.0f;
  for (uint reduce_index = 0; reduce_index < reduce_n; ++reduce_index) {
    sum += input[base + reduce_index * inner_n];
  }
  out[id] = sum / float(reduce_n);
}

kernel void reduce_sum_i32(device const int* input [[buffer(0)]],
                           device int* out [[buffer(1)]],
                           constant uint& output_n [[buffer(2)]],
                           constant uint& reduce_n [[buffer(3)]],
                           constant uint& inner_n [[buffer(4)]],
                           uint id [[thread_position_in_grid]]) {
  if (id >= output_n) {
    return;
  }

  const uint outer_index = id / inner_n;
  const uint inner_index = id - outer_index * inner_n;
  const uint base = outer_index * reduce_n * inner_n + inner_index;
  uint sum = 0;
  for (uint reduce_index = 0; reduce_index < reduce_n; ++reduce_index) {
    sum += uint(input[base + reduce_index * inner_n]);
  }
  out[id] = int(sum);
}

kernel void reduce_max_i32(device const int* input [[buffer(0)]],
                           device int* out [[buffer(1)]],
                           constant uint& output_n [[buffer(2)]],
                           constant uint& reduce_n [[buffer(3)]],
                           constant uint& inner_n [[buffer(4)]],
                           uint id [[thread_position_in_grid]]) {
  if (id >= output_n) {
    return;
  }

  const uint outer_index = id / inner_n;
  const uint inner_index = id - outer_index * inner_n;
  const uint base = outer_index * reduce_n * inner_n + inner_index;
  int max_value = -2147483647 - 1;
  for (uint reduce_index = 0; reduce_index < reduce_n; ++reduce_index) {
    max_value = max(max_value, input[base + reduce_index * inner_n]);
  }
  out[id] = max_value;
}

kernel void transpose_bool(device const uchar* input [[buffer(0)]],
                           device uchar* out [[buffer(1)]],
                           constant uint& n [[buffer(2)]],
                           device const long* metadata [[buffer(3)]],
                           constant ulong& rank [[buffer(4)]],
                           constant long& offset [[buffer(5)]],
                           uint id [[thread_position_in_grid]]) {
  if (id < n) {
    long remaining = id, source = offset;
    for (ulong axis = rank; axis > 0; --axis) {
      const ulong entry = (axis - 1) * 2;
      source += (remaining % metadata[entry]) * metadata[entry + 1];
      remaining /= metadata[entry];
    }
    out[id] = input[source];
  }
}

kernel void concat_bool(device const uchar* input [[buffer(0)]],
                        device uchar* out [[buffer(1)]],
                        constant uint& n [[buffer(2)]],
                        constant ulong& block [[buffer(3)]],
                        constant ulong& output_block [[buffer(4)]],
                        constant ulong& offset [[buffer(5)]],
                        uint id [[thread_position_in_grid]]) {
  if (id < n) out[(id / block) * output_block + offset + id % block] = input[id];
}
