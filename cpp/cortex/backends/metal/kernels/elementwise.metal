#include <metal_stdlib>

using namespace metal;

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

kernel void matmul_f32(device const float* lhs [[buffer(0)]],
                       device const float* rhs [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       constant uint& m [[buffer(3)]],
                       constant uint& k [[buffer(4)]],
                       constant uint& n [[buffer(5)]],
                       uint id [[thread_position_in_grid]]) {
  const uint total = m * n;
  if (id >= total) {
    return;
  }

  const uint row = id / n;
  const uint col = id - row * n;
  float sum = 0.0f;
  for (uint inner = 0; inner < k; ++inner) {
    sum += lhs[row * k + inner] * rhs[inner * n + col];
  }
  out[id] = sum;
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
