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
    out[id] = lhs[id] + rhs[id];
  }
}

kernel void mul_i32(device const int* lhs [[buffer(0)]],
                    device const int* rhs [[buffer(1)]],
                    device int* out [[buffer(2)]],
                    constant uint& n [[buffer(3)]],
                    uint id [[thread_position_in_grid]]) {
  if (id < n) {
    out[id] = lhs[id] * rhs[id];
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
