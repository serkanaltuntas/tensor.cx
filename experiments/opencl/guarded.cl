// Standalone probe fixtures: no relaxed math and no runtime backend registration.
#pragma OPENCL FP_CONTRACT OFF
__kernel void guarded_add(__global const float* a, __global const float* b,
                          __global float* out, const uint n) {
  const size_t i = get_global_id(0);
  if (i < n) out[i] = a[i] + b[i];
}
__kernel void guarded_multiply(__global const float* a, __global const float* b,
                               __global float* out, const uint n) {
  const size_t i = get_global_id(0);
  if (i < n) out[i] = a[i] * b[i];
}
