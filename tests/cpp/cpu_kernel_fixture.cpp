#include <cstdint>
struct MemRef {
  float* allocated;
  float* aligned;
  std::int64_t offset, size, stride;
};
#ifndef FIXTURE_BAD_SYMBOL
extern "C" void tensorcx_launch_v1(void** args) {
  auto* input = static_cast<MemRef*>(args[0]);
  auto* output = static_cast<MemRef*>(args[1]);
  const auto n = *static_cast<std::uint32_t*>(args[2]);
  for (std::uint32_t i = 0; i < n; ++i) output->aligned[i] = input->aligned[i] + 1.0f;
}
#endif
extern "C" const char* tensorcx_manifest_v1() {
#ifdef FIXTURE_BAD_ABI
  return "wrong-abi";
#else
  return "tensorcx.cpu.v1|linux-x86_64|llvm-21.1.8|elementwise-f32-v1|ttu|1|2";
#endif
}
