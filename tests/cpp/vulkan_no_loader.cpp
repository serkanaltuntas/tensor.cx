// Subprocess-only interposer: verify that the optional loader is not a DT_NEEDED
// dependency and that both the extension import and CPU execution still work.
#include <dlfcn.h>
#include <cstring>
#include <cstdlib>

extern "C" void* dlopen(const char* file, int flags) noexcept {
  static auto original = reinterpret_cast<void* (*)(const char*, int)>(dlsym(RTLD_NEXT, "dlopen"));
  if (file && std::strcmp(file, "libvulkan.so.1") == 0) {
    const char* fake = std::getenv("TENSORCX_TEST_VULKAN_INCOMPLETE_LOADER");
    return fake && original ? original(fake, flags) : nullptr;
  }
  return original ? original(file, flags) : nullptr;
}
