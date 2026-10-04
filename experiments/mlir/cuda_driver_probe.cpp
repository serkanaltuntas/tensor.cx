// Research-only Runtime/Driver interoperability probe. No tensor.cx backend hooks.
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <set>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

static_assert(sizeof(void*) == 8 && sizeof(CUdeviceptr) == 8 && sizeof(std::uint32_t) == 4);

void driver(CUresult result) {
  if (result == CUDA_SUCCESS) return;
  const char* message = nullptr;
  cuGetErrorString(result, &message);
  throw std::runtime_error(message ? message : "CUDA Driver failure");
}
void runtime(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Primary {
  CUcontext context{};
  explicit Primary(CUdevice device) : device_(device) {
    driver(cuDevicePrimaryCtxRetain(&context, device));
  }
  ~Primary() { cuDevicePrimaryCtxRelease(device_); }
  CUdevice device_;
};
struct ContextScope {
  explicit ContextScope(CUcontext context) { driver(cuCtxPushCurrent(context)); }
  ~ContextScope() { CUcontext popped{}; cuCtxPopCurrent(&popped); }
};
struct Foreign {
  CUcontext context{};
  explicit Foreign(CUdevice device) { driver(cuCtxCreate(&context, 0, device)); }
  ~Foreign() { cuCtxDestroy(context); }
};
struct Module {
  CUmodule module{};
  explicit Module(const char* ptx) { driver(cuModuleLoadData(&module, ptx)); }
  ~Module() { cuModuleUnload(module); }
};
struct Allocation {
  float* pointer{};
  explicit Allocation(std::size_t count) {
    void* allocated = nullptr;
    runtime(cudaMalloc(&allocated, count * sizeof(float)));
    pointer = static_cast<float*>(allocated);
  }
  ~Allocation() { cudaFree(pointer); }
};
CUcontext current() {
  CUcontext context{};
  driver(cuCtxGetCurrent(&context));
  return context;
}

void emit_case(CUfunction function, std::uint32_t size, std::uint32_t threads,
               std::uint32_t block, bool alias) {
  require(threads <= size && block > 0, "invalid probe dimensions");
  const unsigned blocks = threads ? 1 + (threads - 1) / block : 0;
  // Sentinel space covers rounded-up lanes as well as a tail beyond them.
  const std::size_t capacity = std::max<std::size_t>(size, blocks * block) + 8;
  std::vector<float> a(capacity, -1234), b(capacity, -1234), out(capacity, -1234);
  for (unsigned i = 0; i < size; ++i) {
    a[i] = i * 0.25f;
    b[i] = (i % 7) * 0.5f;
    out[i] = -11;
  }
  Allocation da(capacity), db(capacity), dout(capacity);
  runtime(cudaMemcpy(da.pointer, a.data(), capacity * sizeof(float), cudaMemcpyHostToDevice));
  runtime(cudaMemcpy(db.pointer, b.data(), capacity * sizeof(float), cudaMemcpyHostToDevice));
  runtime(cudaMemcpy(dout.pointer, out.data(), capacity * sizeof(float), cudaMemcpyHostToDevice));
  runtime(cudaDeviceSynchronize());
  for (auto* pointer : {da.pointer, db.pointer, dout.pointer}) {
    CUcontext allocation_context{};
    driver(cuPointerGetAttribute(&allocation_context, CU_POINTER_ATTRIBUTE_CONTEXT,
                                reinterpret_cast<CUdeviceptr>(pointer)));
    require(allocation_context == current(), "allocation/launch context mismatch");
  }
  float* output = alias ? da.pointer : dout.pointer;
  void* arguments[] = {&da.pointer, &db.pointer, &output, &threads};
  if (threads) {
    driver(cuLaunchKernel(function, blocks, 1, 1, block, 1, 1, 0, nullptr, arguments, nullptr));
    driver(cuCtxSynchronize());
  }
  runtime(cudaMemcpy(out.data(), output, capacity * sizeof(float), cudaMemcpyDeviceToHost));
  for (std::size_t i = 0; i < capacity; ++i) {
    const float expected = i < threads ? a[i] + b[i] : (alias ? a[i] : (i < size ? -11 : -1234));
    require(out[i] == expected, "native result/suffix/sentinel mismatch");
  }
  // Inputs are unchanged unless explicitly passed as the native output pointer.
  std::vector<float> after(capacity);
  runtime(cudaMemcpy(after.data(), db.pointer, capacity * sizeof(float), cudaMemcpyDeviceToHost));
  require(after == b, "read-only input b mutated");
  if (!alias) {
    runtime(cudaMemcpy(after.data(), da.pointer, capacity * sizeof(float), cudaMemcpyDeviceToHost));
    require(after == a, "read-only input a mutated");
  }
  std::cout << "{\"size\":" << size << ",\"threads\":" << threads
            << ",\"block\":" << block << ",\"alias\":" << (alias ? "true" : "false")
            << ",\"result\":[";
  for (unsigned i = 0; i < size; ++i) {
    if (i) std::cout << ',';
    std::cout << out[i];
  }
  std::cout << "]}";
}

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("expected PTX file");
    std::ifstream input(argv[1]);
    if (!input) throw std::runtime_error("cannot open PTX file");
    const std::string ptx((std::istreambuf_iterator<char>(input)), {});
    driver(cuInit(0));
    runtime(cudaSetDevice(0));
    runtime(cudaFree(nullptr));
    int runtime_version{}, driver_version{};
    runtime(cudaRuntimeGetVersion(&runtime_version));
    driver(cuDriverGetVersion(&driver_version));
    require(runtime_version == 12040, "probe requires CUDA Runtime 12.4");
    CUdevice device{};
    driver(cuDeviceGet(&device, 0));
    Primary primary(device);
    require(current() == primary.context, "Runtime is not using device-0 primary context");
    int major{}, minor{};
    driver(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    driver(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    require(major == 5 && minor == 2, "this probe's hardware acceptance requires sm_52");
    Foreign foreign(device);
    require(current() == foreign.context, "foreign context setup failed");
    // Exercise restoration while unwinding a failing lookup.
    struct MissingEntry {};
    try {
      ContextScope scope(primary.context);
      Module module(ptx.c_str());
      CUfunction missing{};
      require(cuModuleGetFunction(&missing, module.module, "missing_entry") == CUDA_ERROR_NOT_FOUND,
              "missing entry did not return CUDA_ERROR_NOT_FOUND");
      throw MissingEntry{};
    } catch (const MissingEntry&) {
      // The module and pushed context have unwound before checking restoration.
    }
    require(current() == foreign.context, "failure did not restore foreign context");
    {
      ContextScope scope(primary.context);
      Module module(ptx.c_str());
      CUfunction function{};
      driver(cuModuleGetFunction(&function, module.module, "tensorcx_add"));
      std::cout << "{\"device\":\"sm_52\",\"cases\":[";
      bool first = true;
      for (const auto size : {0u, 1u, 255u, 256u, 257u}) {
        for (const auto block : {7u, 256u}) {
          for (bool alias : {false, true}) {
            for (const auto threads : std::set<unsigned>{0u, size / 2, size}) {
              if (!first) std::cout << ',';
              first = false;
              emit_case(function, size, threads, block, alias);
            }
          }
        }
      }
      std::cout << ']';
    }
    require(current() == foreign.context, "success did not restore foreign context");
    std::cout << ",\"context_restored\":true,\"runtime_version\":" << runtime_version
              << ",\"driver_api_version\":" << driver_version << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
