// Standalone OpenCL 1.2 experiment. Not a tensor.cx execution backend.
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
std::uint64_t created = 0, released = 0, release_errors = 0;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void check(cl_int status, const char* operation) {
  if (status != CL_SUCCESS)
    throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
}
template<class T> class Owned {
 public:
  Owned(T value, cl_int (*destroy)(T)) : value_(value), destroy_(destroy) {
    if (value_) ++created;
  }
  Owned(const Owned&) = delete;
  Owned& operator=(const Owned&) = delete;
  Owned(Owned&& other) noexcept
      : value_(std::exchange(other.value_, nullptr)), destroy_(other.destroy_) {}
  ~Owned() {
    if (value_) {
      if (destroy_(value_) == CL_SUCCESS) ++released;
      else ++release_errors;
    }
  }
  operator T() const { return value_; }
 private:
  T value_;
  cl_int (*destroy_)(T);
};
template<class T, class F>
Owned<T> make(F create, cl_int (*destroy)(T), const char* name) {
  cl_int error = CL_SUCCESS;
  Owned<T> result(create(&error), destroy);
  check(error, name);
  require(static_cast<T>(result) != nullptr, "null OpenCL object");
  return result;
}
struct Idle {
  cl_command_queue queue;
  ~Idle() {
    // Do not unwind/free memory if completion cannot be established.
    if (clFinish(queue) != CL_SUCCESS) {
      std::cerr << "unproven OpenCL completion; retaining handles until exit\n";
      std::_Exit(2);
    }
  }
};
std::string quote(const std::string& text) {
  std::string result = "\"";
  const char* hex = "0123456789abcdef";
  for (unsigned char c : text) {
    if (c == '"' || c == '\\') { result += '\\'; result += static_cast<char>(c); }
    else if (c < 32) { result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15]; }
    else result += static_cast<char>(c);
  }
  return result + '"';
}
std::string platform_text(cl_platform_id platform, cl_platform_info field) {
  std::size_t size = 0;
  check(clGetPlatformInfo(platform, field, 0, nullptr, &size), "platform info size");
  std::vector<char> data(size);
  check(clGetPlatformInfo(platform, field, size, data.data(), nullptr), "platform info");
  require(!data.empty() && data.back() == '\0', "invalid platform string");
  return {data.data()};
}
std::string device_text(cl_device_id device, cl_device_info field) {
  std::size_t size = 0;
  check(clGetDeviceInfo(device, field, 0, nullptr, &size), "device info size");
  std::vector<char> data(size);
  check(clGetDeviceInfo(device, field, size, data.data(), nullptr), "device info");
  require(!data.empty() && data.back() == '\0', "invalid device string");
  return {data.data()};
}
template<class T> T device_value(cl_device_id device, cl_device_info field) {
  T result{};
  check(clGetDeviceInfo(device, field, sizeof(result), &result, nullptr), "device capability");
  return result;
}
struct Device { cl_platform_id platform; cl_device_id device; };
Device select(const std::string& target, const std::string& name) {
  require(target == "nvidia-gpu" || target == "pocl-cpu", "unknown execution target");
  cl_uint count = 0;
  check(clGetPlatformIDs(0, nullptr, &count), "enumerate platforms");
  std::vector<cl_platform_id> platforms(count);
  check(clGetPlatformIDs(count, platforms.data(), nullptr), "read platforms");
  std::vector<Device> matches;
  for (auto platform : platforms) {
    const auto platform_name = platform_text(platform, CL_PLATFORM_NAME);
    const auto platform_vendor = platform_text(platform, CL_PLATFORM_VENDOR);
    const bool nvidia = target == "nvidia-gpu";
    if (nvidia ? platform_vendor != "NVIDIA Corporation"
               : platform_name != "Portable Computing Language") continue;
    cl_uint devices_count = 0;
    const auto status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devices_count);
    if (status == CL_DEVICE_NOT_FOUND) continue;
    check(status, "enumerate devices");
    std::vector<cl_device_id> devices(devices_count);
    check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devices_count, devices.data(), nullptr),
          "read devices");
    for (auto device : devices) {
      if (device_text(device, CL_DEVICE_NAME) != name) continue;
      const auto type = device_value<cl_device_type>(device, CL_DEVICE_TYPE);
      if (type != (nvidia ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU)) continue;
      if (nvidia && device_value<cl_uint>(device, CL_DEVICE_VENDOR_ID) != 0x10de) continue;
      matches.push_back({platform, device});
    }
  }
  require(matches.size() == 1, "expected exactly one explicitly named target device");
  const auto device = matches[0].device;
  require(device_value<cl_bool>(device, CL_DEVICE_AVAILABLE), "device unavailable");
  require(device_value<cl_bool>(device, CL_DEVICE_COMPILER_AVAILABLE), "compiler unavailable");
  require(device_value<std::size_t>(device, CL_DEVICE_MAX_WORK_GROUP_SIZE) >= 64,
          "insufficient work-group capacity");
  return matches[0];
}
void print_bits(const std::vector<float>& data) {
  std::cout << '[';
  bool first = true;
  for (float value : data) {
    if (!first) std::cout << ',';
    first = false;
    std::cout << std::bit_cast<std::uint32_t>(value);
  }
  std::cout << ']';
}
void run_case(cl_context context, cl_command_queue queue, cl_kernel kernel,
              int cycle, int size, int active, const char* operation, bool alias, bool& first) {
  const std::size_t groups = std::max(1, (active + 63) / 64);
  const std::size_t count = std::max<std::size_t>(size, groups * 64) + 8;
  std::vector<float> a(count, -1234), b(count, -1234), out(count, -1234);
  for (int i = 0; i < size; ++i) {
    a[i] = static_cast<float>((i % 37) - 18) * 0.25f;
    b[i] = static_cast<float>((i % 7) - 3) * 0.5f;
    out[i] = -11;
  }
  const auto bytes = count * sizeof(float);
  auto buffer = [&] { return make<cl_mem>([&](auto* e) {
    return clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, nullptr, e);
  }, clReleaseMemObject, "create buffer"); };
  auto a_buffer = buffer(), b_buffer = buffer();
  auto out_buffer = alias ? Owned<cl_mem>(nullptr, clReleaseMemObject) : buffer();
  Idle idle{queue};  // Unwinds before buffer owners, including on enqueue/read failures.
  auto write = [&](cl_mem memory, const auto& values) {
    check(clEnqueueWriteBuffer(queue, memory, CL_TRUE, 0, bytes, values.data(), 0, nullptr, nullptr),
          "upload buffer");
  };
  write(a_buffer, a); write(b_buffer, b);
  if (!alias) write(out_buffer, out);
  const std::array<cl_mem, 3> arguments{a_buffer, b_buffer, alias ? a_buffer : out_buffer};
  for (cl_uint index = 0; index < arguments.size(); ++index)
    check(clSetKernelArg(kernel, index, sizeof(cl_mem), &arguments[index]), "set buffer argument");
  const auto n = static_cast<cl_uint>(active);
  check(clSetKernelArg(kernel, 3, sizeof(n), &n), "set bound argument");
  const std::size_t global = groups * 64, local = 64;
  // Even n=0 dispatches one guarded group; sentinel readback proves no writes.
  check(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
        "launch guarded kernel");
  check(clFinish(queue), "kernel completion");
  auto read = [&](cl_mem memory, auto& values) {
    check(clEnqueueReadBuffer(queue, memory, CL_TRUE, 0, bytes, values.data(), 0, nullptr, nullptr),
          "read buffer");
  };
  read(arguments[2], out); read(a_buffer, a); read(b_buffer, b);
  if (!first) std::cout << ',';
  first = false;
  std::cout << "{\"cycle\":" << cycle << ",\"size\":" << size << ",\"active\":" << active
            << ",\"operation\":" << quote(operation) << ",\"alias\":" << (alias ? "true" : "false")
            << ",\"groups\":" << groups << ",\"result\":";
  print_bits(out); std::cout << ",\"a_after\":"; print_bits(a);
  std::cout << ",\"b_after\":"; print_bits(b); std::cout << '}';
}
void cycle(Device selected, const std::string& source, int index, bool& first) {
  const cl_context_properties properties[]{CL_CONTEXT_PLATFORM,
    reinterpret_cast<cl_context_properties>(selected.platform), 0};
  auto context = make<cl_context>([&](auto* e) {
    return clCreateContext(properties, 1, &selected.device, nullptr, nullptr, e);
  }, clReleaseContext, "create context");
  auto queue = make<cl_command_queue>([&](auto* e) {
    return clCreateCommandQueue(context, selected.device, 0, e);
  }, clReleaseCommandQueue, "create synchronous queue");
  const char* text = source.c_str();
  const auto length = source.size();
  auto program = make<cl_program>([&](auto* e) {
    return clCreateProgramWithSource(context, 1, &text, &length, e);
  }, clReleaseProgram, "create program");
  const auto status = clBuildProgram(program, 1, &selected.device, "-cl-std=CL1.2", nullptr, nullptr);
  if (status != CL_SUCCESS) {
    std::size_t count = 0;
    check(clGetProgramBuildInfo(program, selected.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &count),
          "build log size");
    std::vector<char> log(count + 1, 0);
    check(clGetProgramBuildInfo(program, selected.device, CL_PROGRAM_BUILD_LOG, count, log.data(), nullptr),
          "build log");
    throw std::runtime_error("OpenCL build failed: " + std::string(log.data()));
  }
  auto kernel = [&](const char* name) { return make<cl_kernel>([&](auto* e) {
    return clCreateKernel(program, name, e);
  }, clReleaseKernel, "create kernel"); };
  auto add = kernel("guarded_add"), multiply = kernel("guarded_multiply");
  for (auto k : {static_cast<cl_kernel>(add), static_cast<cl_kernel>(multiply)}) {
    std::size_t maximum = 0;
    check(clGetKernelWorkGroupInfo(k, selected.device, CL_KERNEL_WORK_GROUP_SIZE,
                                  sizeof(maximum), &maximum, nullptr), "kernel work-group limit");
    require(maximum >= 64, "insufficient compiled-kernel work-group capacity");
  }
  for (int size : {0, 1, 63, 64, 65, 255, 256, 257, 1025}) {
    std::vector<int> bounds{0, size / 2, size};
    std::sort(bounds.begin(), bounds.end());
    bounds.erase(std::unique(bounds.begin(), bounds.end()), bounds.end());
    for (int active : bounds) for (bool alias : {false, true}) {
      run_case(context, queue, add, index, size, active, "add", alias, first);
      run_case(context, queue, multiply, index, size, active, "multiply", alias, first);
    }
  }
  check(clFinish(queue), "cycle completion");
}
}  // namespace
int main(int argc, char** argv) {
  try {
    require(argc == 4, "usage: opencl-probe nvidia-gpu|pocl-cpu exact-device-name fixture.cl");
    const std::string target = argv[1];
    auto selected = select(target, argv[2]);
    std::ifstream file(argv[3], std::ios::binary);
    require(static_cast<bool>(file), "missing OpenCL fixture");
    const std::string source{std::istreambuf_iterator<char>(file), {}};
    require(!source.empty(), "empty OpenCL fixture");
    std::cout << "{\"target\":" << quote(target) << ",\"platform_name\":"
              << quote(platform_text(selected.platform, CL_PLATFORM_NAME))
              << ",\"platform_vendor\":" << quote(platform_text(selected.platform, CL_PLATFORM_VENDOR))
              << ",\"platform_version\":" << quote(platform_text(selected.platform, CL_PLATFORM_VERSION));
    for (const auto& [name, field] : std::array<std::pair<const char*, cl_device_info>, 6>{{
      {"device_name", CL_DEVICE_NAME}, {"device_vendor", CL_DEVICE_VENDOR},
      {"device_version", CL_DEVICE_VERSION}, {"driver_version", CL_DRIVER_VERSION},
      {"opencl_c_version", CL_DEVICE_OPENCL_C_VERSION}, {"extensions", CL_DEVICE_EXTENSIONS}}})
      std::cout << ',' << quote(name) << ':' << quote(device_text(selected.device, field));
    std::cout << ",\"device_type\":" << device_value<cl_device_type>(selected.device, CL_DEVICE_TYPE)
              << ",\"vendor_id\":" << device_value<cl_uint>(selected.device, CL_DEVICE_VENDOR_ID)
              << ",\"compiler_available\":true,\"device_available\":true,\"build_options\":\"-cl-std=CL1.2\""
              << ",\"max_work_group_size\":" << device_value<std::size_t>(selected.device, CL_DEVICE_MAX_WORK_GROUP_SIZE)
              << ",\"single_fp_config\":" << device_value<cl_device_fp_config>(selected.device, CL_DEVICE_SINGLE_FP_CONFIG)
              << ",\"cycles\":2,\"cases\":[";
    bool first = true;
    for (int i = 0; i < 2; ++i) cycle(selected, source, i, first);
    require(created == released && release_errors == 0, "OpenCL cleanup failed");
    std::cout << "],\"owned_handles_created\":" << created << ",\"owned_handles_released\":" << released
              << ",\"release_errors\":" << release_errors << ",\"status\":\"passed\"}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
