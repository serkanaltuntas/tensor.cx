#include <Python.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tensorcx/backends/cpu/cpu_backend.h"
#include "tensorcx/backends/cpu/cpu_tensor.h"
#include "tensorcx/backends/cpu/cpu_kernel.h"
#include "tensorcx/backends/null/null_backend.h"
#include "tensorcx/core/dtype.h"
#include "tensorcx/core/shape.h"
#include "tensorcx/core/status.h"
#include "tensorcx/core/profiling.h"

#if TENSORCX_ENABLE_METAL
#include "tensorcx/backends/metal/metal_backend.h"
#include "tensorcx/backends/metal/metal_library.h"
#include "tensorcx/backends/metal/metal_tensor.h"
#endif

#if TENSORCX_ENABLE_CUDA
#include "tensorcx/backends/cuda/cuda_backend.h"
#include "tensorcx/backends/cuda/cuda_kernel.h"
#include "tensorcx/backends/cuda/cuda_memory_pool.h"
#endif

#if TENSORCX_ENABLE_VULKAN
#include "tensorcx/backends/vulkan/vulkan_backend.h"
#include "tensorcx/backends/vulkan/vulkan_context.h"
#include "tensorcx/backends/vulkan/vulkan_memory_pool.h"
#include "tensorcx/backends/vulkan/vulkan_kernel.h"
#endif

#ifndef TENSORCX_RUNTIME_VERSION
#define TENSORCX_RUNTIME_VERSION "0+unknown"
#endif

namespace nb = nanobind;

namespace {

using tensorcx::DType;
using tensorcx::MatmulPreference;
using tensorcx::OpDesc;
using tensorcx::OpKind;
using tensorcx::Shape;
using tensorcx::cpu::CpuTensor;

bool is_bool_like(nb::handle item) {
  if (PyBool_Check(item.ptr())) {
    return true;
  }
  const PyTypeObject* type = Py_TYPE(item.ptr());
  if (type == nullptr || type->tp_name == nullptr) {
    return false;
  }
  const std::string_view type_name(type->tp_name);
  return type_name == "numpy.bool" || type_name == "numpy.bool_";
}

DType parse_dtype(nb::handle dtype, DType inferred) {
  if (dtype.is_none()) {
    return inferred;
  }

  if (!nb::isinstance<nb::str>(dtype)) {
    throw std::invalid_argument("unsupported dtype: expected float32, int32 or bool");
  }
  const std::string name = nb::cast<std::string>(dtype);
  if (name == "float32") {
    return DType::kFloat32;
  }
  if (name == "bool") return DType::kBool;
  if (name == "int32") {
    return DType::kInt32;
  }
  throw std::invalid_argument("unsupported dtype: expected float32, int32 or bool");
}

bool is_index_like(nb::handle item) {
  return PyBool_Check(item.ptr()) || PyIndex_Check(item.ptr());
}

// Keep invalid shape input inside the public ValueError taxonomy instead of
// letting nanobind expose std::bad_cast as RuntimeError. Use Python's index
// protocol so NumPy integer scalars behave like Python ints, while bools stay
// rejected as shape dimensions.
tensorcx::Dim cast_dim_or_throw(nb::handle item) {
  if (is_bool_like(item)) {
    throw std::invalid_argument("shape dimensions must be integers");
  }

  PyObject* index_value = PyNumber_Index(item.ptr());
  if (index_value == nullptr) {
    PyErr_Clear();
    throw std::invalid_argument("shape dimensions must be integers");
  }
  nb::object index = nb::steal<nb::object>(index_value);
  try {
    return nb::cast<tensorcx::Dim>(index);
  } catch (const std::exception&) {
    throw std::invalid_argument("shape dimension is out of range");
  }
}

void validate_cpu_device(const std::string& device) {
  if (device != "cpu") {
    throw std::invalid_argument("native CPU factory only accepts device='cpu'");
  }
}

// nanobind raises std::bad_cast (-> RuntimeError) when a Python int exceeds the
// int32 range. Translate that into a clear ValueError matching the error
// taxonomy instead of leaking an opaque "std::bad_cast".
std::int32_t cast_int32_or_throw(nb::handle item) {
  if (is_bool_like(item)) return PyObject_IsTrue(item.ptr()) != 0;
  try {
    return nb::cast<std::int32_t>(item);
  } catch (const std::exception&) {
    throw std::invalid_argument("integer value is out of range for int32 or is not an integer");
  }
}

float cast_float32_or_throw(nb::handle item) {
  if (is_bool_like(item)) return PyObject_IsTrue(item.ptr()) != 0;
  try {
    return nb::cast<float>(item);
  } catch (const std::exception&) {
    throw std::invalid_argument("float value is not convertible to float32");
  }
}

std::uint32_t cast_uint32_or_throw(nb::handle item, const char* message) {
  if (PyBool_Check(item.ptr())) {
    throw std::invalid_argument(message);
  }

  PyObject* index_value = PyNumber_Index(item.ptr());
  if (index_value == nullptr) {
    PyErr_Clear();
    throw std::invalid_argument(message);
  }
  nb::object index = nb::steal<nb::object>(index_value);
  const auto value = PyLong_AsUnsignedLongLong(index.ptr());
  if (PyErr_Occurred()) {
    PyErr_Clear();
    throw std::invalid_argument(message);
  }
  if (value > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument(message);
  }
  return static_cast<std::uint32_t>(value);
}

Shape parse_shape(nb::handle shape) {
  if (is_index_like(shape)) {
    return Shape{cast_dim_or_throw(shape)};
  }

  Shape result;
  try {
    for (nb::handle item : nb::iter(shape)) {
      result.push_back(cast_dim_or_throw(item));
    }
  } catch (const std::invalid_argument&) {
    throw;
  } catch (const std::exception&) {
    throw std::invalid_argument("shape must be an int or an iterable of ints");
  }
  return result;
}

std::uint8_t cast_bool_or_throw(nb::handle value) {
  if (!PyNumber_Check(value.ptr()) || PyComplex_Check(value.ptr()))
    throw std::invalid_argument("bool data must contain real numbers or booleans");
  const int truth = PyObject_IsTrue(value.ptr());
  if (truth < 0) throw nb::python_error();
  return truth != 0;
}

CpuTensor tensor_from_sequence(nb::handle data, nb::handle dtype, const std::string& device) {
  tensorcx::ProfileScope profile(tensorcx::ProfileCategory::kCopyAllocate, "sequence_import", "cpu");
  validate_cpu_device(device);

  std::vector<nb::object> items;
  bool saw_float = false;
  for (nb::handle item : nb::iter(data)) {
    if (nb::isinstance<nb::list>(item) || nb::isinstance<nb::tuple>(item)) {
      throw std::invalid_argument(
          "tensor() native factory expects a flat numeric sequence; build "
          "higher-rank tensors through tensorcx.tensor()");
    }
    items.emplace_back(nb::borrow<nb::object>(item));
    if (nb::isinstance<nb::float_>(item)) {
      saw_float = true;
    }
  }

  const bool all_bool = !items.empty() && std::all_of(items.begin(), items.end(), [](const nb::object& item) { return is_bool_like(item); });
  if (dtype.is_none() && !all_bool && std::any_of(items.begin(), items.end(), [](const nb::object& item) { return is_bool_like(item); }))
    throw std::invalid_argument("mixed boolean and numeric data requires an explicit dtype");
  const DType inferred = all_bool ? DType::kBool : saw_float ? DType::kFloat32 : DType::kInt32;
  const DType actual_dtype = parse_dtype(dtype, inferred);
  Shape shape{static_cast<tensorcx::Dim>(items.size())};

  switch (actual_dtype) {
    case DType::kBool: {
      std::vector<std::uint8_t> values;
      values.reserve(items.size());
      for (const auto& item : items) values.push_back(cast_bool_or_throw(item));
      return CpuTensor(std::move(shape), std::move(values));
    }
    case DType::kFloat32: {
      std::vector<float> values;
      values.reserve(items.size());
      for (const nb::object& item : items) {
        values.push_back(cast_float32_or_throw(item));
      }
      return CpuTensor(std::move(shape), std::move(values));
    }
    case DType::kInt32: {
      std::vector<std::int32_t> values;
      values.reserve(items.size());
      for (const nb::object& item : items) {
        values.push_back(cast_int32_or_throw(item));
      }
      return CpuTensor(std::move(shape), std::move(values));
    }
  }
  throw std::invalid_argument("unsupported dtype");
}

CpuTensor tensor_from_flat_sequence(
    nb::handle data,
    nb::handle shape,
    nb::handle dtype,
    const std::string& device) {
  tensorcx::ProfileScope profile(tensorcx::ProfileCategory::kCopyAllocate, "flat_sequence_import", "cpu");
  validate_cpu_device(device);

  const Shape parsed_shape = parse_shape(shape);
  const std::int64_t expected_size = tensorcx::numel(parsed_shape);
  const DType actual_dtype = parse_dtype(dtype, DType::kFloat32);

  // The shape fixes the element count up front, so stop consuming the iterable
  // as soon as it yields one element too many. Without this bound an infinite
  // generator would be drained until the process runs out of memory before the
  // length check could ever fire.
  const auto reject_excess_length = [&](std::size_t collected) {
    if (static_cast<std::int64_t>(collected) >= expected_size) {
      throw std::invalid_argument("tensor data length does not match shape");
    }
  };

  switch (actual_dtype) {
    case DType::kBool: {
      std::vector<std::uint8_t> values;
      for (nb::handle item : nb::iter(data)) {
        reject_excess_length(values.size());
        values.push_back(cast_bool_or_throw(item));
      }
      if (static_cast<std::int64_t>(values.size()) != expected_size)
        throw std::invalid_argument("tensor data length does not match shape");
      return CpuTensor(parsed_shape, std::move(values));
    }
    case DType::kFloat32: {
      std::vector<float> values;
      for (nb::handle item : nb::iter(data)) {
        reject_excess_length(values.size());
        values.push_back(cast_float32_or_throw(item));
      }
      if (static_cast<std::int64_t>(values.size()) != expected_size) {
        throw std::invalid_argument("tensor data length does not match shape");
      }
      return CpuTensor(parsed_shape, std::move(values));
    }
    case DType::kInt32: {
      std::vector<std::int32_t> values;
      for (nb::handle item : nb::iter(data)) {
        reject_excess_length(values.size());
        values.push_back(cast_int32_or_throw(item));
      }
      if (static_cast<std::int64_t>(values.size()) != expected_size) {
        throw std::invalid_argument("tensor data length does not match shape");
      }
      return CpuTensor(parsed_shape, std::move(values));
    }
  }
  throw std::invalid_argument("unsupported dtype");
}

nb::tuple shape_tuple(const Shape& shape) {
  nb::tuple result = nb::steal<nb::tuple>(PyTuple_New(static_cast<Py_ssize_t>(shape.size())));
  for (std::size_t i = 0; i < shape.size(); ++i) {
    PyTuple_SET_ITEM(result.ptr(), static_cast<Py_ssize_t>(i), PyLong_FromLongLong(shape[i]));
  }
  return result;
}

// Build a NumPy array that owns a private copy of the contiguous tensor data.
// Copies once via memcpy into a heap buffer whose lifetime is tied to the array
// through a capsule, instead of boxing every element into a Python list. This is
// the nb::ndarray NumPy bridge §5.6 chose nanobind for, and keeps Tensor.numpy()
// O(n) memory copies rather than O(n) PyObject allocations on 1M/16M tensors.
template <typename T>
nb::object export_numpy_copy(std::unique_ptr<T[]> owned, std::size_t count, const Shape& shape) {
  nb::capsule owner(owned.get(), [](void* ptr) noexcept { delete[] static_cast<T*>(ptr); });
  T* data = owned.release();
  // NumPy 2's baseline supports 64 axes. Above that, let the installed NumPy
  // validate its own limit through reshape. Creating a high-rank nanobind
  // ndarray directly can lose the shape or hit its unrecoverable rank check.
  if (shape.size() > 64) {
    nb::object flat = nb::cast(nb::ndarray<nb::numpy, T>(data, 1, &count, owner));
    return flat.attr("reshape")(shape_tuple(shape));
  }
  std::vector<std::size_t> dims(shape.begin(), shape.end());
  return nb::cast(nb::ndarray<nb::numpy, T>(data, dims.size(), dims.data(), owner));
}

template <typename T>
nb::object make_numpy_array(std::span<const T> data, const Shape& shape) {
  std::unique_ptr<T[]> owned(new T[data.size()]);
  if (!data.empty()) {
    std::memcpy(owned.get(), data.data(), data.size() * sizeof(T));
  }
  return export_numpy_copy(std::move(owned), data.size(), shape);
}

nb::object make_bool_numpy(const CpuTensor& tensor) {
  const auto& data = tensor.bool_data();
  std::unique_ptr<bool[]> owned(new bool[data.size()]);
  for (std::size_t i = 0; i < data.size(); ++i) owned[i] = data[i] != 0;
  return export_numpy_copy(std::move(owned), data.size(), tensor.shape());
}

nb::object tensor_to_numpy(const CpuTensor& tensor) {
  tensorcx::ProfileScope profile(tensorcx::ProfileCategory::kCopyAllocate, "numpy_export", "cpu", tensor.buffer()->nbytes());
  switch (tensor.dtype()) {
    case DType::kBool: return make_bool_numpy(tensor);
    case DType::kFloat32:
      return make_numpy_array<float>(tensor.float_data(), tensor.shape());
    case DType::kInt32:
      return make_numpy_array<std::int32_t>(tensor.int32_data(), tensor.shape());
  }
  throw std::invalid_argument("unsupported dtype");
}

std::vector<std::uint8_t> bytes_to_vector(nb::bytes data) {
  char* buffer = nullptr;
  Py_ssize_t size = 0;
  if (PyBytes_AsStringAndSize(data.ptr(), &buffer, &size) != 0) {
    throw std::invalid_argument("metallib must be bytes");
  }
  if (size == 0) {
    return {};
  }
  const auto* first = reinterpret_cast<const std::uint8_t*>(buffer);
  return std::vector<std::uint8_t>(first, first + size);
}

std::string bytes_to_string(nb::bytes data) {
  char* buffer = nullptr;
  Py_ssize_t size = 0;
  if (PyBytes_AsStringAndSize(data.ptr(), &buffer, &size) != 0) {
    throw std::invalid_argument("metallib must be bytes");
  }
  return std::string(buffer, static_cast<std::size_t>(size));
}

// Run backend work with the GIL released so synchronous GPU waits and large
// CPU loops do not stall other Python threads. Everything inside `fn` must be
// pure C++: no Python object may be created, copied, or destroyed while the
// GIL is released. The native layer this calls into is thread-safe (stateless
// backends; mutex-guarded Metal pipeline caches; thread-safe MTLCommandQueue).
template <typename Fn>
auto without_gil(Fn&& fn) {
  nb::gil_scoped_release released;
  return fn();
}

void throw_status(const tensorcx::Status& status) {
  switch (status.code()) {
    case tensorcx::StatusCode::kInvalidArgument:
      throw std::invalid_argument(status.message());
    case tensorcx::StatusCode::kUnavailable:
    case tensorcx::StatusCode::kInternal:
      throw std::runtime_error(status.message());
    case tensorcx::StatusCode::kOk:
      break;
  }
  throw std::runtime_error(status.message());
}

CpuTensor binary_op(const CpuTensor& lhs, const CpuTensor& rhs, OpKind kind) {
  tensorcx::cpu::CpuBackend backend;
  std::array<tensorcx::Tensor, 2> inputs{
      tensorcx::cpu::to_core_tensor(lhs),
      tensorcx::cpu::to_core_tensor(rhs),
  };
  std::array<tensorcx::Tensor, 1> outputs{};
  const tensorcx::BackendExecution execution{
      tensorcx::BackendOpClass::kPrimitive,
      OpDesc{kind},
      std::span<const tensorcx::Tensor>(inputs.data(), inputs.size()),
      std::span<tensorcx::Tensor>(outputs.data(), outputs.size()),
      std::nullopt,
      std::nullopt,
  };
  const auto status = without_gil([&] { return backend.execute(execution); });
  if (!status.ok()) {
    throw_status(status);
  }
  return tensorcx::cpu::from_core_tensor(outputs[0]);
}

CpuTensor cpu_single_input_backend_op(const CpuTensor& input, const OpDesc& op) {
  tensorcx::cpu::CpuBackend backend;
  std::array<tensorcx::Tensor, 1> inputs{
      tensorcx::cpu::to_core_tensor(input),
  };
  std::array<tensorcx::Tensor, 1> outputs{};
  const tensorcx::BackendExecution execution{
      tensorcx::BackendOpClass::kPrimitive,
      op,
      std::span<const tensorcx::Tensor>(inputs.data(), inputs.size()),
      std::span<tensorcx::Tensor>(outputs.data(), outputs.size()),
      std::nullopt,
      std::nullopt,
  };
  const auto status = without_gil([&] { return backend.execute(execution); });
  if (!status.ok()) {
    throw_status(status);
  }
  return tensorcx::cpu::from_core_tensor(outputs[0]);
}

CpuTensor unary_op(const CpuTensor& input, OpKind kind) {
  return cpu_single_input_backend_op(input, OpDesc{kind});
}

CpuTensor axis_unary_op(const CpuTensor& input, OpKind kind, std::int64_t axis) {
  return cpu_single_input_backend_op(input, OpDesc{kind, axis});
}

CpuTensor norm_op(const CpuTensor& input, OpKind kind, std::int64_t axis, double epsilon) {
  return cpu_single_input_backend_op(input, OpDesc{kind, axis, epsilon});
}

CpuTensor reduction_op(const CpuTensor& input, OpKind kind, std::int64_t axis) {
  return cpu_single_input_backend_op(input, OpDesc{kind, axis});
}

CpuTensor fill_op(Shape shape, DType dtype, double value) {
  tensorcx::cpu::CpuBackend backend;
  static_cast<void>(tensorcx::numel(shape));
  const Shape strides = tensorcx::contiguous_strides(shape);
  std::array<tensorcx::Tensor, 1> outputs{tensorcx::Tensor{
      dtype,
      std::move(shape),
      strides,
      tensorcx::Device{"cpu", 0},
      nullptr,
      0,
  }};
  OpDesc op{OpKind::kFill};
  op.scalar_value = value;
  const tensorcx::BackendExecution execution{
      tensorcx::BackendOpClass::kPrimitive,
      op,
      std::span<const tensorcx::Tensor>(),
      std::span<tensorcx::Tensor>(outputs.data(), outputs.size()),
      std::nullopt,
      std::nullopt,
  };
  const auto status = without_gil([&] { return backend.execute(execution); });
  if (!status.ok()) {
    throw_status(status);
  }
  return tensorcx::cpu::from_core_tensor(outputs[0]);
}

CpuTensor matmul_cpu(const CpuTensor& lhs, const CpuTensor& rhs, const std::string& backend) {
  if (backend != "auto" && backend != "cpu" && backend != "reference") {
    throw std::invalid_argument("CPU matmul only supports backend='auto', 'cpu', or 'reference'");
  }
  return binary_op(lhs, rhs, OpKind::kMatmul);
}

template <typename T>
T unwrap(tensorcx::Expected<T> result) {
  if (!result) throw_status(result.status());
  return result.move_value();
}

struct BackendRoute {
  std::string name;
  std::function<bool()> is_available;
  std::function<std::string()> device_name;
  std::function<nb::object(Shape, DType, double)> fill;
  std::function<nb::list()> matmul_backends;
};

bool cpu_backend_available() { return true; }

std::string cpu_backend_device_name() { return "CPU"; }

nb::object cpu_backend_fill(Shape shape, DType dtype, double value) {
  return nb::cast(fill_op(std::move(shape), dtype, value));
}

nb::list cpu_backend_matmul_backends() {
  nb::list result;
  result.append("auto");
  result.append("cpu");
  result.append("reference");
  return result;
}

#if TENSORCX_ENABLE_METAL
bool metal_backend_available() { return tensorcx::metal::available(); }

std::string metal_backend_device_name() {
  const auto names = tensorcx::metal::devices();
  if (!names.empty()) {
    return names.front();
  }
  throw std::invalid_argument("device is not available: metal");
}

nb::object metal_backend_fill(Shape shape, DType dtype, double value) {
  tensorcx::metal::MetalBackend backend;
  static_cast<void>(tensorcx::numel(shape));
  const Shape strides = tensorcx::contiguous_strides(shape);
  std::array<tensorcx::Tensor, 1> outputs{tensorcx::Tensor{
      dtype,
      shape,
      strides,
      tensorcx::Device{"metal", 0},
      nullptr,
      0,
  }};
  OpDesc op{OpKind::kFill};
  op.scalar_value = value;
  const tensorcx::BackendExecution execution{
      tensorcx::BackendOpClass::kPrimitive,
      op,
      std::span<const tensorcx::Tensor>(),
      std::span<tensorcx::Tensor>(outputs.data(), outputs.size()),
      std::nullopt,
      std::nullopt,
  };
  const tensorcx::Status status = without_gil([&] { return backend.execute(execution); });
  if (!status.ok()) {
    throw_status(status);
  }
  return nb::cast(tensorcx::metal::from_core_tensor(outputs[0]));
}

tensorcx::metal::MetalTensor binary_op(
    const tensorcx::metal::MetalTensor& lhs,
    const tensorcx::metal::MetalTensor& rhs,
    OpKind kind) {
  tensorcx::metal::MetalBackend backend;
  std::array<tensorcx::Tensor, 2> inputs{
      tensorcx::metal::to_core_tensor(lhs),
      tensorcx::metal::to_core_tensor(rhs),
  };
  std::array<tensorcx::Tensor, 1> outputs{};
  const tensorcx::BackendExecution execution{
      tensorcx::BackendOpClass::kPrimitive,
      OpDesc{kind},
      std::span<const tensorcx::Tensor>(inputs.data(), inputs.size()),
      std::span<tensorcx::Tensor>(outputs.data(), outputs.size()),
      std::nullopt,
      std::nullopt,
  };
  const tensorcx::Status status = without_gil([&] { return backend.execute(execution); });
  if (!status.ok()) {
    throw_status(status);
  }
  return tensorcx::metal::from_core_tensor(outputs[0]);
}

tensorcx::metal::MetalTensor metal_single_input_backend_op(
    const tensorcx::metal::MetalTensor& input,
    const OpDesc& op) {
  tensorcx::metal::MetalBackend backend;
  std::array<tensorcx::Tensor, 1> inputs{
      tensorcx::metal::to_core_tensor(input),
  };
  std::array<tensorcx::Tensor, 1> outputs{};
  const tensorcx::BackendExecution execution{
      tensorcx::BackendOpClass::kPrimitive,
      op,
      std::span<const tensorcx::Tensor>(inputs.data(), inputs.size()),
      std::span<tensorcx::Tensor>(outputs.data(), outputs.size()),
      std::nullopt,
      std::nullopt,
  };
  const tensorcx::Status status = without_gil([&] { return backend.execute(execution); });
  if (!status.ok()) {
    throw_status(status);
  }
  return tensorcx::metal::from_core_tensor(outputs[0]);
}

tensorcx::metal::MetalTensor unary_op(const tensorcx::metal::MetalTensor& input, OpKind kind) {
  return metal_single_input_backend_op(input, OpDesc{kind});
}

MatmulPreference parse_metal_matmul_preference(const std::string& backend) {
  if (backend == "auto") {
    return MatmulPreference::kAuto;
  }
  if (backend == "custom") {
    return MatmulPreference::kCustom;
  }
  if (backend == "optimized") {
    return MatmulPreference::kOptimized;
  }
  throw std::invalid_argument("unsupported Metal matmul backend: " + backend);
}

tensorcx::metal::MetalTensor metal_matmul_backend_op(
    const tensorcx::metal::MetalTensor& lhs,
    const tensorcx::metal::MetalTensor& rhs,
    MatmulPreference preference) {
  tensorcx::metal::MetalBackend backend;
  std::array<tensorcx::Tensor, 2> inputs{
      tensorcx::metal::to_core_tensor(lhs),
      tensorcx::metal::to_core_tensor(rhs),
  };
  std::array<tensorcx::Tensor, 1> outputs{};
  OpDesc op{OpKind::kMatmul};
  op.matmul_preference = preference;
  const tensorcx::BackendExecution execution{
      tensorcx::BackendOpClass::kPrimitive,
      op,
      std::span<const tensorcx::Tensor>(inputs.data(), inputs.size()),
      std::span<tensorcx::Tensor>(outputs.data(), outputs.size()),
      std::nullopt,
      std::nullopt,
  };
  const tensorcx::Status status = without_gil([&] { return backend.execute(execution); });
  if (!status.ok()) {
    throw_status(status);
  }
  return tensorcx::metal::from_core_tensor(outputs[0]);
}

nb::list metal_backend_matmul_backends() {
  nb::list result;
  result.append("auto");
  result.append("custom");
#if TENSORCX_ENABLE_MPSGRAPH
  result.append("optimized");
#endif
  return result;
}
#endif

#if TENSORCX_ENABLE_CUDA
bool cuda_backend_available() { return tensorcx::cuda::available(); }
std::string cuda_backend_device_name() { return unwrap(tensorcx::cuda::device_name()); }

nb::object cuda_backend_fill(Shape shape, DType dtype, double value) {
  tensorcx::cuda::CudaBackend backend;
  const Shape strides = tensorcx::contiguous_strides(shape);
  std::array<tensorcx::Tensor, 1> outputs{tensorcx::Tensor{
      dtype, std::move(shape), strides, {"cuda", 0}, nullptr, 0}};
  OpDesc op{OpKind::kFill};
  op.scalar_value = value;
  const tensorcx::BackendExecution execution{
      tensorcx::BackendOpClass::kPrimitive, op, {}, outputs, std::nullopt, std::nullopt};
  const auto status = without_gil([&] { return backend.execute(execution); });
  if (!status.ok()) throw_status(status);
  return nb::cast(unwrap(tensorcx::cuda::from_core_tensor(outputs[0])));
}

tensorcx::cuda::CudaTensor binary_op(const tensorcx::cuda::CudaTensor& lhs,
                                  const tensorcx::cuda::CudaTensor& rhs, OpKind kind) {
  tensorcx::cuda::CudaBackend backend;
  std::array<tensorcx::Tensor, 2> inputs{
      tensorcx::cuda::to_core_tensor(lhs), tensorcx::cuda::to_core_tensor(rhs)};
  std::array<tensorcx::Tensor, 1> outputs{};
  const tensorcx::BackendExecution execution{
      tensorcx::BackendOpClass::kPrimitive, OpDesc{kind}, inputs, outputs,
      std::nullopt, std::nullopt};
  const auto status = without_gil([&] { return backend.execute(execution); });
  if (!status.ok()) throw_status(status);
  return unwrap(tensorcx::cuda::from_core_tensor(outputs[0]));
}

tensorcx::cuda::CudaTensor cuda_primitive(const tensorcx::cuda::CudaTensor& input, OpDesc op,
                                      const tensorcx::cuda::CudaTensor* right=nullptr) {
  tensorcx::cuda::CudaBackend backend;
  std::array<tensorcx::Tensor,2> inputs{tensorcx::cuda::to_core_tensor(input),{}};
  if(right)inputs[1]=tensorcx::cuda::to_core_tensor(*right);
  std::array<tensorcx::Tensor,1> outputs{};
  const tensorcx::BackendExecution execution{tensorcx::BackendOpClass::kPrimitive,op,
      std::span<const tensorcx::Tensor>(inputs.data(),right?2:1),outputs,std::nullopt,std::nullopt};
  const auto status=without_gil([&]{return backend.execute(execution);});
  if(!status.ok())throw_status(status);
  return unwrap(tensorcx::cuda::from_core_tensor(outputs[0]));
}
nb::list cuda_backend_matmul_backends() {
  nb::list result;result.append("auto");result.append("custom");return result;
}
#endif

#if TENSORCX_ENABLE_VULKAN
tensorcx::vulkan::VulkanTensor vulkan_primitive(
    const tensorcx::vulkan::VulkanTensor& input, OpDesc op,
    const tensorcx::vulkan::VulkanTensor* right = nullptr) {
  tensorcx::vulkan::VulkanBackend backend;
  std::array<tensorcx::Tensor, 2> inputs{tensorcx::vulkan::to_core_tensor(input), {}};
  if (right) inputs[1] = tensorcx::vulkan::to_core_tensor(*right);
  std::array<tensorcx::Tensor, 1> outputs;
  tensorcx::BackendExecution execution;
  execution.op = std::move(op);
  execution.inputs = std::span<const tensorcx::Tensor>(inputs.data(), right ? 2 : 1);
  execution.outputs = outputs;
  const auto status = without_gil([&] { return backend.execute(execution); });
  if (!status.ok()) throw_status(status);
  return unwrap(tensorcx::vulkan::from_core_tensor(outputs[0]));
}

tensorcx::vulkan::VulkanTensor binary_op(const tensorcx::vulkan::VulkanTensor& lhs,
                                       const tensorcx::vulkan::VulkanTensor& rhs, OpKind kind) {
  return vulkan_primitive(lhs, OpDesc{kind}, &rhs);
}

nb::object vulkan_fill(Shape shape, DType dtype, double value, int index) {
  tensorcx::vulkan::VulkanBackend backend;
  const auto strides = tensorcx::contiguous_strides(shape);
  std::array<tensorcx::Tensor, 1> outputs{tensorcx::Tensor{
      dtype, std::move(shape), strides, {"vulkan", index}, nullptr, 0}};
  tensorcx::BackendExecution execution;
  execution.op = OpDesc{OpKind::kFill}; execution.op.scalar_value = value;
  execution.outputs = outputs;
  const auto status = without_gil([&] { return backend.execute(execution); });
  if (!status.ok()) throw_status(status);
  return nb::cast(unwrap(tensorcx::vulkan::from_core_tensor(outputs[0])));
}
#endif

// Metadata-only view: the typed constructor validates the shared buffer and
// computes contiguous strides without allocating or moving tensor data.
template <typename NativeTensor>
NativeTensor reshape_tensor(const NativeTensor& input, nb::handle requested_shape) {
  Shape shape = parse_shape(requested_shape);
  std::optional<std::size_t> inferred;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == -1) {
      if (inferred) {
        throw std::invalid_argument("reshape permits only one inferred dimension");
      }
      inferred = i;
      shape[i] = 1;
    }
  }
  const auto count = tensorcx::numel(input.shape());
  const auto specified = tensorcx::numel(shape);
  if (inferred) {
    if (specified == 0) {
      throw std::invalid_argument("cannot infer reshape dimension with a zero product");
    }
    if (count % specified != 0) {
      throw std::invalid_argument("reshape must preserve the number of elements");
    }
    shape[*inferred] = count / specified;
  } else if (specified != count) {
    throw std::invalid_argument("reshape must preserve the number of elements");
  }
  return NativeTensor(input.dtype(), std::move(shape), input.buffer());
}

template <typename NativeTensor, typename Backend, typename Convert>
void bind_multi_input_tensor_ops(nb::module_& module, Convert convert) {
  auto inference = [convert](OpDesc op, std::vector<tensorcx::Tensor> inputs) {
    std::array<tensorcx::Tensor, 1> outputs;
    Backend backend; tensorcx::BackendExecution e;
    e.op=std::move(op);e.inputs=inputs;e.outputs=outputs;
    const auto status=without_gil([&] {return backend.execute(e);});
    if(!status.ok())throw_status(status);
    return convert(outputs[0]);
  };
  module.def("linear", [inference](const NativeTensor& x,const NativeTensor& weight,nb::handle bias) {
    std::vector<tensorcx::Tensor> inputs{to_core_tensor(x),to_core_tensor(weight)};
    if(!bias.is_none())inputs.push_back(to_core_tensor(nb::cast<const NativeTensor&>(bias)));
    return inference(OpDesc{OpKind::kLinear},std::move(inputs));
  },nb::arg("input"),nb::arg("weight"),nb::arg("bias")=nb::none());
  module.def("embedding", [inference](const NativeTensor& indices,const NativeTensor& weight) {
    return inference(OpDesc{OpKind::kEmbedding},{to_core_tensor(indices),to_core_tensor(weight)});
  },nb::arg("indices"),nb::arg("weight"));
  module.def("_affine_norm", [inference](const NativeTensor& x,nb::handle weight,nb::handle bias,
                                       nb::handle axis,double eps,bool rms) {
    OpDesc op{rms?OpKind::kAffineRmsNorm:OpKind::kAffineLayerNorm};op.axis=cast_dim_or_throw(axis);op.epsilon=eps;
    op.has_weight=!weight.is_none();op.has_bias=!bias.is_none();
    std::vector<tensorcx::Tensor> inputs{to_core_tensor(x)};
    if(op.has_weight)inputs.push_back(to_core_tensor(nb::cast<const NativeTensor&>(weight)));
    if(op.has_bias)inputs.push_back(to_core_tensor(nb::cast<const NativeTensor&>(bias)));
    return inference(op,std::move(inputs));
  },nb::arg("input"),nb::arg("weight").none(),nb::arg("bias").none(),nb::arg("axis"),nb::arg("eps"),nb::arg("rms"));
  module.def("_attention", [inference](const NativeTensor& query,const NativeTensor& key,const NativeTensor& value,
                                     nb::handle mask,bool causal,nb::handle scale) {
    OpDesc op{OpKind::kAttention};op.causal=causal;
    if(!scale.is_none())op.attention_scale=nb::cast<double>(scale);
    std::vector<tensorcx::Tensor> inputs{to_core_tensor(query),to_core_tensor(key),to_core_tensor(value)};
    if(!mask.is_none())inputs.push_back(to_core_tensor(nb::cast<const NativeTensor&>(mask)));
    return inference(op,std::move(inputs));
  },nb::arg("query"),nb::arg("key"),nb::arg("value"),nb::arg("mask").none(),nb::arg("causal"),nb::arg("scale").none());
  module.def("_clip", [convert](const NativeTensor& x, const NativeTensor& lo, const NativeTensor& hi) {
    std::array<tensorcx::Tensor, 3> inputs{to_core_tensor(x),to_core_tensor(lo),to_core_tensor(hi)};
    std::array<tensorcx::Tensor, 1> outputs;
    Backend backend; tensorcx::BackendExecution e;
    e.op=OpDesc{OpKind::kClip}; e.inputs=inputs; e.outputs=outputs;
    const auto status=without_gil([&] {return backend.execute(e);});
    if(!status.ok()) throw_status(status);
    return convert(outputs[0]);
  },nb::arg("input"),nb::arg("lower"),nb::arg("upper"));
  module.def("_topk", [convert](const NativeTensor& x, nb::handle k, nb::handle axis, bool largest, bool sorted) {
    std::array<tensorcx::Tensor, 1> inputs{to_core_tensor(x)};
    std::array<tensorcx::Tensor, 2> outputs;
    Backend backend; tensorcx::BackendExecution e;
    e.op=OpDesc{OpKind::kTopK}; e.op.k=cast_dim_or_throw(k); e.op.axis=cast_dim_or_throw(axis);
    e.op.largest=largest; e.op.sorted=sorted; e.inputs=inputs; e.outputs=outputs;
    const auto status=without_gil([&] {return backend.execute(e);});
    if(!status.ok()) throw_status(status);
    return nb::make_tuple(convert(outputs[0]),convert(outputs[1]));
  },nb::arg("input"),nb::arg("k"),nb::arg("axis"),nb::arg("largest"),nb::arg("sorted"));
  module.def("where", [convert](const NativeTensor& condition, const NativeTensor& a, const NativeTensor& b) {
    std::array<tensorcx::Tensor, 3> inputs{to_core_tensor(condition), to_core_tensor(a), to_core_tensor(b)};
    std::array<tensorcx::Tensor, 1> outputs;
    Backend backend;
    tensorcx::BackendExecution execution;
    execution.op = OpDesc{OpKind::kWhere}; execution.inputs = inputs; execution.outputs = outputs;
    const auto status = without_gil([&] { return backend.execute(execution); });
    if (!status.ok()) throw_status(status);
    return convert(outputs[0]);
  }, nb::arg("condition"), nb::arg("x"), nb::arg("y"));
  module.def("_concat", [convert](const NativeTensor& first, nb::iterable rest, nb::handle axis) {
    OpDesc op{OpKind::kConcat}; op.axis = cast_dim_or_throw(axis);
    std::vector<tensorcx::Tensor> inputs{to_core_tensor(first)};
    for (nb::handle item : rest) {
      if (!nb::isinstance<NativeTensor>(item))
        throw std::invalid_argument("concat inputs must use the same backend");
      inputs.push_back(to_core_tensor(nb::cast<const NativeTensor&>(item)));
    }
    std::array<tensorcx::Tensor, 1> outputs;
    Backend backend;
    tensorcx::BackendExecution execution;
    execution.op = op; execution.inputs = inputs; execution.outputs = outputs;
    const auto status = without_gil([&] { return backend.execute(execution); });
    if (!status.ok()) throw_status(status);
    return convert(outputs[0]);
  }, nb::arg("first"), nb::arg("rest"), nb::arg("axis"));
}

template <typename NativeTensor, typename ExecuteSingle>
void bind_tensor_extensions(nb::module_& module, ExecuteSingle execute_single) {
  for (auto [name, kind] : {std::pair{"_sum_axes", OpKind::kSum},
                            {"_min_axes", OpKind::kMin}, {"_argmax_axes", OpKind::kArgmax},
                            {"_max_axes", OpKind::kMax}, {"_mean_axes", OpKind::kMean},
                            {"_any_axes", OpKind::kAny}, {"_all_axes", OpKind::kAll}}) {
    module.def(name, [kind, execute_single](const NativeTensor& input, nb::handle axes) {
      OpDesc op{kind}; op.reduction_axes = parse_shape(axes);
      return execute_single(input, op);
    }, nb::arg("input"), nb::arg("axes"));
  }
  for (auto [name,kind] : {std::pair{"log",OpKind::kLog},{"sqrt",OpKind::kSqrt},{"abs",OpKind::kAbs}}) {
    module.def(name,[kind,execute_single](const NativeTensor& x) {return execute_single(x,OpDesc{kind});},nb::arg("input"));
  }
  module.def("_slice", [execute_single](const NativeTensor& input, nb::handle starts,
                                        nb::handle steps, nb::handle lengths) {
    OpDesc op{OpKind::kSlice};
    op.slice_starts = parse_shape(starts); op.slice_steps = parse_shape(steps);
    op.slice_shape = parse_shape(lengths);
    return execute_single(input, op);
  }, nb::arg("input"), nb::arg("starts"), nb::arg("steps"), nb::arg("lengths"));
  module.def("reshape", &reshape_tensor<NativeTensor>, nb::arg("input"), nb::arg("shape"));
  module.def("transpose", [execute_single](const NativeTensor& input, nb::handle axes) {
    OpDesc op{OpKind::kTranspose};
    op.axes = parse_shape(axes);
    return execute_single(input, op);
  }, nb::arg("input"), nb::arg("axes"));
  module.def("astype", [execute_single](const NativeTensor& input, const std::string& dtype) {
    OpDesc op{OpKind::kCast};
    op.target_dtype = parse_dtype(nb::str(dtype.c_str()), DType::kFloat32);
    return execute_single(input, op);
  }, nb::arg("input"), nb::arg("dtype"));
  // Internal interpreter support: compare storage, never expose device pointers.
  module.def("_shares_storage", [](const NativeTensor& lhs, const NativeTensor& rhs) {
    const auto relation = to_core_tensor(lhs).buffer->storage_relation(*to_core_tensor(rhs).buffer);
    if (relation == tensorcx::StorageRelation::kPartialOverlap)
      throw std::invalid_argument("kernel arguments have unsupported partial storage overlap");
    return relation == tensorcx::StorageRelation::kSameRange;
  }, nb::arg("lhs"), nb::arg("rhs"));
  for (auto [name, kind] : {std::pair{"subtract", OpKind::kSubtract},
                            {"divide", OpKind::kDivide}}) {
    module.def(name, [kind](const NativeTensor& lhs, const NativeTensor& rhs) {
      return binary_op(lhs, rhs, kind);
    }, nb::arg("lhs"), nb::arg("rhs"));
  }
  for (auto [name, kind] : {std::pair{"equal", OpKind::kEqual}, {"not_equal", OpKind::kNotEqual},
       {"less", OpKind::kLess}, {"less_equal", OpKind::kLessEqual}, {"greater", OpKind::kGreater},
       {"greater_equal", OpKind::kGreaterEqual}, {"logical_and", OpKind::kLogicalAnd},
       {"logical_or", OpKind::kLogicalOr}, {"logical_xor", OpKind::kLogicalXor}, {"masked_select", OpKind::kMaskedSelect}}) {
    module.def(name, [kind](const NativeTensor& lhs, const NativeTensor& rhs) {
      return binary_op(lhs, rhs, kind);
    }, nb::arg("lhs"), nb::arg("rhs"));
  }
  module.def("logical_not", [execute_single](const NativeTensor& input) {
    return execute_single(input, OpDesc{OpKind::kLogicalNot});
  }, nb::arg("input"));
  module.def("negative", [execute_single](const NativeTensor& input) {
    return execute_single(input, OpDesc{OpKind::kNegate});
  }, nb::arg("input"));
  for (auto [name, kind] : {std::pair{"add_scalar", OpKind::kAddScalar},
                            {"subtract_scalar", OpKind::kSubtractScalar},
                            {"multiply_scalar", OpKind::kMultiplyScalar},
                            {"divide_scalar", OpKind::kDivideScalar}}) {
    module.def(name, [kind, execute_single](const NativeTensor& input,
                                           nb::handle scalar, bool scalar_left) {
      if (is_bool_like(scalar)) {
        throw std::invalid_argument("bool arithmetic scalars are not supported");
      }
      OpDesc op{kind};
      if (input.dtype() == DType::kInt32) {
        op.scalar_value = cast_int32_or_throw(scalar);
      } else {
        try {
          op.scalar_value = nb::cast<double>(scalar);
        } catch (const std::exception&) {
          throw std::invalid_argument("scalar must be a real number convertible to float32");
        }
      }
      op.scalar_left = scalar_left;
      return execute_single(input, op);
    }, nb::arg("input"), nb::arg("scalar"), nb::arg("scalar_left") = false);
  }
}

std::span<const BackendRoute> backend_routes() {
  static const std::vector<BackendRoute> routes = [] {
    std::vector<BackendRoute> result{
      {"cpu", &cpu_backend_available, &cpu_backend_device_name,
       &cpu_backend_fill, &cpu_backend_matmul_backends},
#if TENSORCX_ENABLE_METAL
      {"metal", &metal_backend_available, &metal_backend_device_name,
       &metal_backend_fill, &metal_backend_matmul_backends},
#endif
#if TENSORCX_ENABLE_CUDA
      {"cuda", &cuda_backend_available, &cuda_backend_device_name,
       &cuda_backend_fill, &cuda_backend_matmul_backends},
#endif
    };
#if TENSORCX_ENABLE_VULKAN
    for (std::size_t i = 0; i < tensorcx::vulkan::device_count(); ++i) {
      const int index = static_cast<int>(i);
      result.push_back({tensorcx::vulkan::device_string(index),
        [index] { return tensorcx::vulkan::available(index); },
        [index] { return unwrap(tensorcx::vulkan::device_info(index)).name; },
        [index](Shape shape, DType dtype, double value) {
          return vulkan_fill(std::move(shape), dtype, value, index);
        },
        [] { nb::list names; names.append("auto"); names.append("custom"); return names; }});
    }
#endif
    return result;
  }();
  return routes;
}

const BackendRoute* find_backend_route(std::string_view name) {
  for (const BackendRoute& route : backend_routes()) {
    if (route.name == name) {
      return &route;
    }
  }
  return nullptr;
}

const BackendRoute& require_known_backend_route(const std::string& name) {
  const BackendRoute* route = find_backend_route(name);
  if (route == nullptr) {
    throw std::invalid_argument("device is not available: " + name);
  }
  return *route;
}

const BackendRoute& require_available_backend_route(const std::string& name) {
  const BackendRoute& route = require_known_backend_route(name);
  if (!route.is_available()) {
    throw std::invalid_argument("device is not available: " + name);
  }
  return route;
}

#if TENSORCX_ENABLE_METAL
struct ParsedKernelArguments {
  std::deque<tensorcx::Tensor> tensor_storage;
  std::vector<tensorcx::KernelArgument> arguments;
};

ParsedKernelArguments parse_backend_kernel_arguments(nb::sequence arguments) {
  ParsedKernelArguments parsed;
  // Python sequences may iterate more items than __len__ reports. Tensor
  // addresses must remain stable as arguments are appended.
  for (nb::handle item : arguments) {
    if (nb::isinstance<tensorcx::metal::MetalTensor>(item)) {
      const auto& tensor = nb::cast<const tensorcx::metal::MetalTensor&>(item);
      parsed.tensor_storage.push_back(tensorcx::metal::to_core_tensor(tensor));
      parsed.arguments.push_back(tensorcx::KernelArgument{
          tensorcx::KernelArgumentKind::kTensor,
          &parsed.tensor_storage.back(),
          0,
      });
      continue;
    }
    parsed.arguments.push_back(tensorcx::KernelArgument{
        tensorcx::KernelArgumentKind::kUInt32,
        nullptr,
        cast_uint32_or_throw(item, "kernel scalar arguments must be uint32"),
    });
  }
  return parsed;
}

std::string launch_metal_library_function_via_backend(
    nb::bytes metallib,
    const std::string& function_name,
    nb::sequence arguments,
    nb::handle output,
    nb::handle thread_count,
    nb::handle threads_per_threadgroup) {
  const auto& output_tensor = nb::cast<const tensorcx::metal::MetalTensor&>(output);
  ParsedKernelArguments parsed_arguments = parse_backend_kernel_arguments(arguments);
  std::array<tensorcx::Tensor, 1> outputs{
      tensorcx::metal::to_core_tensor(output_tensor),
  };

  const tensorcx::LaunchConfig launch{
      cast_uint32_or_throw(thread_count, "thread count must be uint32"),
      1,
      1,
      cast_uint32_or_throw(threads_per_threadgroup, "threads per threadgroup must be uint32"),
      1,
      1,
  };
  const tensorcx::CompilationTarget target{
      tensorcx::KernelArtifactKind::kBinary,
      bytes_to_string(metallib),
      function_name,
  };
  const tensorcx::BackendExecution execution{
      tensorcx::BackendOpClass::kKernel,
      OpDesc{},
      std::span<const tensorcx::Tensor>(),
      std::span<tensorcx::Tensor>(outputs.data(), outputs.size()),
      launch,
      target,
      std::span<const tensorcx::KernelArgument>(
          parsed_arguments.arguments.data(),
          parsed_arguments.arguments.size()),
  };

  tensorcx::metal::MetalBackend backend;
  const tensorcx::Status status = without_gil([&] { return backend.execute(execution); });
  if (!status.ok()) {
    throw_status(status);
  }
  return function_name;
}
#endif

}  // namespace

void bind_dlpack(nb::module_& module);

NB_MODULE(_core, module) {
  module.def("_profile_begin", [](nb::handle limit, nb::handle device_timing) {
    if (!PyBool_Check(device_timing.ptr())) throw std::invalid_argument("device_timing must be a bool");
    return tensorcx::begin_profile(cast_uint32_or_throw(limit, "max_events must be an integer in [1, 100000]"),
                                   nb::cast<bool>(device_timing));
  }, nb::arg("max_events"), nb::arg("device_timing") = false);
  module.def("_profile_end", [](nb::handle token) {
    if (is_bool_like(token)) throw std::invalid_argument("profile token must be an integer");
    const auto result = tensorcx::end_profile(nb::cast<std::uint64_t>(token));
    nb::dict report;
    report["elapsed_ns"] = result.elapsed_ns;
    report["dropped_events"] = result.dropped_events;
    nb::list events;
    for (const auto& event : result.events) {
      nb::dict row;
      row["id"] = event.id; row["parent_id"] = event.parent_id;
      row["start_ns"] = event.start_ns; row["duration_ns"] = event.duration_ns;
      row["self_ns"] = event.self_ns;
      row["category"] = tensorcx::profile_category_name(event.category);
      row["name"] = event.name; row["backend"] = event.backend;
      row["bytes"] = event.bytes ? nb::cast(*event.bytes) : nb::none();
      row["operation"] = event.operation ? nb::cast(tensorcx::profile_operation_name(*event.operation)) : nb::none();
      events.append(row);
    }
    report["events"] = events;
    report["device_timing"] = result.device_timing;
    report["dropped_device_events"] = result.dropped_device_events;
    nb::list device_events;
    for (const auto& event : result.device_events) {
      nb::dict row;
      row["id"] = event.id; row["host_event_id"] = event.host_event_id;
      row["name"] = event.name; row["backend"] = event.backend;
      row["operation"] = event.operation ? nb::cast(tensorcx::profile_operation_name(*event.operation)) : nb::none();
      row["elapsed_ms"] = event.elapsed_ms ? nb::cast(*event.elapsed_ms) : nb::none();
      row["error"] = event.error ? nb::cast(event.error) : nb::none();
      device_events.append(row);
    }
    report["device_events"] = device_events;
    return report;
  });
  module.doc() = "Native extension module for tensor.cx.";
  module.def("_build_info", [] {
    nb::dict result;
    result["cpu"] = true;
    result["metal"] = static_cast<bool>(TENSORCX_ENABLE_METAL);
    result["cuda"] = static_cast<bool>(TENSORCX_ENABLE_CUDA);
    result["vulkan"] = static_cast<bool>(TENSORCX_ENABLE_VULKAN);
    return result;
  });
#if TENSORCX_ENABLE_CUDA
  nb::class_<tensorcx::cuda::MemoryPool>(module, "_CudaMemoryPool")
      .def_static("create", [](nb::handle bytes, nb::handle blocks) {
        if (is_bool_like(bytes) || is_bool_like(blocks)) throw std::invalid_argument("memory pool limits must be integers");
        return std::make_shared<tensorcx::cuda::MemoryPool>(nb::cast<std::size_t>(bytes), nb::cast<std::size_t>(blocks));
      })
      .def("enter", [](tensorcx::cuda::MemoryPool& pool) {
        const auto status = without_gil([&] { return pool.enter(); });
        if (!status.ok()) throw_status(status);
      })
      .def("exit", [](tensorcx::cuda::MemoryPool& pool) {
        const auto status = without_gil([&] { return pool.exit(); });
        if (!status.ok()) throw_status(status);
      })
      .def("trim", [](tensorcx::cuda::MemoryPool& pool) {
        const auto status = without_gil([&] { return pool.trim(); });
        if (!status.ok()) throw_status(status);
      })
      .def("stats", [](const tensorcx::cuda::MemoryPool& pool) {
        const auto stats = without_gil([&] { return pool.stats(); });
        nb::dict result;
#define TENSORCX_POOL_FIELD(field) result[#field] = stats.field
        TENSORCX_POOL_FIELD(max_cached_bytes); TENSORCX_POOL_FIELD(max_cached_blocks);
        TENSORCX_POOL_FIELD(cached_bytes); TENSORCX_POOL_FIELD(cached_blocks);
        TENSORCX_POOL_FIELD(in_use_bytes); TENSORCX_POOL_FIELD(peak_in_use_bytes);
        TENSORCX_POOL_FIELD(peak_cached_bytes); TENSORCX_POOL_FIELD(hits); TENSORCX_POOL_FIELD(misses);
        TENSORCX_POOL_FIELD(allocation_calls); TENSORCX_POOL_FIELD(free_calls);
        TENSORCX_POOL_FIELD(allocation_failures); TENSORCX_POOL_FIELD(oom_retries);
        TENSORCX_POOL_FIELD(uncached_returns); TENSORCX_POOL_FIELD(readiness_failures);
        TENSORCX_POOL_FIELD(free_failures); TENSORCX_POOL_FIELD(failed_free_bytes); TENSORCX_POOL_FIELD(active);
#undef TENSORCX_POOL_FIELD
        return result;
      });
  module.def("_cuda_device_info", [] {
    auto info = unwrap(without_gil([] { return tensorcx::cuda::device_info(); }));
    nb::dict memory;
    memory["scope"] = "cuda_device";
    memory["total_bytes"] = info.total_bytes;
    memory["available_bytes"] = info.free_bytes;
    nb::dict result;
    result["name"] = info.name;
    result["architecture"] = "sm_" + std::to_string(info.major) + std::to_string(info.minor);
    result["runtime_version"] = info.runtime_version;
    result["driver_api_version"] = info.driver_version;
    result["memory"] = memory;
    return result;
  });
  nb::class_<tensorcx::cuda::CudaKernelModule>(module, "_CudaKernelModule");
  module.def("_cuda_kernel_support", [] {
    auto status = without_gil([] { return tensorcx::cuda::compiled_kernel_support(); });
    if (!status.ok()) throw_status(status);
  });
  module.def("_load_cuda_kernel", [](const std::string& ptx, const std::string& entry,
      const std::string& kinds, std::uint32_t output, std::uint32_t guard) {
    return unwrap(without_gil([&] {
      return tensorcx::cuda::CudaKernelModule::load(ptx, entry, {kinds, output, guard});
    }));
  });
  module.def("_launch_cuda_kernel",
      [](std::shared_ptr<tensorcx::cuda::CudaKernelModule> compiled, nb::sequence values,
         nb::handle threads, nb::handle block) {
    std::deque<tensorcx::Tensor> tensors;
    std::vector<tensorcx::KernelArgument> arguments;
    for (nb::handle value : values) {
      if (nb::isinstance<tensorcx::cuda::CudaTensor>(value)) {
        tensors.push_back(tensorcx::cuda::to_core_tensor(nb::cast<const tensorcx::cuda::CudaTensor&>(value)));
        arguments.push_back({tensorcx::KernelArgumentKind::kTensor, &tensors.back(), 0});
      } else {
        arguments.push_back({tensorcx::KernelArgumentKind::kUInt32, nullptr,
                             cast_uint32_or_throw(value, "kernel scalar arguments must be uint32")});
      }
    }
    const auto count = cast_uint32_or_throw(threads, "thread_count must be uint32");
    const auto group = cast_uint32_or_throw(block, "block_size must be uint32");
    return unwrap(without_gil([&] {
      return tensorcx::cuda::launch_compiled_kernel(compiled, arguments, count, group);
    }));
  });
#endif
  nb::class_<tensorcx::cpu::CpuKernelModule>(module, "_CpuKernelModule");
  module.def("_cpu_kernel_supported", &tensorcx::cpu::compiled_kernel_supported);
  module.def("_load_cpu_kernel", [](const std::string& path, const std::string& kinds,
                                    std::uint32_t output, std::uint32_t guard) {
    return unwrap(without_gil([&] {
      return tensorcx::cpu::CpuKernelModule::load(path, {kinds, output, guard});
    }));
  });
  module.def("_launch_cpu_kernel",
      [](std::shared_ptr<tensorcx::cpu::CpuKernelModule> compiled, nb::sequence values,
         nb::handle threads, nb::handle block) {
    std::deque<tensorcx::Tensor> tensors;
    std::vector<tensorcx::KernelArgument> arguments;
    for (nb::handle value : values) {
      if (nb::isinstance<CpuTensor>(value)) {
        tensors.push_back(tensorcx::cpu::to_core_tensor(nb::cast<const CpuTensor&>(value)));
        arguments.push_back({tensorcx::KernelArgumentKind::kTensor, &tensors.back(), 0});
      } else {
        arguments.push_back({tensorcx::KernelArgumentKind::kUInt32, nullptr,
                             cast_uint32_or_throw(value, "kernel scalar arguments must be uint32")});
      }
    }
    const auto count = cast_uint32_or_throw(threads, "thread_count must be uint32");
    const auto group = cast_uint32_or_throw(block, "block_size must be uint32");
    return unwrap(without_gil([&] {
      return tensorcx::cpu::launch_compiled_kernel(compiled, arguments, count, group);
    }));
  });
  module.def("version", []() { return TENSORCX_RUNTIME_VERSION; });
  module.attr("float32") = "float32";
  module.attr("int32") = "int32";
  module.attr("bool") = "bool";
  module.def("_backend_contract_smoke_test", []() {
    const auto status = tensorcx::null_backend::contract_smoke_test();
    if (!status.ok()) {
      throw std::runtime_error(status.message());
    }
    return true;
  });
  module.def("_cpu_backend_contract_smoke_test", []() {
    const auto status = tensorcx::cpu::contract_smoke_test();
    if (!status.ok()) {
      throw std::runtime_error(status.message());
    }
    return true;
  });
#if TENSORCX_ENABLE_METAL
  module.def("_metal_backend_contract_smoke_test", []() {
    const auto status = tensorcx::metal::contract_smoke_test();
    if (!status.ok()) {
      throw std::runtime_error(status.message());
    }
    return true;
  });
#endif

  nb::class_<CpuTensor>(module, "CpuTensor")
      .def_prop_ro("shape", [](const CpuTensor& tensor) { return shape_tuple(tensor.shape()); })
      .def_prop_ro("strides", [](const CpuTensor& tensor) { return shape_tuple(tensor.strides()); })
      .def_prop_ro("dtype",
                   [](const CpuTensor& tensor) {
                     return std::string(tensorcx::dtype_name(tensor.dtype()));
                   })
      .def_prop_ro("device", [](const CpuTensor&) { return "cpu"; })
      .def_prop_ro("nbytes", [](const CpuTensor& tensor) { return tensor.buffer()->nbytes(); })
      .def("numpy", &tensor_to_numpy)
      .def("cpu", [](const CpuTensor& tensor) { return tensor; })
      .def("__add__", [](const CpuTensor& lhs, const CpuTensor& rhs) {
        return binary_op(lhs, rhs, OpKind::kAdd);
      })
      .def("__mul__", [](const CpuTensor& lhs, const CpuTensor& rhs) {
        return binary_op(lhs, rhs, OpKind::kMultiply);
      });

  module.def("tensor",
             &tensor_from_sequence,
             nb::arg("data"),
             nb::arg("dtype").none() = nb::none(),
             nb::arg("device") = "cpu");
  module.def("tensor_from_flat",
             &tensor_from_flat_sequence,
             nb::arg("data"),
             nb::arg("shape"),
             nb::arg("dtype") = "float32",
             nb::arg("device") = "cpu");
  module.def("empty",
             [](nb::handle shape, nb::handle dtype, const std::string& device) {
               validate_cpu_device(device);
               return tensorcx::cpu::empty(parse_shape(shape), parse_dtype(dtype, DType::kFloat32));
             },
             nb::arg("shape"),
             nb::arg("dtype") = "float32",
             nb::arg("device") = "cpu");
  module.def("zeros",
             [](nb::handle shape, nb::handle dtype, const std::string& device) {
               validate_cpu_device(device);
               return fill_op(parse_shape(shape), parse_dtype(dtype, DType::kFloat32), 0.0);
             },
             nb::arg("shape"),
             nb::arg("dtype") = "float32",
             nb::arg("device") = "cpu");
  module.def("ones",
             [](nb::handle shape, nb::handle dtype, const std::string& device) {
               validate_cpu_device(device);
               return fill_op(parse_shape(shape), parse_dtype(dtype, DType::kFloat32), 1.0);
             },
             nb::arg("shape"),
             nb::arg("dtype") = "float32",
             nb::arg("device") = "cpu");
  module.def("fill",
             [](nb::handle shape, nb::handle dtype, double value, const std::string& device) -> nb::object {
               const Shape parsed_shape = parse_shape(shape);
               const DType parsed_dtype = parse_dtype(dtype, DType::kFloat32);
               return require_available_backend_route(device).fill(
                   parsed_shape, parsed_dtype, value);
             },
             nb::arg("shape"),
             nb::arg("dtype") = "float32",
             nb::arg("value") = 0.0,
             nb::arg("device") = "cpu");

  module.def("add",
             [](const CpuTensor& lhs, const CpuTensor& rhs) {
               return binary_op(lhs, rhs, OpKind::kAdd);
             },
             nb::arg("lhs"),
             nb::arg("rhs"));
  bind_tensor_extensions<CpuTensor>(module, cpu_single_input_backend_op);
  bind_multi_input_tensor_ops<CpuTensor, tensorcx::cpu::CpuBackend>(module, tensorcx::cpu::from_core_tensor);
  module.def("multiply",
             [](const CpuTensor& lhs, const CpuTensor& rhs) {
               return binary_op(lhs, rhs, OpKind::kMultiply);
             },
             nb::arg("lhs"),
             nb::arg("rhs"));
  module.def("exp",
             [](const CpuTensor& input) {
               return unary_op(input, OpKind::kExp);
             },
             nb::arg("input"));
  module.def("gelu",
             [](const CpuTensor& input) {
               return unary_op(input, OpKind::kGelu);
             },
             nb::arg("input"));
  module.def("silu",
             [](const CpuTensor& input) {
               return unary_op(input, OpKind::kSilu);
             },
             nb::arg("input"));
  module.def("softmax",
             [](const CpuTensor& input, std::int64_t axis) {
               return axis_unary_op(input, OpKind::kSoftmax, axis);
             },
             nb::arg("input"),
             nb::arg("axis"));
  module.def("rmsnorm",
             [](const CpuTensor& input, std::int64_t axis, double eps) {
               return norm_op(input, OpKind::kRmsNorm, axis, eps);
             },
             nb::arg("input"),
             nb::arg("axis"),
             nb::arg("eps") = 1.0e-5);
  module.def("layernorm",
             [](const CpuTensor& input, std::int64_t axis, double eps) {
               return norm_op(input, OpKind::kLayerNorm, axis, eps);
             },
             nb::arg("input"),
             nb::arg("axis"),
             nb::arg("eps") = 1.0e-5);
  module.def("sum",
             [](const CpuTensor& input, std::int64_t axis) {
               return reduction_op(input, OpKind::kSum, axis);
             },
             nb::arg("input"),
             nb::arg("axis"));
  module.def("max",
             [](const CpuTensor& input, std::int64_t axis) {
               return reduction_op(input, OpKind::kMax, axis);
             },
             nb::arg("input"),
             nb::arg("axis"));
  module.def("mean",
             [](const CpuTensor& input, std::int64_t axis) {
               return reduction_op(input, OpKind::kMean, axis);
             },
             nb::arg("input"),
             nb::arg("axis"));
  module.def("matmul_backends",
             [](const std::string& device) {
               return require_known_backend_route(device).matmul_backends();
             },
             nb::arg("device"));
  module.def("matmul",
             &matmul_cpu,
             nb::arg("lhs"),
             nb::arg("rhs"),
             nb::arg("backend") = "auto");
  module.def("is_available",
             [](const std::string& device) {
               const BackendRoute* route = find_backend_route(device);
               return route != nullptr && route->is_available();
             },
             nb::arg("device"));
  module.def("devices", []() {
    nb::list result;
    for (const BackendRoute& route : backend_routes()) {
      if (route.is_available()) {
        result.append(route.name);
      }
    }
    return result;
  });
  module.def("device_name",
             [](const std::string& device) {
               return require_available_backend_route(device).device_name();
             },
             nb::arg("device"));
  module.def("validate_metal_library_function",
             [](nb::bytes metallib, const std::string& function_name) {
#if TENSORCX_ENABLE_METAL
               const std::vector<std::uint8_t> library_bytes = bytes_to_vector(metallib);
               return unwrap(without_gil([&] {
                 return tensorcx::metal::validate_library_function(library_bytes, function_name);
               }));
#else
               (void)metallib;
               (void)function_name;
               throw std::runtime_error("Metal is not available on this system");
#endif
             },
             nb::arg("metallib"),
             nb::arg("function_name"));
  module.def("launch_metal_library_function",
             [](nb::bytes metallib,
                const std::string& function_name,
                nb::sequence arguments,
                nb::handle output,
                nb::handle thread_count,
                nb::handle threads_per_threadgroup) {
#if TENSORCX_ENABLE_METAL
               return launch_metal_library_function_via_backend(
                   metallib,
                   function_name,
                   arguments,
                   output,
                   thread_count,
                   threads_per_threadgroup);
#else
               (void)metallib;
               (void)function_name;
               (void)arguments;
               (void)output;
               (void)thread_count;
               (void)threads_per_threadgroup;
               throw std::runtime_error("Metal is not available on this system");
#endif
             },
             nb::arg("metallib"),
             nb::arg("function_name"),
             nb::arg("arguments"),
             nb::arg("output"),
             nb::arg("thread_count"),
             nb::arg("threads_per_threadgroup"));

#if TENSORCX_ENABLE_VULKAN
  using tensorcx::vulkan::VulkanTensor;
  nb::class_<tensorcx::vulkan::VulkanKernelModule>(module, "_VulkanKernelModule");
  module.def("_vulkan_kernel_support", [](int index) {
    const auto status = without_gil([&] { return tensorcx::vulkan::compiled_kernel_support(index); });
    if (!status.ok()) throw_status(status);
  });
  module.def("_load_vulkan_kernel", [](nb::bytes data, const std::string& kinds,
      nb::handle output, nb::handle guard, int index) {
    const std::string artifact(data.c_str(), data.size());
    const auto out = cast_uint32_or_throw(output, "kernel output index must be uint32");
    const auto bound = cast_uint32_or_throw(guard, "kernel guard index must be uint32");
    return unwrap(without_gil([&] {
      return tensorcx::vulkan::VulkanKernelModule::load(artifact, {kinds, out, bound}, index);
    }));
  });
  module.def("_launch_vulkan_kernel", [](std::shared_ptr<tensorcx::vulkan::VulkanKernelModule> compiled,
      nb::sequence values, nb::handle threads, nb::handle block) {
    std::deque<tensorcx::Tensor> tensors;
    std::vector<tensorcx::KernelArgument> arguments;
    for (nb::handle value : values) {
      if (nb::isinstance<VulkanTensor>(value)) {
        tensors.push_back(tensorcx::vulkan::to_core_tensor(nb::cast<const VulkanTensor&>(value)));
        arguments.push_back({tensorcx::KernelArgumentKind::kTensor, &tensors.back(), 0});
      } else {
        arguments.push_back({tensorcx::KernelArgumentKind::kUInt32, nullptr,
                            cast_uint32_or_throw(value, "kernel scalar arguments must be uint32")});
      }
    }
    const auto count = cast_uint32_or_throw(threads, "thread_count must be uint32");
    const auto group = cast_uint32_or_throw(block, "block_size must be uint32");
    return unwrap(without_gil([&] {
      return tensorcx::vulkan::launch_compiled_kernel(compiled, arguments, count, group);
    }));
  });
  nb::class_<tensorcx::vulkan::MemoryPool>(module, "_VulkanMemoryPool")
      .def_static("create", [](int index, nb::handle bytes, nb::handle blocks) {
        if (is_bool_like(bytes) || is_bool_like(blocks)) throw std::invalid_argument("memory pool limits must be integers");
        return std::make_shared<tensorcx::vulkan::MemoryPool>(index, nb::cast<std::size_t>(bytes), nb::cast<std::size_t>(blocks));
      })
      .def("enter", [](tensorcx::vulkan::MemoryPool& pool) {
        const auto status = without_gil([&] { return pool.enter(); }); if (!status.ok()) throw_status(status);
      })
      .def("exit", [](tensorcx::vulkan::MemoryPool& pool) {
        const auto status = without_gil([&] { return pool.exit(); }); if (!status.ok()) throw_status(status);
      })
      .def("trim", [](tensorcx::vulkan::MemoryPool& pool) {
        const auto status = without_gil([&] { return pool.trim(); }); if (!status.ok()) throw_status(status);
      })
      .def("stats", [](const tensorcx::vulkan::MemoryPool& pool) {
        const auto stats = without_gil([&] { return pool.stats(); }); nb::dict result;
#define TENSORCX_VK_POOL_FIELD(field) result[#field] = stats.field
        TENSORCX_VK_POOL_FIELD(max_cached_bytes); TENSORCX_VK_POOL_FIELD(max_cached_blocks);
        TENSORCX_VK_POOL_FIELD(cached_bytes); TENSORCX_VK_POOL_FIELD(cached_blocks);
        TENSORCX_VK_POOL_FIELD(in_use_bytes); TENSORCX_VK_POOL_FIELD(peak_in_use_bytes);
        TENSORCX_VK_POOL_FIELD(peak_cached_bytes); TENSORCX_VK_POOL_FIELD(hits); TENSORCX_VK_POOL_FIELD(misses);
        TENSORCX_VK_POOL_FIELD(allocation_calls); TENSORCX_VK_POOL_FIELD(free_calls);
        TENSORCX_VK_POOL_FIELD(allocation_failures); TENSORCX_VK_POOL_FIELD(oom_retries);
        TENSORCX_VK_POOL_FIELD(uncached_returns); TENSORCX_VK_POOL_FIELD(frame_hits);
        TENSORCX_VK_POOL_FIELD(frame_misses); TENSORCX_VK_POOL_FIELD(frame_returns);
        TENSORCX_VK_POOL_FIELD(frame_reset_failures); TENSORCX_VK_POOL_FIELD(cached_frames);
        TENSORCX_VK_POOL_FIELD(active);
#undef TENSORCX_VK_POOL_FIELD
        return result;
      });
  nb::class_<VulkanTensor>(module, "VulkanTensor")
      .def_prop_ro("shape", [](const VulkanTensor& x) { return shape_tuple(x.shape()); })
      .def_prop_ro("strides", [](const VulkanTensor& x) { return shape_tuple(x.strides()); })
      .def_prop_ro("dtype", [](const VulkanTensor& x) { return std::string(tensorcx::dtype_name(x.dtype())); })
      .def_prop_ro("device", [](const VulkanTensor& x) { return tensorcx::vulkan::device_string(x.device().index); })
      .def_prop_ro("nbytes", &VulkanTensor::nbytes);
  module.def("_vulkan_device_count", &tensorcx::vulkan::device_count);
  module.def("_vulkan_validation_errors", &tensorcx::vulkan::validation_errors);
  module.def("_vulkan_discovery_error", &tensorcx::vulkan::discovery_error);
  module.def("_vulkan_device_info", [](int index) {
    const auto context = unwrap(without_gil([&] { return tensorcx::vulkan::get_context(index); }));
    const auto info = context->info;
    const auto stats = context->statistics();
    nb::dict result, memory;
    result["name"] = info.name;
    result["api_version"] = std::to_string(VK_API_VERSION_MAJOR(info.api_version)) + "." +
        std::to_string(VK_API_VERSION_MINOR(info.api_version)) + "." + std::to_string(VK_API_VERSION_PATCH(info.api_version));
    result["driver_version_raw"] = info.driver_version;
    result["vendor_id"] = info.vendor_id;
    result["max_storage_buffer_range"] = info.max_storage_buffer_range;
    memory["total_bytes"] = info.device_local_bytes;
    memory["available_bytes"] = nb::none(); memory["scope"] = "device_local_heaps";
    result["memory"] = memory;
    nb::dict allocations, execution, roles, timestamps;
#define TENSORCX_VK_ALLOCATION(field) allocations[#field] = stats.field
    TENSORCX_VK_ALLOCATION(physical_blocks); TENSORCX_VK_ALLOCATION(physical_bytes);
    TENSORCX_VK_ALLOCATION(peak_physical_bytes); TENSORCX_VK_ALLOCATION(cached_blocks);
    TENSORCX_VK_ALLOCATION(cached_bytes); TENSORCX_VK_ALLOCATION(allocation_calls);
    TENSORCX_VK_ALLOCATION(free_calls); TENSORCX_VK_ALLOCATION(allocation_failures);
#undef TENSORCX_VK_ALLOCATION
    const std::array<const char*, 3> names{"tensor", "internal", "staging"};
    for (std::size_t i = 0; i < names.size(); ++i) {
      nb::dict role; role["live_blocks"] = stats.live_blocks[i]; role["requested_bytes"] = stats.live_bytes[i];
      roles[names[i]] = role;
    }
    allocations["live"] = roles; allocations["scope"] = "tensorcx_owned_vulkan_allocations";
    allocations["physical_bytes_unit"] = "VkMemoryRequirements.size";
    result["allocations"] = allocations;
#define TENSORCX_VK_EXECUTION(field) execution[#field] = stats.field
    TENSORCX_VK_EXECUTION(uploads); TENSORCX_VK_EXECUTION(downloads); TENSORCX_VK_EXECUTION(dispatches);
    TENSORCX_VK_EXECUTION(submissions); TENSORCX_VK_EXECUTION(command_buffer_creations);
    TENSORCX_VK_EXECUTION(fence_creations); TENSORCX_VK_EXECUTION(descriptor_pool_creations);
    TENSORCX_VK_EXECUTION(descriptor_set_allocations); TENSORCX_VK_EXECUTION(query_pool_creations);
    TENSORCX_VK_EXECUTION(frame_reuses);
#undef TENSORCX_VK_EXECUTION
    result["execution"] = execution;
    timestamps["supported"] = info.timestamp_bits > 0 && info.timestamp_bits <= 64 &&
        context->CreateQueryPool && context->DestroyQueryPool && context->CmdResetQueryPool &&
        context->CmdWriteTimestamp && context->GetQueryPoolResults;
    timestamps["valid_bits"] = info.timestamp_bits;
    timestamps["period_ns"] = info.timestamp_period;
    result["timestamps"] = timestamps;
    return result;
  }, nb::arg("index") = 0);
  module.def("_vulkan_empty", [](nb::handle shape, nb::handle dtype, int index) {
    Shape parsed = parse_shape(shape);
    const auto parsed_dtype = parse_dtype(dtype, DType::kFloat32);
    return unwrap(without_gil([&] { return tensorcx::vulkan::empty(std::move(parsed), parsed_dtype, index); }));
  }, nb::arg("shape"), nb::arg("dtype") = "float32", nb::arg("index") = 0);
  module.def("cpu_to_vulkan", [](const CpuTensor& tensor, int index) {
    return unwrap(without_gil([&] { return tensorcx::vulkan::from_cpu(tensor, index); }));
  }, nb::arg("tensor"), nb::arg("index") = 0);
  module.def("vulkan_to_cpu", [](const VulkanTensor& tensor) {
    return unwrap(without_gil([&] { return tensorcx::vulkan::to_cpu(tensor); }));
  }, nb::arg("tensor"));
  bind_tensor_extensions<VulkanTensor>(module, [](const VulkanTensor& x, const OpDesc& op) {
    return vulkan_primitive(x, op);
  });
  bind_multi_input_tensor_ops<VulkanTensor, tensorcx::vulkan::VulkanBackend>(module,
      [](const tensorcx::Tensor& x) { return unwrap(tensorcx::vulkan::from_core_tensor(x)); });
  for (auto [name, kind] : {std::pair{"add", OpKind::kAdd}, {"multiply", OpKind::kMultiply}})
    module.def(name, [kind](const VulkanTensor& a, const VulkanTensor& b) { return binary_op(a, b, kind); },
               nb::arg("lhs"), nb::arg("rhs"));
  for (auto [name, kind] : {std::pair{"sum", OpKind::kSum}, {"max", OpKind::kMax}, {"mean", OpKind::kMean},
                          {"min", OpKind::kMin}, {"argmax", OpKind::kArgmax}})
    module.def(name, [kind](const VulkanTensor& x, std::int64_t axis) {
      OpDesc op{kind}; op.axis = axis; return vulkan_primitive(x, op);
    }, nb::arg("input"), nb::arg("axis"));
  // Explicit errors for unsupported public primitive overloads; never a CPU fallback.
  for (auto [name, kind] : {std::pair{"exp", OpKind::kExp}, {"gelu", OpKind::kGelu}, {"silu", OpKind::kSilu}})
    module.def(name, [kind](const VulkanTensor& x) { return vulkan_primitive(x, OpDesc{kind}); }, nb::arg("input"));
  module.def("softmax", [](const VulkanTensor& x, std::int64_t axis) {
    OpDesc op{OpKind::kSoftmax}; op.axis = axis; return vulkan_primitive(x, op);
  }, nb::arg("input"), nb::arg("axis"));
  for (auto [name, kind] : {std::pair{"rmsnorm", OpKind::kRmsNorm}, {"layernorm", OpKind::kLayerNorm}})
    module.def(name, [kind](const VulkanTensor& x, std::int64_t axis, double epsilon) {
      OpDesc op{kind}; op.axis = axis; op.epsilon = epsilon; return vulkan_primitive(x, op);
    }, nb::arg("input"), nb::arg("axis"), nb::arg("eps") = 1.0e-5);
  module.def("matmul", [](const VulkanTensor& a, const VulkanTensor& b, const std::string& backend) {
    OpDesc op{OpKind::kMatmul};
    if (backend == "auto") op.matmul_preference = MatmulPreference::kAuto;
    else if (backend == "custom") op.matmul_preference = MatmulPreference::kCustom;
    else throw std::invalid_argument("Vulkan matmul supports auto or custom");
    return vulkan_primitive(a, op, &b);
  }, nb::arg("lhs"), nb::arg("rhs"), nb::arg("backend") = "auto");
#endif

#if TENSORCX_ENABLE_CUDA
  using tensorcx::cuda::CudaTensor;
  nb::class_<CudaTensor>(module, "CudaTensor")
      .def_prop_ro("shape", [](const CudaTensor& tensor) { return shape_tuple(tensor.shape()); })
      .def_prop_ro("strides", [](const CudaTensor& tensor) { return shape_tuple(tensor.strides()); })
      .def_prop_ro("dtype", [](const CudaTensor& tensor) {
        return std::string(tensorcx::dtype_name(tensor.dtype()));
      })
      .def_prop_ro("device", [](const CudaTensor&) { return "cuda"; })
      .def_prop_ro("nbytes", &CudaTensor::nbytes);
  module.def("_cuda_empty", [](nb::handle shape, nb::handle dtype) {
    Shape parsed_shape = parse_shape(shape);
    const DType parsed_dtype = parse_dtype(dtype, DType::kFloat32);
    return unwrap(without_gil([&] {
      return tensorcx::cuda::empty(std::move(parsed_shape), parsed_dtype);
    }));
  }, nb::arg("shape"), nb::arg("dtype") = "float32");
  bind_multi_input_tensor_ops<CudaTensor, tensorcx::cuda::CudaBackend>(module, [](const tensorcx::Tensor& tensor) {
    return unwrap(tensorcx::cuda::from_core_tensor(tensor));
  });
  bind_tensor_extensions<CudaTensor>(module, [](const CudaTensor& input, const OpDesc& op) {
    return cuda_primitive(input, op);
  });
  module.def("cpu_to_cuda", [](const CpuTensor& tensor) {
    return unwrap(without_gil([&] { return tensorcx::cuda::from_cpu(tensor); }));
  }, nb::arg("tensor"));
  module.def("cuda_to_cpu", [](const CudaTensor& tensor) {
    return unwrap(without_gil([&] { return tensorcx::cuda::to_cpu(tensor); }));
  }, nb::arg("tensor"));
  module.def("add", [](const CudaTensor& lhs, const CudaTensor& rhs) {
    return binary_op(lhs, rhs, OpKind::kAdd);
  }, nb::arg("lhs"), nb::arg("rhs"));
  module.def("multiply", [](const CudaTensor& lhs, const CudaTensor& rhs) {
    return binary_op(lhs, rhs, OpKind::kMultiply);
  }, nb::arg("lhs"), nb::arg("rhs"));
  for (auto [name,kind] : {std::pair{"exp",OpKind::kExp}, {"gelu",OpKind::kGelu}, {"silu",OpKind::kSilu}}) {
    module.def(name,[kind](const CudaTensor& input){return cuda_primitive(input,OpDesc{kind});},nb::arg("input"));
  }
  for (auto [name,kind] : {std::pair{"sum",OpKind::kSum}, {"max",OpKind::kMax},
                          {"mean",OpKind::kMean},{"softmax",OpKind::kSoftmax}}) {
    module.def(name,[kind](const CudaTensor& input,std::int64_t axis){
      OpDesc op{kind};op.axis=axis;return cuda_primitive(input,op);
    },nb::arg("input"),nb::arg("axis"));
  }
  for (auto [name,kind] : {std::pair{"rmsnorm",OpKind::kRmsNorm},{"layernorm",OpKind::kLayerNorm}}) {
    module.def(name,[kind](const CudaTensor& input,std::int64_t axis,double eps){
      OpDesc op{kind};op.axis=axis;op.epsilon=eps;return cuda_primitive(input,op);
    },nb::arg("input"),nb::arg("axis"),nb::arg("eps")=1.0e-5);
  }
  module.def("matmul",[](const CudaTensor& lhs,const CudaTensor& rhs,const std::string& backend){
    OpDesc op{OpKind::kMatmul};
    if(backend=="auto")op.matmul_preference=MatmulPreference::kAuto;
    else if(backend=="custom")op.matmul_preference=MatmulPreference::kCustom;
    else throw std::invalid_argument("CUDA matmul supports auto or custom");
    return cuda_primitive(lhs,op,&rhs);
  },nb::arg("lhs"),nb::arg("rhs"),nb::arg("backend")="auto");

#endif

#if TENSORCX_ENABLE_METAL
  nb::class_<tensorcx::metal::MetalTensor>(module, "MetalTensor")
      .def_prop_ro("shape", [](const tensorcx::metal::MetalTensor& tensor) {
        return shape_tuple(tensor.shape());
      })
      .def_prop_ro("strides", [](const tensorcx::metal::MetalTensor& tensor) {
        return shape_tuple(tensor.strides());
      })
      .def_prop_ro("dtype",
                   [](const tensorcx::metal::MetalTensor& tensor) {
                     return std::string(tensorcx::dtype_name(tensor.dtype()));
                   })
      .def_prop_ro("device", [](const tensorcx::metal::MetalTensor&) { return "metal"; })
      .def_prop_ro("nbytes",
                   [](const tensorcx::metal::MetalTensor& tensor) {
                     return tensor.nbytes();
                   })
      .def("__add__", [](const tensorcx::metal::MetalTensor& lhs,
                         const tensorcx::metal::MetalTensor& rhs) {
        return binary_op(lhs, rhs, OpKind::kAdd);
      })
      .def("__mul__", [](const tensorcx::metal::MetalTensor& lhs,
                         const tensorcx::metal::MetalTensor& rhs) {
        return binary_op(lhs, rhs, OpKind::kMultiply);
      });

  bind_tensor_extensions<tensorcx::metal::MetalTensor>(module, metal_single_input_backend_op);
  bind_multi_input_tensor_ops<tensorcx::metal::MetalTensor, tensorcx::metal::MetalBackend>(
      module, tensorcx::metal::from_core_tensor);
  module.def("cpu_to_metal",
             [](const CpuTensor& tensor) {
               return unwrap(without_gil([&] { return tensorcx::metal::from_cpu(tensor); }));
             },
             nb::arg("tensor"));
  module.def("metal_to_cpu",
             [](const tensorcx::metal::MetalTensor& tensor) {
               return unwrap(without_gil([&] { return tensorcx::metal::to_cpu(tensor); }));
             },
             nb::arg("tensor"));
  module.def("add",
             [](const tensorcx::metal::MetalTensor& lhs, const tensorcx::metal::MetalTensor& rhs) {
               return binary_op(lhs, rhs, OpKind::kAdd);
             },
             nb::arg("lhs"),
             nb::arg("rhs"));
  module.def("multiply",
             [](const tensorcx::metal::MetalTensor& lhs, const tensorcx::metal::MetalTensor& rhs) {
               return binary_op(lhs, rhs, OpKind::kMultiply);
             },
             nb::arg("lhs"),
             nb::arg("rhs"));
  module.def("exp",
             [](const tensorcx::metal::MetalTensor& input) {
               return unary_op(input, OpKind::kExp);
             },
             nb::arg("input"));
  module.def("gelu",
             [](const tensorcx::metal::MetalTensor& input) {
               return unary_op(input, OpKind::kGelu);
             },
             nb::arg("input"));
  module.def("silu",
             [](const tensorcx::metal::MetalTensor& input) {
               return unary_op(input, OpKind::kSilu);
             },
             nb::arg("input"));
  module.def("softmax",
             [](const tensorcx::metal::MetalTensor& input, std::int64_t axis) {
               return metal_single_input_backend_op(input, OpDesc{OpKind::kSoftmax, axis});
             },
             nb::arg("input"),
             nb::arg("axis"));
  module.def("rmsnorm",
             [](const tensorcx::metal::MetalTensor& input, std::int64_t axis, double eps) {
               return metal_single_input_backend_op(input, OpDesc{OpKind::kRmsNorm, axis, eps});
             },
             nb::arg("input"),
             nb::arg("axis"),
             nb::arg("eps") = 1.0e-5);
  module.def("layernorm",
             [](const tensorcx::metal::MetalTensor& input, std::int64_t axis, double eps) {
               return metal_single_input_backend_op(input, OpDesc{OpKind::kLayerNorm, axis, eps});
             },
             nb::arg("input"),
             nb::arg("axis"),
             nb::arg("eps") = 1.0e-5);
  module.def("sum",
             [](const tensorcx::metal::MetalTensor& input, std::int64_t axis) {
               return metal_single_input_backend_op(input, OpDesc{OpKind::kSum, axis});
             },
             nb::arg("input"),
             nb::arg("axis"));
  module.def("max",
             [](const tensorcx::metal::MetalTensor& input, std::int64_t axis) {
               return metal_single_input_backend_op(input, OpDesc{OpKind::kMax, axis});
             },
             nb::arg("input"),
             nb::arg("axis"));
  module.def("mean",
             [](const tensorcx::metal::MetalTensor& input, std::int64_t axis) {
               return metal_single_input_backend_op(input, OpDesc{OpKind::kMean, axis});
             },
             nb::arg("input"),
             nb::arg("axis"));
  module.def("matmul",
             [](const tensorcx::metal::MetalTensor& lhs,
                const tensorcx::metal::MetalTensor& rhs,
                const std::string& backend) {
               return metal_matmul_backend_op(
                   lhs,
                   rhs,
                   parse_metal_matmul_preference(backend));
             },
             nb::arg("lhs"),
             nb::arg("rhs"),
             nb::arg("backend") = "auto");
#endif
  bind_dlpack(module);
}
