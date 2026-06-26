#include <Python.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "cortex/backends/cpu/cpu_backend.h"
#include "cortex/backends/cpu/cpu_tensor.h"
#include "cortex/backends/null/null_backend.h"
#include "cortex/core/dtype.h"
#include "cortex/core/shape.h"
#include "cortex/core/status.h"

#if CORTEX_ENABLE_METAL
#include "cortex/backends/metal/metal_backend.h"
#include "cortex/backends/metal/metal_kernels.h"
#include "cortex/backends/metal/metal_library.h"
#if CORTEX_ENABLE_MPSGRAPH
#include "cortex/backends/metal/metal_mpsgraph.h"
#endif
#include "cortex/backends/metal/metal_tensor.h"
#endif

#ifndef CORTEX_RUNTIME_VERSION
#define CORTEX_RUNTIME_VERSION "0+unknown"
#endif

namespace nb = nanobind;

namespace {

using cortex::DType;
using cortex::OpDesc;
using cortex::OpKind;
using cortex::Shape;
using cortex::cpu::CpuTensor;

DType parse_dtype(nb::handle dtype, DType inferred) {
  if (dtype.is_none()) {
    return inferred;
  }

  if (!nb::isinstance<nb::str>(dtype)) {
    throw std::invalid_argument("unsupported dtype: expected float32 or int32");
  }
  const std::string name = nb::cast<std::string>(dtype);
  if (name == "float32") {
    return DType::kFloat32;
  }
  if (name == "int32") {
    return DType::kInt32;
  }
  throw std::invalid_argument("unsupported dtype: expected float32 or int32");
}

bool is_index_like(nb::handle item) {
  return PyBool_Check(item.ptr()) || PyIndex_Check(item.ptr());
}

// Keep invalid shape input inside the public ValueError taxonomy instead of
// letting nanobind expose std::bad_cast as RuntimeError. Use Python's index
// protocol so NumPy integer scalars behave like Python ints, while bools stay
// rejected as shape dimensions.
cortex::Dim cast_dim_or_throw(nb::handle item) {
  if (PyBool_Check(item.ptr())) {
    throw std::invalid_argument("shape dimensions must be integers");
  }

  PyObject* index_value = PyNumber_Index(item.ptr());
  if (index_value == nullptr) {
    PyErr_Clear();
    throw std::invalid_argument("shape dimensions must be integers");
  }
  nb::object index = nb::steal<nb::object>(index_value);
  try {
    return nb::cast<cortex::Dim>(index);
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
  try {
    return nb::cast<std::int32_t>(item);
  } catch (const std::exception&) {
    throw std::invalid_argument("integer value is out of range for int32 or is not an integer");
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

CpuTensor tensor_from_sequence(nb::handle data, nb::handle dtype, const std::string& device) {
  validate_cpu_device(device);

  std::vector<nb::object> items;
  bool saw_float = false;
  for (nb::handle item : nb::iter(data)) {
    if (nb::isinstance<nb::list>(item) || nb::isinstance<nb::tuple>(item)) {
      throw std::invalid_argument(
          "tensor() native factory expects a flat numeric sequence; build "
          "higher-rank tensors through cortex_runtime.tensor()");
    }
    items.emplace_back(nb::borrow<nb::object>(item));
    if (nb::isinstance<nb::float_>(item)) {
      saw_float = true;
    }
  }

  const DType inferred = saw_float ? DType::kFloat32 : DType::kInt32;
  const DType actual_dtype = parse_dtype(dtype, inferred);
  Shape shape{static_cast<cortex::Dim>(items.size())};

  switch (actual_dtype) {
    case DType::kFloat32: {
      std::vector<float> values;
      values.reserve(items.size());
      for (const nb::object& item : items) {
        values.push_back(nb::cast<float>(item));
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
  validate_cpu_device(device);

  const Shape parsed_shape = parse_shape(shape);
  const std::int64_t expected_size = cortex::numel(parsed_shape);
  const DType actual_dtype = parse_dtype(dtype, DType::kFloat32);

  switch (actual_dtype) {
    case DType::kFloat32: {
      std::vector<float> values;
      for (nb::handle item : nb::iter(data)) {
        values.push_back(nb::cast<float>(item));
      }
      if (static_cast<std::int64_t>(values.size()) != expected_size) {
        throw std::invalid_argument("tensor data length does not match shape");
      }
      return CpuTensor(parsed_shape, std::move(values));
    }
    case DType::kInt32: {
      std::vector<std::int32_t> values;
      for (nb::handle item : nb::iter(data)) {
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
nb::object make_numpy_array(const std::vector<T>& data, const Shape& shape) {
  const std::size_t ndim = shape.size();
  std::vector<std::size_t> dims(ndim);
  for (std::size_t i = 0; i < ndim; ++i) {
    dims[i] = static_cast<std::size_t>(shape[i]);
  }

  T* owned = new T[data.size()];
  if (!data.empty()) {
    std::memcpy(owned, data.data(), data.size() * sizeof(T));
  }
  nb::capsule owner(owned, [](void* ptr) noexcept { delete[] static_cast<T*>(ptr); });
  return nb::cast(nb::ndarray<nb::numpy, T>(owned, ndim, dims.data(), owner));
}

nb::object tensor_to_numpy(const CpuTensor& tensor) {
  switch (tensor.dtype()) {
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

void throw_status(const cortex::Status& status) {
  switch (status.code()) {
    case cortex::StatusCode::kInvalidArgument:
      throw std::invalid_argument(status.message());
    case cortex::StatusCode::kUnavailable:
    case cortex::StatusCode::kInternal:
      throw std::runtime_error(status.message());
    case cortex::StatusCode::kOk:
      break;
  }
  throw std::runtime_error(status.message());
}

CpuTensor binary_op(const CpuTensor& lhs, const CpuTensor& rhs, OpKind kind) {
  cortex::cpu::CpuBackend backend;
  std::array<cortex::Tensor, 2> inputs{
      cortex::cpu::to_core_tensor(lhs),
      cortex::cpu::to_core_tensor(rhs),
  };
  std::array<cortex::Tensor, 1> outputs{};
  const cortex::BackendExecution execution{
      cortex::BackendOpClass::kPrimitive,
      OpDesc{kind},
      std::span<const cortex::Tensor>(inputs.data(), inputs.size()),
      std::span<cortex::Tensor>(outputs.data(), outputs.size()),
      std::nullopt,
      std::nullopt,
  };
  const auto status = backend.execute(execution);
  if (!status.ok()) {
    throw_status(status);
  }
  return cortex::cpu::from_core_tensor(outputs[0]);
}

CpuTensor unary_op(const CpuTensor& input, OpKind kind) {
  return cortex::cpu::execute_unary(OpDesc{kind}, input);
}

CpuTensor axis_unary_op(const CpuTensor& input, OpKind kind, std::int64_t axis) {
  return cortex::cpu::execute_unary(OpDesc{kind, axis}, input);
}

CpuTensor norm_op(const CpuTensor& input, OpKind kind, std::int64_t axis, double epsilon) {
  return cortex::cpu::execute_unary(OpDesc{kind, axis, epsilon}, input);
}

CpuTensor reduction_op(const CpuTensor& input, OpKind kind, std::int64_t axis) {
  return cortex::cpu::reduce(OpDesc{kind, axis}, input);
}

CpuTensor matmul_cpu(const CpuTensor& lhs, const CpuTensor& rhs, const std::string& backend) {
  if (backend != "auto" && backend != "cpu" && backend != "reference") {
    throw std::invalid_argument("CPU matmul only supports backend='auto', 'cpu', or 'reference'");
  }
  return cortex::cpu::matmul(lhs, rhs);
}

#if CORTEX_ENABLE_METAL
std::vector<cortex::metal::KernelArgument> parse_metal_kernel_arguments(nb::sequence arguments) {
  std::vector<cortex::metal::KernelArgument> parsed;
  parsed.reserve(nb::len(arguments));
  for (nb::handle item : arguments) {
    if (nb::isinstance<cortex::metal::MetalTensor>(item)) {
      const auto& tensor = nb::cast<const cortex::metal::MetalTensor&>(item);
      parsed.push_back(cortex::metal::KernelArgument{
          cortex::metal::KernelArgument::Kind::kTensor,
          &tensor,
          0,
      });
      continue;
    }
    parsed.push_back(cortex::metal::KernelArgument{
        cortex::metal::KernelArgument::Kind::kUInt32,
        nullptr,
        cast_uint32_or_throw(item, "kernel scalar arguments must be uint32"),
    });
  }
  return parsed;
}

template <typename T>
T unwrap(cortex::Expected<T> result) {
  if (!result) {
    throw_status(result.status());
  }
  return result.move_value();
}
#endif

}  // namespace

NB_MODULE(_core, module) {
  module.doc() = "Native extension module for Cortex Runtime.";
  module.def("version", []() { return CORTEX_RUNTIME_VERSION; });
  module.attr("float32") = "float32";
  module.attr("int32") = "int32";
  module.def("_backend_contract_smoke_test", []() {
    const auto status = cortex::null_backend::contract_smoke_test();
    if (!status.ok()) {
      throw std::runtime_error(status.message());
    }
    return true;
  });

  nb::class_<CpuTensor>(module, "CpuTensor")
      .def_prop_ro("shape", [](const CpuTensor& tensor) { return shape_tuple(tensor.shape()); })
      .def_prop_ro("strides", [](const CpuTensor& tensor) { return shape_tuple(tensor.strides()); })
      .def_prop_ro("dtype",
                   [](const CpuTensor& tensor) {
                     return std::string(cortex::dtype_name(tensor.dtype()));
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
               return cortex::cpu::empty(parse_shape(shape), parse_dtype(dtype, DType::kFloat32));
             },
             nb::arg("shape"),
             nb::arg("dtype") = "float32",
             nb::arg("device") = "cpu");
  module.def("zeros",
             [](nb::handle shape, nb::handle dtype, const std::string& device) {
               validate_cpu_device(device);
               return cortex::cpu::fill(parse_shape(shape), parse_dtype(dtype, DType::kFloat32), 0.0);
             },
             nb::arg("shape"),
             nb::arg("dtype") = "float32",
             nb::arg("device") = "cpu");
  module.def("ones",
             [](nb::handle shape, nb::handle dtype, const std::string& device) {
               validate_cpu_device(device);
               return cortex::cpu::fill(parse_shape(shape), parse_dtype(dtype, DType::kFloat32), 1.0);
             },
             nb::arg("shape"),
             nb::arg("dtype") = "float32",
             nb::arg("device") = "cpu");
  module.def("fill",
             [](nb::handle shape, nb::handle dtype, double value, const std::string& device) -> nb::object {
               const Shape parsed_shape = parse_shape(shape);
               const DType parsed_dtype = parse_dtype(dtype, DType::kFloat32);
               if (device == "cpu") {
                 return nb::cast(cortex::cpu::fill(parsed_shape, parsed_dtype, value));
               }
#if CORTEX_ENABLE_METAL
               if (device == "metal") {
                 return nb::cast(unwrap(cortex::metal::fill(
                     OpDesc{OpKind::kFill}, parsed_shape, parsed_dtype, value)));
               }
#endif
               throw std::invalid_argument("device is not available: " + device);
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
               nb::list result;
               if (device == "cpu") {
                 result.append("auto");
                 result.append("cpu");
                 result.append("reference");
                 return result;
               }
#if CORTEX_ENABLE_METAL
               if (device == "metal") {
                 result.append("auto");
                 result.append("custom");
#if CORTEX_ENABLE_MPSGRAPH
                 result.append("optimized");
#endif
                 return result;
               }
#endif
               throw std::invalid_argument("device is not available: " + device);
             },
             nb::arg("device"));
  module.def("matmul",
             &matmul_cpu,
             nb::arg("lhs"),
             nb::arg("rhs"),
             nb::arg("backend") = "auto");
  module.def("is_available",
             [](const std::string& device) {
               if (device == "cpu") {
                 return true;
               }
#if CORTEX_ENABLE_METAL
               if (device == "metal") {
                 return cortex::metal::available();
               }
#endif
               return false;
             },
             nb::arg("device"));
  module.def("devices", []() {
    nb::list result;
    result.append("cpu");
#if CORTEX_ENABLE_METAL
    if (cortex::metal::available()) {
      result.append("metal");
    }
#endif
    return result;
  });
  module.def("device_name",
             [](const std::string& device) {
               if (device == "cpu") {
                 return std::string("CPU");
               }
#if CORTEX_ENABLE_METAL
               if (device == "metal") {
                 const auto names = cortex::metal::devices();
                 if (!names.empty()) {
                   return names.front();
                 }
               }
#endif
               throw std::invalid_argument("device is not available: " + device);
             },
             nb::arg("device"));
  module.def("validate_metal_library_function",
             [](nb::bytes metallib, const std::string& function_name) {
#if CORTEX_ENABLE_METAL
               return unwrap(cortex::metal::validate_library_function(
                   bytes_to_vector(metallib), function_name));
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
                nb::handle thread_count,
                nb::handle threads_per_threadgroup) {
#if CORTEX_ENABLE_METAL
               return unwrap(cortex::metal::launch_library_function(
                   bytes_to_vector(metallib),
                   function_name,
                   parse_metal_kernel_arguments(arguments),
                   cast_uint32_or_throw(thread_count, "thread count must be uint32"),
                   cast_uint32_or_throw(
                       threads_per_threadgroup,
                       "threads per threadgroup must be uint32")));
#else
               (void)metallib;
               (void)function_name;
               (void)arguments;
               (void)thread_count;
               (void)threads_per_threadgroup;
               throw std::runtime_error("Metal is not available on this system");
#endif
             },
             nb::arg("metallib"),
             nb::arg("function_name"),
             nb::arg("arguments"),
             nb::arg("thread_count"),
             nb::arg("threads_per_threadgroup"));

#if CORTEX_ENABLE_METAL
  nb::class_<cortex::metal::MetalTensor>(module, "MetalTensor")
      .def_prop_ro("shape", [](const cortex::metal::MetalTensor& tensor) {
        return shape_tuple(tensor.shape());
      })
      .def_prop_ro("strides", [](const cortex::metal::MetalTensor& tensor) {
        return shape_tuple(tensor.strides());
      })
      .def_prop_ro("dtype",
                   [](const cortex::metal::MetalTensor& tensor) {
                     return std::string(cortex::dtype_name(tensor.dtype()));
                   })
      .def_prop_ro("device", [](const cortex::metal::MetalTensor&) { return "metal"; })
      .def_prop_ro("nbytes",
                   [](const cortex::metal::MetalTensor& tensor) {
                     return tensor.nbytes();
                   })
      .def("__add__", [](const cortex::metal::MetalTensor& lhs,
                         const cortex::metal::MetalTensor& rhs) {
        return unwrap(cortex::metal::execute_binary(OpDesc{OpKind::kAdd}, lhs, rhs));
      })
      .def("__mul__", [](const cortex::metal::MetalTensor& lhs,
                         const cortex::metal::MetalTensor& rhs) {
        return unwrap(cortex::metal::execute_binary(OpDesc{OpKind::kMultiply}, lhs, rhs));
      });

  module.def("cpu_to_metal",
             [](const CpuTensor& tensor) { return unwrap(cortex::metal::from_cpu(tensor)); },
             nb::arg("tensor"));
  module.def("metal_to_cpu",
             [](const cortex::metal::MetalTensor& tensor) {
               return unwrap(cortex::metal::to_cpu(tensor));
             },
             nb::arg("tensor"));
  module.def("add",
             [](const cortex::metal::MetalTensor& lhs, const cortex::metal::MetalTensor& rhs) {
               return unwrap(cortex::metal::execute_binary(OpDesc{OpKind::kAdd}, lhs, rhs));
             },
             nb::arg("lhs"),
             nb::arg("rhs"));
  module.def("multiply",
             [](const cortex::metal::MetalTensor& lhs, const cortex::metal::MetalTensor& rhs) {
               return unwrap(cortex::metal::execute_binary(OpDesc{OpKind::kMultiply}, lhs, rhs));
             },
             nb::arg("lhs"),
             nb::arg("rhs"));
  module.def("exp",
             [](const cortex::metal::MetalTensor& input) {
               return unwrap(cortex::metal::execute_unary(OpDesc{OpKind::kExp}, input));
             },
             nb::arg("input"));
  module.def("gelu",
             [](const cortex::metal::MetalTensor& input) {
               return unwrap(cortex::metal::execute_unary(OpDesc{OpKind::kGelu}, input));
             },
             nb::arg("input"));
  module.def("silu",
             [](const cortex::metal::MetalTensor& input) {
               return unwrap(cortex::metal::execute_unary(OpDesc{OpKind::kSilu}, input));
             },
             nb::arg("input"));
  module.def("softmax",
             [](const cortex::metal::MetalTensor& input, std::int64_t axis) {
               return unwrap(cortex::metal::execute_unary(OpDesc{OpKind::kSoftmax, axis}, input));
             },
             nb::arg("input"),
             nb::arg("axis"));
  module.def("rmsnorm",
             [](const cortex::metal::MetalTensor& input, std::int64_t axis, double eps) {
               return unwrap(cortex::metal::execute_unary(OpDesc{OpKind::kRmsNorm, axis, eps}, input));
             },
             nb::arg("input"),
             nb::arg("axis"),
             nb::arg("eps") = 1.0e-5);
  module.def("layernorm",
             [](const cortex::metal::MetalTensor& input, std::int64_t axis, double eps) {
               return unwrap(cortex::metal::execute_unary(OpDesc{OpKind::kLayerNorm, axis, eps}, input));
             },
             nb::arg("input"),
             nb::arg("axis"),
             nb::arg("eps") = 1.0e-5);
  module.def("sum",
             [](const cortex::metal::MetalTensor& input, std::int64_t axis) {
               return unwrap(cortex::metal::reduce(OpDesc{OpKind::kSum, axis}, input));
             },
             nb::arg("input"),
             nb::arg("axis"));
  module.def("max",
             [](const cortex::metal::MetalTensor& input, std::int64_t axis) {
               return unwrap(cortex::metal::reduce(OpDesc{OpKind::kMax, axis}, input));
             },
             nb::arg("input"),
             nb::arg("axis"));
  module.def("mean",
             [](const cortex::metal::MetalTensor& input, std::int64_t axis) {
               return unwrap(cortex::metal::reduce(OpDesc{OpKind::kMean, axis}, input));
             },
             nb::arg("input"),
             nb::arg("axis"));
  module.def("matmul",
             [](const cortex::metal::MetalTensor& lhs,
                const cortex::metal::MetalTensor& rhs,
                const std::string& backend) {
               if (backend == "auto") {
#if CORTEX_ENABLE_MPSGRAPH
                 return unwrap(cortex::metal::matmul_mpsgraph(lhs, rhs));
#else
                 return unwrap(cortex::metal::matmul_custom(lhs, rhs));
#endif
               }
               if (backend == "optimized") {
#if CORTEX_ENABLE_MPSGRAPH
                 return unwrap(cortex::metal::matmul_mpsgraph(lhs, rhs));
#else
                 throw std::invalid_argument("optimized Metal matmul backend is not available");
#endif
               }
               if (backend == "custom") {
                 return unwrap(cortex::metal::matmul_custom(lhs, rhs));
               }
               throw std::invalid_argument("unsupported Metal matmul backend: " + backend);
             },
             nb::arg("lhs"),
             nb::arg("rhs"),
             nb::arg("backend") = "auto");
#endif
}
