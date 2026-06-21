cmake_minimum_required(VERSION 3.20)

project(cortex_runtime VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

add_library(cortex_core STATIC
  cpp/cortex/core/dtype.cpp
  cpp/cortex/core/shape.cpp
  cpp/cortex/core/status.cpp
  cpp/cortex/backends/cpu/cpu_backend.cpp
  cpp/cortex/backends/cpu/cpu_buffer.cpp
  cpp/cortex/backends/cpu/cpu_tensor.cpp
)
target_compile_features(cortex_core PUBLIC cxx_std_20)
target_include_directories(cortex_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp
)
set_target_properties(cortex_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

nanobind_add_module(_core
  bindings/python_module.cpp
)
target_compile_features(_core PRIVATE cxx_std_20)
target_link_libraries(_core PRIVATE cortex_core)
target_compile_definitions(_core PRIVATE
  CORTEX_RUNTIME_VERSION="${PROJECT_VERSION}"
)

install(TARGETS _core LIBRARY DESTINATION cortex_runtime)
