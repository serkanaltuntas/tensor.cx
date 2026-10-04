cmake_minimum_required(VERSION 3.20)

project(tensorcx VERSION 0.1.0 LANGUAGES CXX)

# Reject the old flags rather than silently building a different backend.
get_cmake_property(TENSORCX_CACHE_VARIABLES CACHE_VARIABLES)
foreach(TENSORCX_CACHE_VARIABLE IN LISTS TENSORCX_CACHE_VARIABLES)
  if(TENSORCX_CACHE_VARIABLE MATCHES "^CORTEX_")
    string(REGEX REPLACE "^CORTEX_" "TENSORCX_" TENSORCX_RENAMED_VARIABLE "${TENSORCX_CACHE_VARIABLE}")
    message(FATAL_ERROR "${TENSORCX_CACHE_VARIABLE} was renamed to ${TENSORCX_RENAMED_VARIABLE}; use a fresh build directory. See docs/NAMING.md.")
  endif()
endforeach()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Warning flags for the project's own translation units (not -Werror, and not
# applied to third-party headers, which are marked SYSTEM below).
set(TENSORCX_WARNING_FLAGS)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(TENSORCX_WARNING_FLAGS -Wall -Wextra)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

include(FetchContent)

set(TENSORCX_ENABLE_METAL "AUTO" CACHE STRING "Enable the Apple Metal backend: AUTO, ON, or OFF")
set_property(CACHE TENSORCX_ENABLE_METAL PROPERTY STRINGS AUTO ON OFF)
option(TENSORCX_ENABLE_CUDA "Build the optional CUDA prototype (requires CUDA toolkit)" OFF)
option(TENSORCX_BUILD_TESTS "Build tensor.cx C++ test executables" OFF)
option(TENSORCX_ENABLE_SANITIZERS
  "Build with AddressSanitizer and UndefinedBehaviorSanitizer (GCC/Clang only)" OFF)

if(TENSORCX_ENABLE_SANITIZERS)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "TENSORCX_ENABLE_SANITIZERS requires GCC or Clang.")
  endif()
  # Applied globally so the runtime, backends, and test executables are all
  # instrumented. The Python extension is also built with sanitizers under this
  # flag; run only the C++ test executables in sanitizer CI (importing the
  # instrumented extension requires an ASan-preloaded Python).
  # nvcc does not accept host sanitizer flags. Instrument the C++ runtime and
  # backend host code; device kernels are checked by CUDA parity/contract tests.
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined>"
    "$<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>"
  )
  add_link_options(-fsanitize=address,undefined)
endif()
string(TOUPPER "${TENSORCX_ENABLE_METAL}" TENSORCX_ENABLE_METAL_REQUEST)
if(TENSORCX_ENABLE_METAL_REQUEST STREQUAL "TRUE" OR
   TENSORCX_ENABLE_METAL_REQUEST STREQUAL "YES" OR
   TENSORCX_ENABLE_METAL_REQUEST STREQUAL "Y" OR
   TENSORCX_ENABLE_METAL_REQUEST STREQUAL "1")
  set(TENSORCX_ENABLE_METAL_REQUEST "ON")
elseif(TENSORCX_ENABLE_METAL_REQUEST STREQUAL "FALSE" OR
       TENSORCX_ENABLE_METAL_REQUEST STREQUAL "NO" OR
       TENSORCX_ENABLE_METAL_REQUEST STREQUAL "N" OR
       TENSORCX_ENABLE_METAL_REQUEST STREQUAL "0")
  set(TENSORCX_ENABLE_METAL_REQUEST "OFF")
endif()

set(TENSORCX_METAL_TOOLCHAIN_AVAILABLE OFF)
if(APPLE)
  find_program(XCRUN_EXECUTABLE xcrun)
  if(XCRUN_EXECUTABLE)
    execute_process(
      COMMAND ${XCRUN_EXECUTABLE} -sdk macosx --find metal
      RESULT_VARIABLE TENSORCX_METAL_FIND_RESULT
      OUTPUT_QUIET ERROR_QUIET
    )
    execute_process(
      COMMAND ${XCRUN_EXECUTABLE} -sdk macosx --find metallib
      RESULT_VARIABLE TENSORCX_METALLIB_FIND_RESULT
      OUTPUT_QUIET ERROR_QUIET
    )
    if(TENSORCX_METAL_FIND_RESULT EQUAL 0 AND TENSORCX_METALLIB_FIND_RESULT EQUAL 0)
      set(TENSORCX_METAL_TOOLCHAIN_AVAILABLE ON)
    endif()
  endif()
endif()

if(TENSORCX_ENABLE_METAL_REQUEST STREQUAL "AUTO")
  if(APPLE AND TENSORCX_METAL_TOOLCHAIN_AVAILABLE)
    set(TENSORCX_ENABLE_METAL ON)
  else()
    set(TENSORCX_ENABLE_METAL OFF)
    if(APPLE)
      message(STATUS
        "Apple Metal toolchain is incomplete; defaulting TENSORCX_ENABLE_METAL=OFF. "
        "Install Xcode's metal/metallib tools or configure with "
        "-DTENSORCX_ENABLE_METAL=ON to require Metal.")
    endif()
  endif()
elseif(TENSORCX_ENABLE_METAL_REQUEST STREQUAL "ON")
  if(NOT APPLE)
    message(FATAL_ERROR "TENSORCX_ENABLE_METAL=ON requires Apple platform support.")
  endif()
  if(NOT TENSORCX_METAL_TOOLCHAIN_AVAILABLE)
    message(FATAL_ERROR
      "TENSORCX_ENABLE_METAL=ON but the Metal command-line toolchain is incomplete "
      "(`xcrun -sdk macosx --find metal` and `xcrun -sdk macosx --find metallib` "
      "must both succeed). Install Xcode and its Metal toolchain, or configure "
      "with -DTENSORCX_ENABLE_METAL=OFF.")
  endif()
  set(TENSORCX_ENABLE_METAL ON)
elseif(TENSORCX_ENABLE_METAL_REQUEST STREQUAL "OFF")
  set(TENSORCX_ENABLE_METAL OFF)
else()
  message(FATAL_ERROR "TENSORCX_ENABLE_METAL must be AUTO, ON, or OFF.")
endif()

option(TENSORCX_ENABLE_MPSGRAPH "Enable the MPSGraph Metal matmul path" ${TENSORCX_ENABLE_METAL})

add_library(tensorcx_core STATIC
  cpp/tensorcx/core/backend.cpp
  cpp/tensorcx/core/dtype.cpp
  cpp/tensorcx/core/operation.cpp
  cpp/tensorcx/core/shape.cpp
  cpp/tensorcx/core/status.cpp
  cpp/tensorcx/backends/cpu/cpu_kernel.cpp
  cpp/tensorcx/backends/cpu/cpu_backend.cpp
  cpp/tensorcx/backends/cpu/cpu_buffer.cpp
  cpp/tensorcx/backends/cpu/cpu_tensor.cpp
  cpp/tensorcx/backends/null/null_backend.cpp
)
target_link_libraries(tensorcx_core PRIVATE ${CMAKE_DL_LIBS})
target_compile_features(tensorcx_core PUBLIC cxx_std_20)
target_compile_options(tensorcx_core PRIVATE ${TENSORCX_WARNING_FLAGS})
target_include_directories(tensorcx_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp
)
set_target_properties(tensorcx_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(TENSORCX_BUILD_TESTS)
  enable_testing()
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    add_library(tensorcx_cpu_kernel_fixture SHARED tests/cpp/cpu_kernel_fixture.cpp)
    add_library(tensorcx_cpu_bad_abi_fixture SHARED tests/cpp/cpu_kernel_fixture.cpp)
    target_compile_definitions(tensorcx_cpu_bad_abi_fixture PRIVATE FIXTURE_BAD_ABI)
    add_library(tensorcx_cpu_bad_symbol_fixture SHARED tests/cpp/cpu_kernel_fixture.cpp)
    target_compile_definitions(tensorcx_cpu_bad_symbol_fixture PRIVATE FIXTURE_BAD_SYMBOL)
    add_executable(tensorcx_cpu_kernel_tests tests/cpp/test_cpu_kernel.cpp)
    target_link_libraries(tensorcx_cpu_kernel_tests PRIVATE tensorcx_core)
    add_dependencies(tensorcx_cpu_kernel_tests tensorcx_cpu_kernel_fixture
                     tensorcx_cpu_bad_abi_fixture tensorcx_cpu_bad_symbol_fixture)
    add_test(NAME tensorcx_cpu_kernel_tests COMMAND tensorcx_cpu_kernel_tests
      $<TARGET_FILE:tensorcx_cpu_kernel_fixture> $<TARGET_FILE:tensorcx_cpu_bad_abi_fixture>
      $<TARGET_FILE:tensorcx_cpu_bad_symbol_fixture>)
  endif()
  add_executable(tensorcx_backend_contract_tests
    tests/cpp/test_backend_contract.cpp
  )
  target_compile_features(tensorcx_backend_contract_tests PRIVATE cxx_std_20)
  target_compile_options(tensorcx_backend_contract_tests PRIVATE ${TENSORCX_WARNING_FLAGS})
  target_link_libraries(tensorcx_backend_contract_tests PRIVATE tensorcx_core)
  add_test(NAME tensorcx_backend_contract_tests COMMAND tensorcx_backend_contract_tests)
  set_tests_properties(tensorcx_backend_contract_tests PROPERTIES TIMEOUT 30)
endif()

if(TENSORCX_ENABLE_METAL)
  FetchContent_Declare(
    metal_cpp
    GIT_REPOSITORY https://github.com/bkaradzic/metal-cpp.git
    GIT_TAG c9727bc9468a90d7ea8fc89d5ee03b8d8992a570
  )
  FetchContent_MakeAvailable(metal_cpp)

  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
  find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
  find_library(METAL_FRAMEWORK Metal REQUIRED)
  if(TENSORCX_ENABLE_MPSGRAPH)
    enable_language(OBJCXX)
    find_library(MPSGRAPH_FRAMEWORK MetalPerformanceShadersGraph REQUIRED)
  endif()

  set(TENSORCX_METAL_KERNEL_SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tensorcx/backends/metal/kernels/elementwise.metal
  )
  set(TENSORCX_METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/tensorcx_elementwise.air)
  set(TENSORCX_METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/tensorcx_elementwise.metallib)
  set(TENSORCX_METAL_EMBEDDED_CPP
    ${CMAKE_CURRENT_BINARY_DIR}/tensorcx_elementwise_metallib.cpp
  )

  add_custom_command(
    OUTPUT ${TENSORCX_METAL_AIR}
    COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metal
            -c ${TENSORCX_METAL_KERNEL_SOURCE}
            -o ${TENSORCX_METAL_AIR}
    DEPENDS ${TENSORCX_METAL_KERNEL_SOURCE}
    VERBATIM
  )
  add_custom_command(
    OUTPUT ${TENSORCX_METAL_LIB}
    COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metallib
            ${TENSORCX_METAL_AIR}
            -o ${TENSORCX_METAL_LIB}
    DEPENDS ${TENSORCX_METAL_AIR}
    VERBATIM
  )
  add_custom_command(
    OUTPUT ${TENSORCX_METAL_EMBEDDED_CPP}
    COMMAND ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py
            --input ${TENSORCX_METAL_LIB}
            --output ${TENSORCX_METAL_EMBEDDED_CPP}
            --include tensorcx/backends/metal/metal_kernels_data.h
            --namespace tensorcx::metal
            --array kElementwiseMetallib
            --size kElementwiseMetallibSize
    DEPENDS ${TENSORCX_METAL_LIB}
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py
            ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tensorcx/backends/metal/metal_kernels_data.h
    VERBATIM
  )

  set(TENSORCX_METAL_SOURCES
    ${TENSORCX_METAL_EMBEDDED_CPP}
    cpp/tensorcx/backends/metal/metal_backend.cpp
    cpp/tensorcx/backends/metal/metal_buffer.cpp
    cpp/tensorcx/backends/metal/metal_context.cpp
    cpp/tensorcx/backends/metal/metal_cpp.cpp
    cpp/tensorcx/backends/metal/metal_kernels.cpp
    cpp/tensorcx/backends/metal/metal_library.cpp
    cpp/tensorcx/backends/metal/metal_tensor.cpp
  )
  if(TENSORCX_ENABLE_MPSGRAPH)
    list(APPEND TENSORCX_METAL_SOURCES
      cpp/tensorcx/backends/metal/metal_mpsgraph.mm
    )
  endif()

  add_library(tensorcx_metal STATIC
    ${TENSORCX_METAL_SOURCES}
  )
  target_compile_features(tensorcx_metal PUBLIC cxx_std_20)
  target_compile_options(tensorcx_metal PRIVATE ${TENSORCX_WARNING_FLAGS})
  target_include_directories(tensorcx_metal PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
  )
  # metal-cpp is third-party; mark it SYSTEM so its headers don't trip our warnings.
  target_include_directories(tensorcx_metal SYSTEM PUBLIC
    ${metal_cpp_SOURCE_DIR}
  )
  target_compile_definitions(tensorcx_metal PUBLIC
    TENSORCX_ENABLE_METAL=1
    TENSORCX_ENABLE_MPSGRAPH=$<BOOL:${TENSORCX_ENABLE_MPSGRAPH}>
  )
  target_link_libraries(tensorcx_metal PUBLIC
    tensorcx_core
    ${FOUNDATION_FRAMEWORK}
    ${QUARTZCORE_FRAMEWORK}
    ${METAL_FRAMEWORK}
  )
  if(TENSORCX_ENABLE_MPSGRAPH)
    target_link_libraries(tensorcx_metal PUBLIC ${MPSGRAPH_FRAMEWORK})
    set_source_files_properties(
      cpp/tensorcx/backends/metal/metal_mpsgraph.mm
      PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )
  endif()
  set_target_properties(tensorcx_metal PROPERTIES POSITION_INDEPENDENT_CODE ON)
  if(TENSORCX_BUILD_TESTS)
    add_executable(tensorcx_metal_backend_contract_tests
      tests/cpp/test_metal_backend_contract.cpp
    )
    target_compile_features(tensorcx_metal_backend_contract_tests PRIVATE cxx_std_20)
    target_compile_options(tensorcx_metal_backend_contract_tests PRIVATE ${TENSORCX_WARNING_FLAGS})
    target_link_libraries(tensorcx_metal_backend_contract_tests PRIVATE tensorcx_metal)
    add_test(NAME tensorcx_metal_backend_contract_tests COMMAND tensorcx_metal_backend_contract_tests)
    set_tests_properties(tensorcx_metal_backend_contract_tests PROPERTIES SKIP_RETURN_CODE 77)
  endif()
else()
  set(TENSORCX_ENABLE_METAL OFF)
  set(TENSORCX_ENABLE_MPSGRAPH OFF)
endif()

if(TENSORCX_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit 12.4 REQUIRED)
  add_library(tensorcx_cuda STATIC
    cpp/tensorcx/backends/cuda/cuda_backend.cpp
    cpp/tensorcx/backends/cuda/cuda_context.cpp
    cpp/tensorcx/backends/cuda/cuda_kernel.cpp
    cpp/tensorcx/backends/cuda/kernels/elementwise.cu
    cpp/tensorcx/backends/cuda/kernels/primitives.cu
  )
  target_compile_features(tensorcx_cuda PUBLIC cxx_std_20)
  target_compile_options(tensorcx_cuda PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${TENSORCX_WARNING_FLAGS}>")
  target_link_libraries(tensorcx_cuda PUBLIC tensorcx_core CUDA::cudart CUDA::cuda_driver)
  set_target_properties(tensorcx_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
  )
  if(TENSORCX_BUILD_TESTS)
    add_executable(tensorcx_cuda_kernel_tests tests/cpp/test_cuda_kernel.cpp)
    target_link_libraries(tensorcx_cuda_kernel_tests PRIVATE tensorcx_cuda)
    add_test(NAME tensorcx_cuda_kernel_tests COMMAND tensorcx_cuda_kernel_tests
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/fixtures/cuda_add_sm52.ptx")
    set_tests_properties(tensorcx_cuda_kernel_tests PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(tensorcx_cuda_backend_contract_tests tests/cpp/test_cuda_backend_contract.cpp)
    target_link_libraries(tensorcx_cuda_backend_contract_tests PRIVATE tensorcx_cuda)
    add_test(NAME tensorcx_cuda_backend_contract_tests COMMAND tensorcx_cuda_backend_contract_tests)
    set_tests_properties(tensorcx_cuda_backend_contract_tests PROPERTIES SKIP_RETURN_CODE 77)
  endif()
endif()

nanobind_add_module(_core
  bindings/python_module.cpp
)
target_compile_features(_core PRIVATE cxx_std_20)
target_link_libraries(_core PRIVATE tensorcx_core)
target_compile_definitions(_core PRIVATE
  TENSORCX_RUNTIME_VERSION="${PROJECT_VERSION}"
  TENSORCX_ENABLE_CUDA=$<BOOL:${TENSORCX_ENABLE_CUDA}>
  TENSORCX_ENABLE_METAL=$<BOOL:${TENSORCX_ENABLE_METAL}>
  TENSORCX_ENABLE_MPSGRAPH=$<BOOL:${TENSORCX_ENABLE_MPSGRAPH}>
)

if(TENSORCX_ENABLE_METAL)
  target_link_libraries(_core PRIVATE tensorcx_metal)
endif()

if(TENSORCX_ENABLE_CUDA)
  target_link_libraries(_core PRIVATE tensorcx_cuda)
endif()

install(TARGETS _core LIBRARY DESTINATION tensorcx)
