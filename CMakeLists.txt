cmake_minimum_required(VERSION 3.20)

project(cortex_runtime VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Warning flags for the project's own translation units (not -Werror, and not
# applied to third-party headers, which are marked SYSTEM below).
set(CORTEX_WARNING_FLAGS)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CORTEX_WARNING_FLAGS -Wall -Wextra)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

include(FetchContent)

set(CORTEX_ENABLE_METAL "AUTO" CACHE STRING "Enable the Apple Metal backend: AUTO, ON, or OFF")
set_property(CACHE CORTEX_ENABLE_METAL PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${CORTEX_ENABLE_METAL}" CORTEX_ENABLE_METAL_REQUEST)
if(CORTEX_ENABLE_METAL_REQUEST STREQUAL "TRUE" OR
   CORTEX_ENABLE_METAL_REQUEST STREQUAL "YES" OR
   CORTEX_ENABLE_METAL_REQUEST STREQUAL "Y" OR
   CORTEX_ENABLE_METAL_REQUEST STREQUAL "1")
  set(CORTEX_ENABLE_METAL_REQUEST "ON")
elseif(CORTEX_ENABLE_METAL_REQUEST STREQUAL "FALSE" OR
       CORTEX_ENABLE_METAL_REQUEST STREQUAL "NO" OR
       CORTEX_ENABLE_METAL_REQUEST STREQUAL "N" OR
       CORTEX_ENABLE_METAL_REQUEST STREQUAL "0")
  set(CORTEX_ENABLE_METAL_REQUEST "OFF")
endif()

set(CORTEX_METAL_TOOLCHAIN_AVAILABLE OFF)
if(APPLE)
  find_program(XCRUN_EXECUTABLE xcrun)
  if(XCRUN_EXECUTABLE)
    execute_process(
      COMMAND ${XCRUN_EXECUTABLE} -sdk macosx --find metal
      RESULT_VARIABLE CORTEX_METAL_FIND_RESULT
      OUTPUT_QUIET ERROR_QUIET
    )
    execute_process(
      COMMAND ${XCRUN_EXECUTABLE} -sdk macosx --find metallib
      RESULT_VARIABLE CORTEX_METALLIB_FIND_RESULT
      OUTPUT_QUIET ERROR_QUIET
    )
    if(CORTEX_METAL_FIND_RESULT EQUAL 0 AND CORTEX_METALLIB_FIND_RESULT EQUAL 0)
      set(CORTEX_METAL_TOOLCHAIN_AVAILABLE ON)
    endif()
  endif()
endif()

if(CORTEX_ENABLE_METAL_REQUEST STREQUAL "AUTO")
  if(APPLE AND CORTEX_METAL_TOOLCHAIN_AVAILABLE)
    set(CORTEX_ENABLE_METAL ON)
  else()
    set(CORTEX_ENABLE_METAL OFF)
    if(APPLE)
      message(STATUS
        "Apple Metal toolchain is incomplete; defaulting CORTEX_ENABLE_METAL=OFF. "
        "Install Xcode's metal/metallib tools or configure with "
        "-DCORTEX_ENABLE_METAL=ON to require Metal.")
    endif()
  endif()
elseif(CORTEX_ENABLE_METAL_REQUEST STREQUAL "ON")
  if(NOT APPLE)
    message(FATAL_ERROR "CORTEX_ENABLE_METAL=ON requires Apple platform support.")
  endif()
  if(NOT CORTEX_METAL_TOOLCHAIN_AVAILABLE)
    message(FATAL_ERROR
      "CORTEX_ENABLE_METAL=ON but the Metal command-line toolchain is incomplete "
      "(`xcrun -sdk macosx --find metal` and `xcrun -sdk macosx --find metallib` "
      "must both succeed). Install Xcode and its Metal toolchain, or configure "
      "with -DCORTEX_ENABLE_METAL=OFF.")
  endif()
  set(CORTEX_ENABLE_METAL ON)
elseif(CORTEX_ENABLE_METAL_REQUEST STREQUAL "OFF")
  set(CORTEX_ENABLE_METAL OFF)
else()
  message(FATAL_ERROR "CORTEX_ENABLE_METAL must be AUTO, ON, or OFF.")
endif()

option(CORTEX_ENABLE_MPSGRAPH "Enable the MPSGraph Metal matmul path" ${CORTEX_ENABLE_METAL})

add_library(cortex_core STATIC
  cpp/cortex/core/dtype.cpp
  cpp/cortex/core/shape.cpp
  cpp/cortex/core/status.cpp
  cpp/cortex/backends/cpu/cpu_backend.cpp
  cpp/cortex/backends/cpu/cpu_buffer.cpp
  cpp/cortex/backends/cpu/cpu_tensor.cpp
  cpp/cortex/backends/null/null_backend.cpp
)
target_compile_features(cortex_core PUBLIC cxx_std_20)
target_compile_options(cortex_core PRIVATE ${CORTEX_WARNING_FLAGS})
target_include_directories(cortex_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp
)
set_target_properties(cortex_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CORTEX_ENABLE_METAL)
  FetchContent_Declare(
    metal_cpp
    GIT_REPOSITORY https://github.com/bkaradzic/metal-cpp.git
    GIT_TAG c9727bc9468a90d7ea8fc89d5ee03b8d8992a570
  )
  FetchContent_MakeAvailable(metal_cpp)

  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
  find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
  find_library(METAL_FRAMEWORK Metal REQUIRED)
  if(CORTEX_ENABLE_MPSGRAPH)
    enable_language(OBJCXX)
    find_library(MPSGRAPH_FRAMEWORK MetalPerformanceShadersGraph REQUIRED)
  endif()

  set(CORTEX_METAL_KERNEL_SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/cortex/backends/metal/kernels/elementwise.metal
  )
  set(CORTEX_METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/cortex_elementwise.air)
  set(CORTEX_METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/cortex_elementwise.metallib)
  set(CORTEX_METAL_EMBEDDED_CPP
    ${CMAKE_CURRENT_BINARY_DIR}/cortex_elementwise_metallib.cpp
  )

  add_custom_command(
    OUTPUT ${CORTEX_METAL_AIR}
    COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metal
            -c ${CORTEX_METAL_KERNEL_SOURCE}
            -o ${CORTEX_METAL_AIR}
    DEPENDS ${CORTEX_METAL_KERNEL_SOURCE}
    VERBATIM
  )
  add_custom_command(
    OUTPUT ${CORTEX_METAL_LIB}
    COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metallib
            ${CORTEX_METAL_AIR}
            -o ${CORTEX_METAL_LIB}
    DEPENDS ${CORTEX_METAL_AIR}
    VERBATIM
  )
  add_custom_command(
    OUTPUT ${CORTEX_METAL_EMBEDDED_CPP}
    COMMAND ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py
            --input ${CORTEX_METAL_LIB}
            --output ${CORTEX_METAL_EMBEDDED_CPP}
            --include cortex/backends/metal/metal_kernels_data.h
            --namespace cortex::metal
            --array kElementwiseMetallib
            --size kElementwiseMetallibSize
    DEPENDS ${CORTEX_METAL_LIB}
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py
            ${CMAKE_CURRENT_SOURCE_DIR}/cpp/cortex/backends/metal/metal_kernels_data.h
    VERBATIM
  )

  set(CORTEX_METAL_SOURCES
    ${CORTEX_METAL_EMBEDDED_CPP}
    cpp/cortex/backends/metal/metal_backend.cpp
    cpp/cortex/backends/metal/metal_buffer.cpp
    cpp/cortex/backends/metal/metal_context.cpp
    cpp/cortex/backends/metal/metal_cpp.cpp
    cpp/cortex/backends/metal/metal_kernels.cpp
    cpp/cortex/backends/metal/metal_library.cpp
    cpp/cortex/backends/metal/metal_tensor.cpp
  )
  if(CORTEX_ENABLE_MPSGRAPH)
    list(APPEND CORTEX_METAL_SOURCES
      cpp/cortex/backends/metal/metal_mpsgraph.mm
    )
  endif()

  add_library(cortex_metal STATIC
    ${CORTEX_METAL_SOURCES}
  )
  target_compile_features(cortex_metal PUBLIC cxx_std_20)
  target_compile_options(cortex_metal PRIVATE ${CORTEX_WARNING_FLAGS})
  target_include_directories(cortex_metal PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
  )
  # metal-cpp is third-party; mark it SYSTEM so its headers don't trip our warnings.
  target_include_directories(cortex_metal SYSTEM PUBLIC
    ${metal_cpp_SOURCE_DIR}
  )
  target_compile_definitions(cortex_metal PUBLIC
    CORTEX_ENABLE_METAL=1
    CORTEX_ENABLE_MPSGRAPH=$<BOOL:${CORTEX_ENABLE_MPSGRAPH}>
  )
  target_link_libraries(cortex_metal PUBLIC
    cortex_core
    ${FOUNDATION_FRAMEWORK}
    ${QUARTZCORE_FRAMEWORK}
    ${METAL_FRAMEWORK}
  )
  if(CORTEX_ENABLE_MPSGRAPH)
    target_link_libraries(cortex_metal PUBLIC ${MPSGRAPH_FRAMEWORK})
    set_source_files_properties(
      cpp/cortex/backends/metal/metal_mpsgraph.mm
      PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )
  endif()
  set_target_properties(cortex_metal PROPERTIES POSITION_INDEPENDENT_CODE ON)
else()
  set(CORTEX_ENABLE_METAL OFF)
  set(CORTEX_ENABLE_MPSGRAPH OFF)
endif()

nanobind_add_module(_core
  bindings/python_module.cpp
)
target_compile_features(_core PRIVATE cxx_std_20)
target_link_libraries(_core PRIVATE cortex_core)
target_compile_definitions(_core PRIVATE
  CORTEX_RUNTIME_VERSION="${PROJECT_VERSION}"
  CORTEX_ENABLE_METAL=$<BOOL:${CORTEX_ENABLE_METAL}>
  CORTEX_ENABLE_MPSGRAPH=$<BOOL:${CORTEX_ENABLE_MPSGRAPH}>
)

if(CORTEX_ENABLE_METAL)
  target_link_libraries(_core PRIVATE cortex_metal)
endif()

install(TARGETS _core LIBRARY DESTINATION cortex_runtime)
