cmake_minimum_required(VERSION 3.4.1)  # for example
project(pytorch_dart_plugin)

set(CMAKE_CXX_STANDARD 14)  # needed for libtorch
set(CMAKE_CXX_FLAGS "-fvisibility=hidden")
add_library( pytorch_dart

# Sets the library as a shared library.
SHARED

# Provides a relative path to your source file(s).
../cgotorch/torch.cc
../cgotorch/tensor.cc)


set(LIBTORCH_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libtorch-android")
file(GLOB PYTORCH_INCLUDE_DIRS "${LIBTORCH_BASE_DIR}/include")
set(LIBTORCH_LIB_DIR "${LIBTORCH_BASE_DIR}/lib/${ANDROID_ABI}")
message(${PYTORCH_INCLUDE_DIRS})
add_library(libcplusplus SHARED IMPORTED)
set_target_properties(libcplusplus PROPERTIES IMPORTED_LOCATION ${LIBTORCH_LIB_DIR}/libc++_shared.so)

add_library(libfbjni SHARED IMPORTED)
set_target_properties(libfbjni PROPERTIES IMPORTED_LOCATION ${LIBTORCH_LIB_DIR}/libfbjni.so)

add_library(libpytorch_jni SHARED IMPORTED)
set_target_properties(libpytorch_jni PROPERTIES IMPORTED_LOCATION ${LIBTORCH_LIB_DIR}/libpytorch_jni.so)


target_include_directories(pytorch_dart PRIVATE
        ${PYTORCH_INCLUDE_DIRS}/torch/csrc/api/include
  )

  target_include_directories(pytorch_dart PRIVATE
        ${PYTORCH_INCLUDE_DIRS}
  )


target_link_libraries(pytorch_dart
        libcplusplus
        libfbjni
        libpytorch_jni
  )

