cmake_minimum_required(VERSION 3.4.1)  # for example
project(native_pytorch)

set(CMAKE_CXX_STANDARD 14)  # needed for libtorch

add_library( native_pytorch

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             ../ios/Classes/native_pytorch.cpp )


set(LIBTORCH_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libtorch-android")
file(GLOB PYTORCH_INCLUDE_DIRS "${LIBTORCH_BASE_DIR}/include")
set(LIBTORCH_LIB_DIR "${LIBTORCH_BASE_DIR}/lib/${ANDROID_ABI}")

add_library(libcplusplus SHARED IMPORTED)
set_target_properties(libcplusplus PROPERTIES IMPORTED_LOCATION ${LIBTORCH_LIB_DIR}/libc++_shared.so)

add_library(libfbjni SHARED IMPORTED)
set_target_properties(libfbjni PROPERTIES IMPORTED_LOCATION ${LIBTORCH_LIB_DIR}/libfbjni.so)

add_library(libpytorch_jni SHARED IMPORTED)
set_target_properties(libpytorch_jni PROPERTIES IMPORTED_LOCATION ${LIBTORCH_LIB_DIR}/libpytorch_jni.so)


target_include_directories(native_pytorch PRIVATE
        ${PYTORCH_INCLUDE_DIRS}
  )


target_link_libraries(native_pytorch
        libcplusplus
        libfbjni
        libpytorch_jni
  )

