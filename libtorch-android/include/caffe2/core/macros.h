// Automatically generated header file for caffe2 macros. These
// macros are used to build the Caffe2 binary, and if you are
// building a dependent library, they will need to be set as well
// for your program to link correctly.

#pragma once

// Caffe2 version. The plan is to increment the minor version every other week
// as a track point for bugs, until we find a proper versioning cycle.

#define CAFFE2_VERSION_MAJOR 1
#define CAFFE2_VERSION_MINOR 13
#define CAFFE2_VERSION_PATCH 0

static_assert(
    CAFFE2_VERSION_MINOR < 100,
    "Programming error: you set a minor version that is too big.");
static_assert(
    CAFFE2_VERSION_PATCH < 100,
    "Programming error: you set a patch version that is too big.");

#define CAFFE2_VERSION                                         \
  (CAFFE2_VERSION_MAJOR * 10000 + CAFFE2_VERSION_MINOR * 100 + \
   CAFFE2_VERSION_PATCH)

/* #undef CAFFE2_BUILD_SHARED_LIBS */
/* #undef CAFFE2_FORCE_FALLBACK_CUDA_MPI */
/* #undef CAFFE2_HAS_MKL_DNN */
/* #undef CAFFE2_HAS_MKL_SGEMM_PACK */
/* #undef CAFFE2_PERF_WITH_AVX */
/* #undef CAFFE2_PERF_WITH_AVX2 */
/* #undef CAFFE2_PERF_WITH_AVX512 */
/* #undef CAFFE2_THREADPOOL_MAIN_IMBALANCE */
/* #undef CAFFE2_THREADPOOL_STATS */
#define CAFFE2_USE_EXCEPTION_PTR
/* #undef CAFFE2_USE_ACCELERATE */
/* #undef CAFFE2_USE_CUDNN */
#define CAFFE2_USE_EIGEN_FOR_BLAS
/* #undef CAFFE2_USE_FBCODE */
/* #undef CAFFE2_USE_GOOGLE_GLOG */
/* #undef CAFFE2_USE_LITE_PROTO */
/* #undef CAFFE2_USE_MKL */
/* #undef USE_MKLDNN */
/* #undef CAFFE2_USE_NVTX */
/* #undef CAFFE2_USE_ITT */
/* #undef CAFFE2_USE_TRT */

#ifndef EIGEN_MPL2_ONLY
#define EIGEN_MPL2_ONLY
#endif

// Useful build settings that are recorded in the compiled binary
#define CAFFE2_BUILD_STRINGS { \
  {"TORCH_VERSION", "1.13.0"}, \
  {"CXX_COMPILER", "/opt/ndk/android-ndk-r19c/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++"}, \
  {"CXX_FLAGS", "-g -DANDROID -fdata-sections -ffunction-sections -funwind-tables -fstack-protector-strong -no-canonical-prefixes -mfpu=vfpv3-d16 -fno-addrsig -march=armv7-a -mthumb -Wa,--noexecstack -Wformat -Werror=format-security -stdlib=libc++ -frtti -fexceptions  -ffunction-sections -fdata-sections -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DUSE_VULKAN_WRAPPER -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DUSE_VULKAN -DUSE_VULKAN_API -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wunused-local-typedefs -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wvla-extension -Wno-range-loop-analysis -Wno-pass-failed -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -Wconstant-conversion -Wno-invalid-partial-specialization -Wno-typedef-redefinition -Wno-unused-private-field -Wno-inconsistent-missing-override -Wno-c++14-extensions -Wno-constexpr-not-const -Wno-missing-braces -Wunused-lambda-capture -Wunused-local-typedef -Qunused-arguments -fcolor-diagnostics -fdiagnostics-color=always -fno-math-errno -fno-trapping-math -Werror=format -g0"}, \
  {"BUILD_TYPE", "Release"}, \
  {"BLAS_INFO", ""}, \
  {"LAPACK_INFO", ""}, \
  {"USE_CUDA", "OFF"}, \
  {"USE_ROCM", "OFF"}, \
  {"CUDA_VERSION", ""}, \
  {"ROCM_VERSION", ""}, \
  {"USE_CUDNN", "OFF"}, \
  {"CUDNN_VERSION", ""}, \
  {"USE_NCCL", "OFF"}, \
  {"USE_MPI", "OFF"}, \
  {"USE_GFLAGS", "OFF"}, \
  {"USE_GLOG", "OFF"}, \
  {"USE_GLOO", ""}, \
  {"USE_NNPACK", "ON"}, \
  {"USE_OPENMP", "OFF"}, \
  {"FORCE_FALLBACK_CUDA_MPI", ""}, \
  {"HAS_MKL_DNN", ""}, \
  {"HAS_MKL_SGEMM_PACK", ""}, \
  {"PERF_WITH_AVX", ""}, \
  {"PERF_WITH_AVX2", ""}, \
  {"PERF_WITH_AVX512", ""}, \
  {"USE_EXCEPTION_PTR", "1"}, \
  {"USE_ACCELERATE", ""}, \
  {"USE_EIGEN_FOR_BLAS", "ON"}, \
  {"USE_LITE_PROTO", ""}, \
  {"USE_MKL", ""}, \
  {"USE_MKLDNN", "OFF"}, \
  {"USE_NVTX", ""}, \
  {"USE_ITT", ""}, \
  {"USE_TRT", ""}, \
}
