#pragma once

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma once
#include <limits>
#include <assert.h>
#include <cmath>
#include <cstring>

#define UNUSED(x) (void)(x)
#define DEBUG_ONLY(x) (void)(x)

#ifdef _WIN32
#include <intrin.h>

#define EXPORT_API(ret) extern "C" __declspec(dllexport) ret
#define EXPORT_API_ANDROID_OR_IOS(ret) extern "C" __declspec(dllexport) ret

#elif __ANDROID__
#define EXPORT_API(ret) ret
#define EXPORT_API_ANDROID_OR_IOS(ret) extern "C" __attribute__((visibility("default"))) ret

#elif __APPLE__
    #include "TargetConditionals.h"
    #ifdef TARGET_OS_IPHONE
    #define EXPORT_API(ret) ret
    #define EXPORT_API_ANDROID_OR_IOS(ret) extern "C" __attribute__((visibility("default"))) ret
    #endif
#else
#include "UnixSal.h"

#define EXPORT_API(ret) extern "C" __attribute__((visibility("default"))) ret
#define EXPORT_API_ANDROID_OR_IOS(ret) extern "C" __attribute__((visibility("default"))) ret
#define __forceinline __attribute__((always_inline)) inline
#endif

// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#pragma once

#include <string>

#include "torch/torch.h"

extern thread_local char *torch_last_err;

typedef torch::Tensor *Tensor;
typedef torch::Scalar *Scalar;
typedef torch::Generator* Generator;
typedef c10::Storage* Storage;
typedef torch::nn::utils::rnn::PackedSequence* PackedSequence;

typedef std::shared_ptr<torch::autograd::SavedVariable> * SavedVariable;
typedef std::shared_ptr<torch::nn::Module> * NNModule;
typedef std::shared_ptr<torch::nn::AnyModule> * NNAnyModule;
typedef std::shared_ptr<torch::optim::Optimizer> * Optimizer;
typedef std::shared_ptr<torch::jit::CompilationUnit> * JITCompilationUnit;
typedef std::shared_ptr<torch::jit::Module>* JITModule;
typedef std::shared_ptr<torch::jit::Method>* JITMethod;
typedef std::shared_ptr<torch::jit::Function> * JITFunction;
typedef std::shared_ptr<c10::Type> * JITType;
typedef std::shared_ptr<c10::TensorType>* JITTensorType;

struct TensorArray {
    Tensor *array;
    int64_t size;
};



//typedef std::shared_ptr<torch::jit::DimensionedTensorType>* JITDimensionedTensorType;

#define THS_API TH_API

#define CATCH(x) \
  try { \
    torch_last_err = 0; \
    x \
  } catch (const c10::Error e) { \
      torch_last_err = strdup(e.what()); \
  } catch (const std::runtime_error e) { \
      torch_last_err = strdup(e.what()); \
  }

#define CATCH_RETURN_RES(ty, dflt, stmt) \
    ty res = dflt; \
    CATCH(  \
        stmt;  \
    );  \
    return res;

#define CATCH_RETURN(ty, dflt, expr) CATCH_RETURN_RES(ty, dflt, res = expr)
#define CATCH_RETURN_NNModule(stmt) CATCH_RETURN_RES(NNModule, nullptr, stmt)
#define CATCH_RETURN_Tensor(stmt) CATCH_RETURN_RES(Tensor, nullptr, stmt)

// Return undefined tensors as nullptr to C#
inline Tensor ResultTensor(const at::Tensor & res)
{
    if (res.defined())
        return new torch::Tensor(res);
    else
        return nullptr;
}

#define CATCH_TENSOR(expr) \
    at::Tensor res = at::Tensor(); \
    CATCH(  \
        res = expr;  \
    );  \
    return ResultTensor(res);

#define CATCH_TENSORS_2(expr) \
    at::Tensor fst = at::Tensor();  \
    at::Tensor snd = at::Tensor();  \
    CATCH(  \
        std::tie(fst,snd) = expr;  \
    );  \
    res1 = ResultTensor(fst); \
    res2 = ResultTensor(snd);     

#define CATCH_SCALAR(expr) \
    at::Scalar res = at::Scalar(); \
    CATCH(  \
        res = expr;  \
    );  \
    return ResultTensor(res);


// Utility method used to built sharable strings.
const char * make_sharable_string(const std::string str);

// Method concerting arrays of tensor pointers into arrays of tensors.
template<class T>
std::vector<T> toTensors(torch::Tensor ** tensorPtrs, const int length)
{
    std::vector<T> tensors;

    if (tensorPtrs != nullptr) {
        for (int i = 0; i < length; i++)
        {
            tensors.push_back(*tensorPtrs[i]);
        }
    }
    return tensors;
}

// Utilities for NN namespace.

template <typename T>
Tensor get_weight(const NNModule module)
{
    CATCH_TENSOR((*module)->as<T>()->weight);
}

template <typename T>
void set_weight(const NNModule module, const Tensor weights)
{
    CATCH(
        (*module)->as<T>()->weight = *weights;
    );
}

template <typename T>
Tensor get_bias(const NNModule module)
{
    CATCH_TENSOR((*module)->as<T>()->bias);
}

template <typename T>
void set_bias(const NNModule module, const Tensor bias)
{
    CATCH(
        (*module)->as<T>()->bias = *bias;
    );
}

template <typename T>
Tensor get_weight_ih(const NNModule module)
{
    CATCH_TENSOR((*module)->as<T>()->weight_ih);
}

template <typename T>
Tensor get_weight_hh(const NNModule module)
{
    CATCH_TENSOR((*module)->as<T>()->weight_hh);
}

template <typename T>
void set_weight_ih(const NNModule module, const Tensor weights)
{
    CATCH(
        (*module)->as<T>()->weight_ih = *weights;
    );
}

template <typename T>
void set_weight_hh(const NNModule module, const Tensor weights)
{
    CATCH(
        (*module)->as<T>()->weight_hh = *weights;
    );
}

template <typename T>
Tensor get_bias_ih(const NNModule module)
{
    CATCH_TENSOR((*module)->as<T>()->bias_ih);
}

template <typename T>
Tensor get_bias_hh(const NNModule module)
{
    CATCH_TENSOR((*module)->as<T>()->bias_hh);
}

template <typename T>
void set_bias_ih(const NNModule module, const Tensor bias)
{
    CATCH(
        (*module)->as<T>()->bias_ih = *bias;
    );
}

template <typename T>
void set_bias_hh(const NNModule module, const Tensor bias)
{
    CATCH(
        (*module)->as<T>()->bias_hh = *bias;
    );
}

#define WIH_BASE 0
#define WHH_BASE 1
#define BIH_BASE 2
#define BHH_BASE 3

template <typename T>
Tensor get_weight_ih(const NNModule module, const int64_t idx)
{
    CATCH_TENSOR((*module)->as<T>()->all_weights()[WIH_BASE + idx * 4]);
}

template <typename T>
Tensor get_weight_hh(const NNModule module, const int64_t idx)
{
    CATCH_TENSOR((*module)->as<T>()->all_weights()[WHH_BASE + idx * 4]);
}

template <typename T>
void set_weight_ih(const NNModule module, const Tensor weights, const int64_t idx)
{
    CATCH(
        (*module)->as<T>()->all_weights()[WIH_BASE + idx * 4] = *weights;
    );
}

template <typename T>
void set_weight_hh(const NNModule module, const Tensor weights, const int64_t idx)
{
    CATCH(
        (*module)->as<T>()->all_weights()[WHH_BASE + idx * 4] = *weights;
    );
}

template <typename T>
Tensor get_bias_ih(const NNModule module, const int64_t idx)
{
    CATCH_TENSOR((*module)->as<T>()->all_weights()[BIH_BASE + idx * 4]);
}

template <typename T>
Tensor get_bias_hh(const NNModule module, const int64_t idx)
{
    CATCH_TENSOR((*module)->as<T>()->all_weights()[BHH_BASE + idx * 4]);
}

template <typename T>
void set_bias_ih(const NNModule module, const Tensor bias, const int64_t idx)
{
    CATCH(
        (*module)->as<T>()->all_weights()[BIH_BASE + idx * 4] = *bias;
    );
}

template <typename T>
void set_bias_hh(const NNModule module, const Tensor bias, const int64_t idx)
{
    CATCH(
        (*module)->as<T>()->all_weights()[BHH_BASE + idx * 4] = *bias;
    );
}

template<typename TImpl>
NNModule create_module(NNAnyModule* outAsAnyModule)
{
    auto mod = std::make_shared<TImpl>();

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != nullptr)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<TImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    return new std::shared_ptr<torch::nn::Module>(mod);
}

template<typename TImpl, typename TOptions>
NNModule create_module(const TOptions& opts, NNAnyModule* outAsAnyModule)
{
    auto mod = std::make_shared<TImpl>(opts);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != nullptr)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<TImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    return new std::shared_ptr<torch::nn::Module>(mod);
}

inline
torch::nn::init::NonlinearityType get_nl_type(const int64_t nl)
{
    switch (nl)
    {
    default:
    case 0:  return torch::kLinear;
    case 1:  return torch::kConv1D;
    case 2:  return torch::kConv2D;
    case 3:  return torch::kConv3D;
    case 4:  return torch::kConvTranspose1D;
    case 5:  return torch::kConvTranspose2D;
    case 6:  return torch::kConvTranspose3D;
    case 7:  return torch::kSigmoid;
    case 8:  return torch::kTanh;
    case 9:  return torch::kReLU;
    case 10: return torch::kLeakyReLU;
    }
}

// API.

// Sets manually the seed.
EXPORT_API(void)      THSTorch_manual_seed(const int64_t seed);
EXPORT_API(void) THSCuda_manual_seed(const int64_t seed);
EXPORT_API(void) THSCuda_manual_seed_all(const int64_t seed);

EXPORT_API(Generator) THSGenerator_manual_seed(const int64_t seed);
EXPORT_API(void) THSGenerator_gen_manual_seed(const Generator gen, const int64_t seed);

EXPORT_API(Tensor) THSGenerator_get_rng_state(const Generator gen);
EXPORT_API(void)  THSGenerator_set_rng_state(const Generator gen, const Tensor tensor);

EXPORT_API(Generator) THSGenerator_default_generator();
EXPORT_API(Generator) THSGenerator_new(uint64_t seed, int64_t device, int64_t index);
EXPORT_API(int64_t)   THSGenerator_initial_seed(const Generator gen);
EXPORT_API(void)      THSGenerator_dispose(const Generator generator);

EXPORT_API(int) THSTorchCuda_is_available();
EXPORT_API(int) THSTorchCuda_cudnn_is_available();
EXPORT_API(int) THSTorchCuda_device_count();
EXPORT_API(void) THSTorchCuda_synchronize(const int64_t device);

EXPORT_API(bool)  THSBackend_cublas_get_allow_tf32();
EXPORT_API(void)  THSBackend_cublas_set_allow_tf32(const bool flag);
EXPORT_API(bool)  THSBackend_cudnn_get_allow_tf32();
EXPORT_API(void)  THSBackend_cudnn_set_allow_tf32(const bool flag);

EXPORT_API(bool)  THSBackend_cuda_get_allow_fp16_reduced_precision_reduction();
EXPORT_API(void)  THSBackend_cuda_set_allow_fp16_reduced_precision_reduction(const bool flag);

EXPORT_API(bool)  THSBackend_cuda_get_enable_flash_sdp();
EXPORT_API(void)  THSBackend_cuda_set_enable_flash_sdp(const bool flag);
EXPORT_API(bool)  THSBackend_cuda_get_enable_math_sdp();
EXPORT_API(void)  THSBackend_cuda_set_enable_math_sdp(const bool flag);

EXPORT_API(int) THSTorch_get_num_threads();
EXPORT_API(void) THSTorch_set_num_threads(const int threads);

EXPORT_API(int) THSTorch_get_num_interop_threads();
EXPORT_API(void) THSTorch_set_num_interop_threads(const int threads);

// Returns the latest error. This is thread-local.

EXPORT_API_ANDROID_OR_IOS(const char *) THSTorch_get_and_reset_last_err();

EXPORT_API(int) THSTorch_can_cast(const int type1, const int type2);
EXPORT_API(int) THSTorch_promote_types(const int type1, const int type2);

EXPORT_API_ANDROID_OR_IOS(Scalar) THSTorch_int8_to_scalar(int8_t value);
EXPORT_API_ANDROID_OR_IOS(Scalar) THSTorch_uint8_to_scalar(uint8_t value);
EXPORT_API_ANDROID_OR_IOS(Scalar) THSTorch_int16_to_scalar(short value);
EXPORT_API_ANDROID_OR_IOS(Scalar) THSTorch_int32_to_scalar(int value);
EXPORT_API_ANDROID_OR_IOS(Scalar) THSTorch_int64_to_scalar(long value);
EXPORT_API_ANDROID_OR_IOS(Scalar) THSTorch_float32_to_scalar(float value);
EXPORT_API_ANDROID_OR_IOS(Scalar) THSTorch_float64_to_scalar(double value);
EXPORT_API_ANDROID_OR_IOS(Scalar) THSTorch_bool_to_scalar(bool value);
EXPORT_API_ANDROID_OR_IOS(Scalar) THSTorch_float16_to_scalar(float value);
EXPORT_API_ANDROID_OR_IOS(Scalar) THSTorch_bfloat16_to_scalar(float value);

EXPORT_API_ANDROID_OR_IOS(Scalar) THSTorch_complex32_to_scalar(float real, float imaginary);
EXPORT_API_ANDROID_OR_IOS(Scalar) THSTorch_complex64_to_scalar(double real, double imaginary);

EXPORT_API_ANDROID_OR_IOS(int8_t) THSTorch_scalar_to_int8(Scalar value);
EXPORT_API_ANDROID_OR_IOS(uint8_t) THSTorch_scalar_to_uint8(Scalar value);
EXPORT_API_ANDROID_OR_IOS(int16_t) THSTorch_scalar_to_int16(Scalar value);
EXPORT_API_ANDROID_OR_IOS(int32_t) THSTorch_scalar_to_int32(Scalar value);
EXPORT_API_ANDROID_OR_IOS(int64_t) THSTorch_scalar_to_int64(Scalar value);
EXPORT_API_ANDROID_OR_IOS(float) THSTorch_scalar_to_float32(Scalar value);
EXPORT_API_ANDROID_OR_IOS(double) THSTorch_scalar_to_float64(Scalar value);
EXPORT_API_ANDROID_OR_IOS(bool) THSTorch_scalar_to_bool(Scalar value);

EXPORT_API_ANDROID_OR_IOS(void) THSTorch_scalar_to_float16(Scalar value, unsigned short* res);

EXPORT_API_ANDROID_OR_IOS(void) THSTorch_scalar_to_complex32(Scalar value, float* (*allocator)(size_t length));
EXPORT_API_ANDROID_OR_IOS(void) THSTorch_scalar_to_complex64(Scalar value, double* (*allocator)(size_t length));

EXPORT_API_ANDROID_OR_IOS(int8_t) THSTorch_scalar_type(Scalar value);

// Dispose the scalar.
EXPORT_API(void) THSTorch_dispose_scalar(Scalar scalar);

// Math functions not available in .NET standard libs

EXPORT_API(double) THSSpecial_erf_scalar(const double x);
EXPORT_API(double) THSSpecial_erfc_scalar(const double x);


// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.


#include "torch/torch.h"
#include "torch/cuda.h"

void THSTorch_manual_seed(const int64_t seed)
{
    torch::manual_seed(seed);
}

Generator THSGenerator_manual_seed(const int64_t seed)
{
    auto gen = at::globalContext().defaultGenerator(at::DeviceType::CPU);
    gen.set_current_seed(seed);
    return new at::Generator(gen.getIntrusivePtr());
}
#ifdef __ANDROID__
void THSCuda_manual_seed(const int64_t seed)
{
    CATCH()
}

void THSCuda_manual_seed_all(const int64_t seed)
{
    CATCH()
}
#else
void THSCuda_manual_seed(const int64_t seed)
{
    CATCH(torch::cuda::manual_seed(seed);)
}

void THSCuda_manual_seed_all(const int64_t seed)
{
    CATCH(torch::cuda::manual_seed_all(seed);)
}
#endif
bool THSBackend_cublas_get_allow_tf32()
{
    auto result = false;
    CATCH(result = at::globalContext().allowTF32CuBLAS(););
    return result;
}

void THSBackend_cublas_set_allow_tf32(const bool flag)
{
    CATCH(at::globalContext().setAllowTF32CuBLAS(flag););
}

bool THSBackend_cudnn_get_allow_tf32()
{
    auto result = false;
    CATCH(result = at::globalContext().allowTF32CuDNN(););
    return result;
}

void THSBackend_cudnn_set_allow_tf32(const bool flag)
{
    CATCH(at::globalContext().setAllowTF32CuDNN(flag););
}

bool THSBackend_cuda_get_allow_fp16_reduced_precision_reduction()
{
    auto result = false;
    CATCH(result = at::globalContext().allowFP16ReductionCuBLAS(););
    return result;
}

void THSBackend_cuda_set_allow_fp16_reduced_precision_reduction(const bool flag)
{
    CATCH(at::globalContext().setAllowFP16ReductionCuBLAS(flag););
}

bool THSBackend_cuda_get_enable_flash_sdp()
{
    auto result = false;
    CATCH(result = at::globalContext().userEnabledFlashSDP(););
    return result;
}

void THSBackend_cuda_set_enable_flash_sdp(const bool flag)
{
    CATCH(at::globalContext().setSDPUseFlash(flag););
}

bool THSBackend_cuda_get_enable_math_sdp()
{
    auto result = false;
    CATCH(result = at::globalContext().userEnabledMathSDP(););
    return result;
}

void THSBackend_cuda_set_enable_math_sdp(const bool flag)
{
    CATCH(at::globalContext().setSDPUseMath(flag););
}

void THSGenerator_gen_manual_seed(const Generator generator, const int64_t seed)
{
    generator->set_current_seed(seed);
}

Generator THSGenerator_default_generator()
{
    auto gen = at::globalContext().defaultGenerator(at::DeviceType::CPU);
    return new at::Generator(gen.getIntrusivePtr());
}

int64_t THSGenerator_initial_seed(const Generator gen)
{
    return gen->current_seed();
}

Tensor THSGenerator_get_rng_state(const Generator gen)
{
    CATCH_TENSOR(gen->get_state());
}

void  THSGenerator_set_rng_state(const Generator gen, const Tensor tensor)
{
    gen->set_state(*tensor);
}


Generator THSGenerator_new(uint64_t seed, int64_t device, int64_t index)
{
    // TODO: Support creation of GPU RNGs. 'device' and 'index' are in the
    //       function signature in preparation thereof.
    return new at::Generator(at::detail::createCPUGenerator(seed));
}

void THSGenerator_dispose(const Generator generator)
{
    delete generator;
}

#ifdef __ANDROID__
int THSTorchCuda_is_available()
{
    return 0;
}

int THSTorchCuda_cudnn_is_available()
{
    return 0;
}

int THSTorchCuda_device_count()
{
    return 0;
}

void THSTorchCuda_synchronize(const int64_t device_index)
{
    CATCH();
}
#else

int THSTorchCuda_is_available()
{
    return torch::cuda::is_available();
}

int THSTorchCuda_cudnn_is_available()
{
    return torch::cuda::cudnn_is_available();
}

int THSTorchCuda_device_count()
{
    return (int)torch::cuda::device_count();
}

void THSTorchCuda_synchronize(const int64_t device_index)
{
    CATCH(torch::cuda::synchronize(device_index);)
}

#endif

const char * THSTorch_get_and_reset_last_err()
{
    char *tmp = torch_last_err;
    torch_last_err = nullptr;
    
    return tmp;
}

int THSTorch_get_num_threads()
{
    CATCH_RETURN_RES(int, -1, res = torch::get_num_threads());
}

void THSTorch_set_num_threads(const int threads)
{
    torch::set_num_threads(threads);
}

int THSTorch_get_num_interop_threads()
{
    CATCH_RETURN_RES(int, -1, res = torch::get_num_interop_threads());
}

void THSTorch_set_num_interop_threads(const int threads)
{
    torch::set_num_interop_threads(threads);
}

int THSTorch_can_cast(const int type1, const int type2)
{
    CATCH_RETURN_RES(int, -1, res = (int)torch::can_cast((c10::ScalarType)type1, (c10::ScalarType)type2));
}

int THSTorch_promote_types(const int type1, const int type2)
{
    CATCH_RETURN_RES(int, -1, res = (int)torch::promote_types((c10::ScalarType)type1, (c10::ScalarType)type2));
}


Scalar THSTorch_int8_to_scalar(int8_t value)
{
    return new torch::Scalar(value);
}

Scalar THSTorch_uint8_to_scalar(uint8_t value)
{
    return new torch::Scalar(value);
}

Scalar THSTorch_int16_to_scalar(short value)
{
    return new torch::Scalar(value);
}

Scalar THSTorch_int32_to_scalar(int value)
{
    return new torch::Scalar(value);
}

Scalar THSTorch_int64_to_scalar(long value)
{
    return new torch::Scalar(int64_t(value));
}

Scalar THSTorch_float32_to_scalar(float value)
{
    return new torch::Scalar(value);
}

Scalar THSTorch_float64_to_scalar(double value)
{
    return new torch::Scalar(value);
}

Scalar THSTorch_float16_to_scalar(float value)
{
    return new torch::Scalar((c10::Half)value);
}

Scalar THSTorch_bfloat16_to_scalar(float value)
{
    return new torch::Scalar((c10::BFloat16)value);
}

Scalar THSTorch_bool_to_scalar(bool value)
{
    return new torch::Scalar(value);
}

Scalar THSTorch_complex32_to_scalar(float real, float imaginary)
{
    return new torch::Scalar(c10::complex<float>(real, imaginary));
}

Scalar THSTorch_complex64_to_scalar(double real, double imaginary)
{
    return new torch::Scalar(c10::complex<double>(real, imaginary));
}

int8_t THSTorch_scalar_to_int8(Scalar value)
{
    return value->toChar();
}

uint8_t THSTorch_scalar_to_uint8(Scalar value)
{
    return value->toByte();
}

int16_t THSTorch_scalar_to_int16(Scalar value)
{
    return value->toShort();
}

int32_t THSTorch_scalar_to_int32(Scalar value)
{
    return value->toInt();
}

int64_t THSTorch_scalar_to_int64(Scalar value)
{
    return value->toLong();
}

float THSTorch_scalar_to_float32(Scalar value)
{
    return value->toFloat();
}

double THSTorch_scalar_to_float64(Scalar value)
{
    return value->toDouble();
}

void THSTorch_scalar_to_float16(Scalar value, unsigned short *res)
{
    *res = value->toHalf().x;
}

void THSTorch_scalar_to_complex32(Scalar value, float* (*allocator)(size_t length))
{
    auto result = value->toComplexFloat();
    auto space = allocator(2);
    space[0] = result.real();
    space[1] = result.imag();
}

void THSTorch_scalar_to_complex64(Scalar value, double* (*allocator)(size_t length))
{
    auto result = value->toComplexDouble();
    auto space = allocator(2);
    space[0] = result.real();
    space[1] = result.imag();
}

bool THSTorch_scalar_to_bool(Scalar value)
{
    return value->toBool();
}

int8_t THSTorch_scalar_type(Scalar value)
{
    return (int8_t)value->type();
}

void THSTorch_dispose_scalar(Scalar scalar)
{
    delete scalar;
}

double THSSpecial_erf_scalar(const double x)
{
    return erf(x);
}

double THSSpecial_erfc_scalar(const double x)
{
    return erfc(x);
}