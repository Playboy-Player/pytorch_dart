// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
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

#include "torch/script.h"

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

// Copied from libtorch to share the type as an int8_t.
enum TypeKind : int8_t {
#define DEFINE_TYPE(T) T,
    C10_FORALL_TYPES(DEFINE_TYPE)
#undef DEFINE_TYPE
};

// API.

struct TensorOrScalar
{
    int64_t TypeCode;
    int64_t ArrayIndex;
    ptrdiff_t Handle;
};


EXPORT_API_ANDROID_OR_IOS(JITModule) THSJIT_load(const char* filename, int64_t device, int64_t index);
EXPORT_API(JITModule) THSJIT_load_byte_array(char* bytes, int64_t size, int64_t device, int64_t index);

EXPORT_API(void) THSJIT_save(JITModule module, const char* filename);
EXPORT_API(void) THSJIT_save_byte_array(JITModule module, char* bytes, int64_t size);

EXPORT_API(JITCompilationUnit) THSJIT_compile(const char* script);

EXPORT_API(void) THSJIT_Module_dispose(const JITModule module);
EXPORT_API(void) THSJIT_CompilationUnit_dispose(const JITCompilationUnit module);

EXPORT_API(int) THSJIT_Module_num_inputs(const JITModule method);
EXPORT_API(int) THSJIT_Module_num_outputs(const JITModule method);

EXPORT_API_ANDROID_OR_IOS(void) THSJIT_Module_forward(const JITModule module, const TensorOrScalar* tensorPtrs, const int length, TensorOrScalar* (*allocator)(int32_t idx, size_t length), int8_t* typeCode, int32_t idx);
EXPORT_API(void) THSJIT_Module_invoke(const JITModule module, const char* name, const TensorOrScalar* tensorPtrs, const int length, TensorOrScalar* (*allocator)(int32_t idx, size_t length), int8_t* typeCode, int32_t idx);

EXPORT_API(void) THSJIT_CompilationUnit_Invoke(const JITCompilationUnit module, const char* method, const TensorOrScalar* tensorPtrs, const int length, TensorOrScalar* (*allocator)(int32_t idx, size_t length), int8_t* typeCode, int32_t idx);

EXPORT_API(int) THSJIT_Module_is_training(JITModule module);
EXPORT_API(void) THSJIT_Module_train(JITModule module, bool on);
EXPORT_API(void) THSJIT_Module_eval(JITModule module);

EXPORT_API(void) THSJIT_Module_to_device_dtype(JITModule module, int8_t dtype, int64_t device, int64_t index);
EXPORT_API(void) THSJIT_Module_to_device(JITModule module, int64_t device, int64_t index);
EXPORT_API(void) THSJIT_Module_to_dtype(JITModule module, int8_t dtype);

EXPORT_API(JITType) THSJIT_Module_getInputType(JITModule module, int8_t dtype);

EXPORT_API(int8_t) THSJIT_Type_kind(JITType handle);
EXPORT_API(void*) THSJIT_Type_cast(const JITType type);

EXPORT_API(int8_t) THSJIT_TensorType_dtype(const JITTensorType type);
EXPORT_API(void) THSJIT_TensorType_sizes(const JITTensorType type, int64_t* (*allocator)(int64_t length));

EXPORT_API(void) THSJIT_Type_dispose(const JITType type);
EXPORT_API(void) THSJIT_TensorType_dispose(const JITTensorType type);

EXPORT_API(void) THSJIT_Module_modules(const JITModule module, JITModule* (*allocator)(size_t length));
EXPORT_API(void) THSJIT_Module_named_modules(const JITModule module,
    JITModule* (*allocator)(size_t length),
    const char** (*allocator2)(size_t length));

EXPORT_API(void) THSJIT_Module_named_children(const JITModule module,
    JITModule* (*allocator)(size_t length),
    const char** (*allocator2)(size_t length));

EXPORT_API(JITMethod) THSJIT_Module_get_method(const JITModule module, const char* name);

EXPORT_API(void) THSJIT_Module_parameters(const JITModule module, Tensor* (*allocator)(size_t length));
EXPORT_API(void) THSJIT_Module_named_parameters(const JITModule module,
    Tensor* (*allocator)(size_t length),
    const char** (*allocator2)(size_t length));

EXPORT_API(void) THSJIT_Module_named_buffers(const JITModule module,
    Tensor* (*allocator)(size_t length),
    const char** (*allocator2)(size_t length));

EXPORT_API(void) THSJIT_Module_named_attributes(const JITModule module, bool recurse,
    Tensor* (*allocator)(size_t length),
    const char** (*allocator2)(size_t length));

EXPORT_API(void) THSJIT_Module_set_attribute(const JITModule module, const char* name, Tensor tensor);

EXPORT_API(int) THSJIT_Method_num_inputs(const JITMethod method);

EXPORT_API(void) THSJIT_Method_dispose(const JITMethod method);

EXPORT_API(const char*) THSJIT_Method_name(const JITMethod method);

EXPORT_API_ANDROID_OR_IOS(TensorOrScalar*) THSJIT_AllocateTensorOrScalarArray(int32_t size);
EXPORT_API_ANDROID_OR_IOS(void) THSJIT_FreeTensorOrScalarArray(TensorOrScalar* ptr);
EXPORT_API_ANDROID_OR_IOS(void) THSJIT_SetTensorOrScalar(TensorOrScalar* array, int32_t index, int64_t type_code, int64_t array_index, ptrdiff_t handle);
EXPORT_API_ANDROID_OR_IOS(TensorOrScalar*) THSJIT_GetTensorOrScalar(TensorOrScalar* array, int32_t index);

//// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.


JITModule THSJIT_load(const char* filename, int64_t device, int64_t index)
{
    c10::DeviceType dev = c10::kCPU;
    if (device == 1)
        dev = c10::kCUDA;
    if (device == 13)
        dev = c10::kMPS;

    CATCH(
        auto res = torch::jit::load(filename, torch::Device(dev, index));
        auto copy = new torch::jit::Module(res);
        return new std::shared_ptr<torch::jit::Module>(copy);
    );

    return nullptr;
}

JITModule THSJIT_load_byte_array(char* bytes, int64_t size, int64_t device, int64_t index)
{
    c10::DeviceType dev = c10::kCPU;
    if (device == 1)
        dev = c10::kCUDA;
    if (device == 13)
        dev = c10::kMPS;

    CATCH(
        std::istringstream stream(std::string(bytes, size));

        auto res = torch::jit::load(stream, torch::Device(dev, index));
        auto copy = new torch::jit::Module(res);
        return new std::shared_ptr<torch::jit::Module>(copy);
    );

    return nullptr;
}
#ifdef __ANDROID__
JITCompilationUnit THSJIT_compile(const char* script)
{
    CATCH(
       
    );

    return nullptr;
}

void THSJIT_save(JITModule module, const char* filename)
{
    CATCH(
        
    );
}

void THSJIT_save_byte_array(JITModule module, char* bytes, int64_t size)
{
    CATCH(
       
    );
}

#else
JITCompilationUnit THSJIT_compile(const char* script)
{
    CATCH(
        auto res = torch::jit::compile(script);
        return new std::shared_ptr<torch::jit::CompilationUnit>(res);
    );

    return nullptr;
}

void THSJIT_save(JITModule module, const char* filename)
{
    CATCH(
        (*module)->save(filename);
    );
}

void THSJIT_save_byte_array(JITModule module, char* bytes, int64_t size)
{
    CATCH(
        std::ostringstream stream(std::string(bytes, size));

        (*module)->save(stream);
    );
}
#endif
int THSJIT_Module_is_training(JITModule module)
{
    return (*module)->is_training();
}

void THSJIT_Module_train(JITModule module, bool on)
{
    (*module)->train(on);
}

void THSJIT_Module_eval(JITModule module)
{
    (*module)->eval();
}

void THSJIT_Module_to_device_dtype(JITModule module, int8_t dtype, int64_t device, int64_t index)
{
    c10::DeviceType dev = c10::kCPU;
    if (device == 1)
        dev = c10::kCUDA;
    if (device == 13)
        dev = c10::kMPS;

    (*module)->to(torch::Device(dev, index));
}

void THSJIT_Module_to_device(JITModule module, int64_t device, int64_t index)
{
    c10::Device dev = (device == 1) ? torch::Device(c10::kCUDA, index) : torch::Device(c10::kCPU);

    (*module)->to(dev);
}

void THSJIT_Module_to_dtype(JITModule module, int8_t dtype)
{
    (*module)->to((at::ScalarType)dtype);
}

void THSJIT_Module_modules(const JITModule module, JITModule* (*allocator)(size_t length))
{
    auto modules = (*module)->modules();
    JITModule* result = allocator(modules.size());
    int i = 0;
    for (const auto& child : modules) {
        auto copy = new torch::jit::Module(child);
        result[i++] = new std::shared_ptr<torch::jit::Module>(copy);
    }
}

void THSJIT_Module_named_modules(const JITModule module,
    JITModule* (*allocator)(size_t length),
    const char** (*allocator2)(size_t length))
{
    auto modules = (*module)->named_modules();
    JITModule* result = allocator(modules.size());
    const char** names = allocator2(modules.size());
    int i = 0;
    for (const auto& child : modules) {
        auto copy = new torch::jit::Module(child.value);
        result[i] = new std::shared_ptr<torch::jit::Module>(copy);
        names[i] = make_sharable_string(child.name);
        i++;
    }
}

void THSJIT_Module_named_children(const JITModule module,
    JITModule* (*allocator)(size_t length),
    const char** (*allocator2)(size_t length))
{
    auto modules = (*module)->named_children();
    JITModule* result = allocator(modules.size());
    const char** names = allocator2(modules.size());
    int i = 0;
    for (const auto& child : modules) {
        auto copy = new torch::jit::Module(child.value);
        result[i] = new std::shared_ptr<torch::jit::Module>(copy);
        names[i] = make_sharable_string(child.name);
        i++;
    }
}

void THSJIT_Module_parameters(const JITModule module, Tensor* (*allocator)(size_t length))
{
    auto parameters = (*module)->parameters();
    Tensor* result = allocator(parameters.size());
    int i = 0;
    for (const auto& child : parameters) {
        result[i++] = new torch::Tensor(child);
    }
}

void THSJIT_Module_named_parameters(const JITModule module,
    Tensor* (*allocator)(size_t length),
    const char** (*allocator2)(size_t length))
{
    auto parameters = (*module)->named_parameters();
    Tensor* result = allocator(parameters.size());
    const char** names = allocator2(parameters.size());
    int i = 0;
    for (const auto& child : parameters) {
        result[i] = new torch::Tensor(child.value);
        names[i] = make_sharable_string(child.name);
        i++;
    }
}

void THSJIT_Module_named_buffers(const JITModule module,
    Tensor* (*allocator)(size_t length),
    const char** (*allocator2)(size_t length))
{
    auto parameters = (*module)->named_buffers();
    Tensor* result = allocator(parameters.size());
    const char** names = allocator2(parameters.size());
    int i = 0;
    for (const auto& child : parameters) {
        result[i] = new torch::Tensor(child.value);
        names[i] = make_sharable_string(child.name);
        i++;
    }
}

void THSJIT_Module_named_attributes(const JITModule module, bool recurse,
    Tensor* (*allocator)(size_t length),
    const char** (*allocator2)(size_t length))
{
    auto attributes = (*module)->named_attributes(recurse);
    Tensor* result = allocator(attributes.size());
    const char** names = allocator2(attributes.size());
    int i = 0;
    for (const auto& child : attributes) {
        if (!child.name.empty() && child.value.isTensor())
        {
            auto& t = child.value.toTensor();
            result[i] = new torch::Tensor(child.value.toTensor());
            names[i] = make_sharable_string(child.name);
            i++;
        }
    }
}

void THSJIT_Module_set_attribute(const JITModule module, const char *name, Tensor tensor)
{
    CATCH((*module)->setattr(name, *tensor););
}

JITMethod THSJIT_Module_get_method(const JITModule module, const char* name)
{
    auto method = (*module)->get_method(name);
    auto copy = new torch::jit::Method(method);
    return new std::shared_ptr<torch::jit::Method>(copy);
}

TensorOrScalar* validate(TensorOrScalar* ptr)
{
    if (ptr == nullptr)
        torch_last_err = strdup("null returned from TorchSharp return value allocator.");
    return ptr;
}
TensorOrScalar* ReturnHelper(c10::IValue result, TensorOrScalar* (*allocator)(int32_t idx, size_t length), int8_t* typeCode, int32_t* idx)
{
    // TypeCode:
    //
    // 0 -- Not supported
    // 1 -- Single tensor
    // 2 -- Tuple of tensors
    // 3 -- List of tensors
    // 4 -- Single scalar
    // 5 -- Scalar tuple
    // 6 -- List of scalars
    // 7 -- List of scalars and tensors
    // 8 -- None / null
    // 9 -- List of anything
    // 10 -- Tuple of anything

    if (result.isNone())
    {
        TensorOrScalar* output = validate(allocator(idx[0]++, 1));
        if (output == nullptr) return output;
        output[0] = { 8, -1, (ptrdiff_t)0 };
        *typeCode = 8;
        return output;
    }

    if (result.isScalar())
    {
        TensorOrScalar* output = validate(allocator(idx[0]++, 1));
        if (output == nullptr) return output;
        output[0] = { 0, -1, (ptrdiff_t)new torch::Scalar(result.toScalar()) };
        *typeCode = 4;
        return output;
    }

    if (result.isTensor()) {
        TensorOrScalar* output = validate(allocator(idx[0]++, 1));
        if (output == nullptr) return output;
        output[0] = { 0, -1, (ptrdiff_t)ResultTensor(result.toTensor()) };
        *typeCode = 1;
        return output;
    }

    if (result.isTensorList()) {
        auto list = result.toTensorList();
        *typeCode = 3;
        TensorOrScalar* output = validate(allocator(idx[0]++, list.size()));
        if (output == nullptr) return output;
        for (size_t i = 0; i < list.size(); i++)
            output[i] = { 0, -1, (ptrdiff_t)ResultTensor(list[i]) };
        return output;
    }

    if (result.isList())
    {
        int foundTensor = 0;
        int foundScalar = 0;
        int foundNull = 0;
        int foundListOrTuple = 0;

        auto list = result.toList();
        TensorOrScalar* output = validate(allocator(idx[0]++, list.size()));
        if (output == nullptr) return output;

        for (int i = 0; i < list.size(); ++i)
        {
            output[i].Handle = -1;
            c10::IValue value = list[i];

            if (value.isTensor())
            {
                output[i] = { 0, -1, (ptrdiff_t)ResultTensor(value.toTensor()) };
                foundTensor += 1;
                continue;
            }
            if (value.isScalar())
            {
                output[i] = { 4, -1, (ptrdiff_t)new torch::Scalar(value.toScalar()) };
                foundScalar += 1;
                continue;
            }
            if (value.isNone())
            {
                output[i] = { 8, -1, (ptrdiff_t)0 };
                foundNull += 1;
                continue;
            }
            else {
                int8_t nestedTC = 0;
                int64_t arrIdx = idx[0];
                auto nested = ReturnHelper(value, allocator, &nestedTC, idx);
                if (nested == nullptr) return nested;
                foundListOrTuple += 1;
                output[i] = { nestedTC, arrIdx, (ptrdiff_t)nested };
            }
        }

        if (foundListOrTuple > 0) {
            *typeCode = 9;
        }
        else {
            *typeCode = 7;
            if (foundScalar == 0 && foundNull == 0)
                *typeCode = 3;
            if (foundTensor == 0 && foundNull == 0)
                *typeCode = 6;
        }
        return output;
    }

    if (result.isTuple()) {
        int foundTensor = 0;
        int foundScalar = 0;
        int foundNull = 0;
        int foundListOrTuple = 0;

        auto& list = result.toTuple()->elements();
        TensorOrScalar* output = validate(allocator(idx[0]++, list.size()));
        if (output == nullptr) return output;

        for (int i = 0; i < list.size(); ++i)
        {
            output[i].Handle = -1;
            c10::IValue value = list[i];

            if (value.isTensor())
            {
                output[i] = { 0, -1, (ptrdiff_t)ResultTensor(value.toTensor()) };
                foundTensor += 1;
                continue;
            }
            if (value.isScalar())
            {
                output[i] = { 4, -1, (ptrdiff_t)new torch::Scalar(value.toScalar()) };
                foundScalar += 1;
                continue;
            }
            if (value.isNone())
            {
                output[i] = { 8, -1, (ptrdiff_t)0 };
                foundNull += 1;
                continue;
            }
            else {
                int8_t nestedTC = 0;
                int64_t arrIdx = idx[0];
                auto nested = ReturnHelper(value, allocator, &nestedTC, idx);
                if (nested == nullptr) return nested;
                foundListOrTuple += 1;
                output[i] = { nestedTC, arrIdx, (ptrdiff_t)nested };
            }
        }

        *typeCode = 10;
        if (foundListOrTuple == 0) {
            if (foundScalar == 0 && foundNull == 0)
                *typeCode = 2;
            if (foundTensor == 0 && foundNull == 0)
                *typeCode = 5;
        }

        return output;
    }

    *typeCode = 0;
    return nullptr;
}

c10::impl::GenericList toScalarValueList(const TensorOrScalar* tensorPtrs, const int length)
{
    auto list = c10::impl::GenericList(c10::ScalarTypeType::get());

    if (tensorPtrs != nullptr) {
        for (int i = 0; i < length; i++)
        {
            switch (tensorPtrs[i].TypeCode) {
            case 1:
                list.push_back(*(torch::Scalar*)(tensorPtrs[i].Handle));
                break;
            }
        }
    }

    return list;
}

c10::impl::GenericList toTensorValueList(const TensorOrScalar* tensorPtrs, const int length)
{
    auto list = c10::impl::GenericList(c10::TensorType::get());

    if (tensorPtrs != nullptr) {
        for (int i = 0; i < length; i++)
        {
            switch (tensorPtrs[i].TypeCode) {
            case 0:
                list.push_back(*(torch::Tensor*)(tensorPtrs[i].Handle));
                break;
            }
        }
    }

    return list;
}

std::vector<c10::IValue> toIValue(const TensorOrScalar* tensorPtrs, const int length)
{
    // TypeCode:
    //
    // 0 -- Single tensor
    // 1 -- Single scalar
    // 2 -- Boolean
    // 3 -- Int32
    // 5 -- List of tensors
    // 6 -- List of scalars
    // 8 -- None / null

    std::vector<c10::IValue> tensors;

    if (tensorPtrs != nullptr) {
        for (int i = 0; i < length; i++)
        {
            switch (tensorPtrs[i].TypeCode) {
            case 0:
                tensors.push_back(*(torch::Tensor*)(tensorPtrs[i].Handle));
                break;
            case 1:
                tensors.push_back(*(torch::Scalar*)(tensorPtrs[i].Handle));
                break;
            case 2:
                tensors.push_back(tensorPtrs[i].Handle != 0);
                break;
            case 3:
                tensors.push_back((int)tensorPtrs[i].Handle);
                break;
            case 5:
            {
                auto ts = toTensorValueList(reinterpret_cast<const TensorOrScalar*>(tensorPtrs[i].Handle), (int)tensorPtrs[i].ArrayIndex);
                tensors.push_back(ts);
                break;
            }
            case 6:
            {
                auto ts = toScalarValueList(reinterpret_cast<const TensorOrScalar*>(tensorPtrs[i].Handle), (int)tensorPtrs[i].ArrayIndex);
                tensors.push_back(ts);
                break;
            }
            //case 4:
            //    tensors.push_back(c10::IValue(tensorPtrs[i].Handle)); // Clang on MacOS doesn't like. Pass as Scalar from .NET.
            //    break;
            case 8:
                tensors.push_back(c10::nullopt);
                break;
            }
        }
    }
    return tensors;
}

void THSJIT_Module_forward(const JITModule module, const TensorOrScalar* tensorPtrs, const int length, TensorOrScalar* (*allocator)(int32_t idx, size_t length), int8_t* typeCode, int32_t idx)
{
    *typeCode = 0;

    CATCH(
        auto result = (*module)->forward(toIValue(tensorPtrs, length));
        ReturnHelper(result, allocator, typeCode, &idx);
    )
}

void THSJIT_Module_invoke(const JITModule module, const char* name, const TensorOrScalar* tensorPtrs, const int length, TensorOrScalar* (*allocator)(int32_t idx, size_t length), int8_t* typeCode, int32_t idx)
{
    *typeCode = 0;

    CATCH(
        auto method = (*module)->get_method(name);
        auto result = method(toIValue(tensorPtrs, length));
        ReturnHelper(result, allocator, typeCode, &idx);
    )
}

void THSJIT_CompilationUnit_Invoke(const JITCompilationUnit module, const char* method, const TensorOrScalar* tensorPtrs, const int length, TensorOrScalar* (*allocator)(int32_t idx, size_t length), int8_t* typeCode, int32_t idx)
{
    *typeCode = 0;

    CATCH(
        auto args = toIValue(tensorPtrs, length);
        auto func = (*module)->find_function(method);
        auto result = (*func)(args);
        ReturnHelper(result, allocator, typeCode, &idx);
    )
}

void THSJIT_Module_dispose(const JITModule module)
{
    delete module;
}

const char* THSJIT_Method_name(const JITMethod method)
{
    return make_sharable_string((*method)->name());
}

int THSJIT_Method_num_inputs(const JITMethod method)
{
    return (int)(*method)->num_inputs();
}

int THSJIT_Module_num_inputs(const JITModule module)
{
    return (int)(*module)->get_method("forward").num_inputs() - 1; // Don't count the 'self' argument.
}

int THSJIT_Module_num_outputs(const JITModule module)
{
    return (int)(*module)->get_method("forward").function().getSchema().returns().size();
}

JITFunction THSJIT_Method_function(const JITMethod method)
{
    return new std::shared_ptr<torch::jit::Function>(&(*method)->function());
}

void THSJIT_Method_dispose(const JITMethod method)
{
    delete method;
}


//-------------------------------------------------------------------------------------
// JITFunction

int THSJIT_Function_num_inputs(const JITFunction function)
{
    return (int)(*function)->num_inputs();
}

// TODO other function operations

void THSJIT_Function_dispose(const JITFunction function)
{
    delete function;
}

void THSJIT_Type_dispose(const JITType type)
{
    delete type;
}

void THSJIT_TensorType_dispose(const JITTensorType type)
{
    delete type;
}

void THSJIT_CompilationUnit_dispose(const JITCompilationUnit module)
{
    delete module;
}

void* THSJIT_Type_cast(const JITType type)
{
    switch ((*type)->kind())
    {
    case c10::TypeKind::TensorType:
        return new std::shared_ptr<torch::jit::TensorType>((*type)->cast<c10::TensorType>());
    default:
        return nullptr;
    }
}

int8_t THSJIT_TensorType_dtype(const JITTensorType type)
{
    auto scT = (*type)->scalarType();
    if (scT.has_value()) {
        return (int8_t)scT.value();
    }
    else {
        return -1;
    }
}

void THSJIT_TensorType_sizes(const JITTensorType type, int64_t* (*allocator)(int64_t length))
{
    //CATCH(
    auto& t = *type;
    auto dim = t->dim();
    auto res = (*type)->sizes().concrete_sizes();
    if (res.has_value()) {
        const size_t sz = res.value().size();
        auto& vec = res.value();
        int64_t* result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = vec[i];
    }
    //);
}

int8_t THSJIT_Type_kind(const JITType type)
{
    switch ((*type)->kind())
    {
    case c10::TypeKind::TensorType:
        return (int8_t)TypeKind::TensorType;
    default:
        return -1;
    }
}

JITType THSJIT_Module_getInputType(JITModule module, int8_t index)
{
    auto typ = (*module)->type();
    c10::TypeKind kind = typ->kind();
    auto& schema = typ->getMethod("forward").getSchema();
    return new std::shared_ptr<c10::Type>(schema.arguments()[1 + index].type()->cast<c10::TensorType>());
}

void THSJIT_typeDispose(const JITType type)
{
    delete type;
}


TensorOrScalar* THSJIT_AllocateTensorOrScalarArray(int32_t size)
{
    auto result = new TensorOrScalar[size];
    memset(result, 0, size * sizeof(TensorOrScalar));
    return result;
}

void THSJIT_FreeTensorOrScalarArray(TensorOrScalar* ptr)
{
    delete ptr;
}

void THSJIT_SetTensorOrScalar(TensorOrScalar* array, int32_t index, int64_t type_code, int64_t array_index, ptrdiff_t handle)
{
    array[index].TypeCode = type_code;
    array[index].ArrayIndex = array_index;
    array[index].Handle = handle;
}

TensorOrScalar* THSJIT_GetTensorOrScalar(TensorOrScalar* array, int32_t index)
{
    return array + index;
}