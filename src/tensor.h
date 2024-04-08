/* Copyright 2020, GoTorch Authors */
#pragma once
#if defined(__GNUC__)
// Attributes to prevent 'unused' function from being removed and to make it visible
#define FUNCTION_ATTRIBUTE __attribute__((visibility("default"))) __attribute__((used))
#elif defined(_MSC_VER)
// Marking a function for export
#define FUNCTION_ATTRIBUTE __declspec(dllexport)
#endif
#include "torch.h"

#ifdef __cplusplus
extern "C" {
#endif

////////////////////////////////////////////////////////////////////////////////
// Tensor construction and operations, torch
////////////////////////////////////////////////////////////////////////////////

FUNCTION_ATTRIBUTE const char *Tensor_Detach(Tensor a, Tensor *result);
const char *Tensor_String(Tensor a);
FUNCTION_ATTRIBUTE const char *Tensor_Print(Tensor a);
void Tensor_Close(Tensor a);
void FreeString(const char *s);
const char *Tensor_Save(Tensor tensor, const char *path);
const char *Tensor_Load(const char *path, Tensor *result);
FUNCTION_ATTRIBUTE const char *Tensor_Dim(Tensor tensor, int64_t *dim);
FUNCTION_ATTRIBUTE const char *Tensor_Shape(Tensor tensor, int64_t *dims);
FUNCTION_ATTRIBUTE const char *Tensor_Dtype(Tensor tensor, int8_t *dtype);
const char *Tensor_SetData(Tensor self, Tensor new_data);
FUNCTION_ATTRIBUTE const char *Tensor_FromBlob(void *data, int8_t dtype, int64_t *sizes_data,
                            int64_t sizes_data_len, Tensor *result);
const char *Tensor_To(Tensor input, Device device, int8_t dtype,
                      Tensor *output);
const char *Tensor_CastTo(Tensor input, int8_t dtype, Tensor *output);
const char *Tensor_CopyTo(Tensor input, Device device, Tensor *output);
const char *Tensor_PinMemory(Tensor input, Tensor *output);
const char *Tensor_CUDA(Tensor input, Device device, int8_t non_blocking,
                        Tensor *output);

////////////////////////////////////////////////////////////////////////////////
// Backward, Gradient
////////////////////////////////////////////////////////////////////////////////

void Tensor_Backward(Tensor a);
Tensor Tensor_Grad(Tensor a);

////////////////////////////////////////////////////////////////////////////////
// Get elements
////////////////////////////////////////////////////////////////////////////////

const char *Item(Tensor a, float *result);
const char *ItemInt64(Tensor a, int64_t *result);
const char *ItemFloat64(Tensor a, double *result);

FUNCTION_ATTRIBUTE const char *Tensor_Index(Tensor a, int64_t *index, int64_t index_len,
                         Tensor *result);

#ifdef __cplusplus
}
#endif
