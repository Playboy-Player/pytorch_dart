/* Copyright 2020, GoTorch Authors */

#pragma once
#if defined(__GNUC__)
// Attributes to prevent 'unused' function from being removed and to make it visible
#define FUNCTION_ATTRIBUTE __attribute__((visibility("default"))) __attribute__((used))
#elif defined(_MSC_VER)
// Marking a function for export
#define FUNCTION_ATTRIBUTE __declspec(dllexport)
#endif
#include "torchdef.h"
#ifdef __cplusplus
extern "C" {
#endif

////////////////////////////////////////////////////////////////////////////////
// Tensor construction and operations, torch
////////////////////////////////////////////////////////////////////////////////

// torch.randn
const char *RandN(int64_t *size, int64_t length, int64_t require_grad,
                  Tensor *result);
// torch.rand
const char *Rand(int64_t *size, int64_t length, int64_t require_grad,
                 Tensor *result);
// torch.empty
FUNCTION_ATTRIBUTE const char *Empty(int64_t *size, int64_t length, int64_t require_grad,
                  Tensor *result);
// torch.ones
FUNCTION_ATTRIBUTE const char *Ones(int64_t *size, int64_t length, int64_t require_grad,
                 Tensor *result);
// torch.eye
FUNCTION_ATTRIBUTE const char *Eye(int64_t n, int64_t m, int64_t require_grad, Tensor *result);
// torch.full, only for float32
FUNCTION_ATTRIBUTE const char *Full(int64_t *size, int64_t length, float value,
                 int64_t require_grad, Tensor *result);
// torch.arange
FUNCTION_ATTRIBUTE const char *Arange(float start, float end, float step, int64_t require_grad,
                   Tensor *result);
// torch.linspace
FUNCTION_ATTRIBUTE const char *Linspace(float start, float end, int64_t steps,
                     int64_t require_grad, Tensor *result);
// torch.logspace
FUNCTION_ATTRIBUTE const char *Logspace(float start, float end, int64_t steps, double base,
                     int64_t require_grad, Tensor *result);

FUNCTION_ATTRIBUTE const char *Equal(Tensor a, Tensor b, int64_t *result);

FUNCTION_ATTRIBUTE const char *MM(Tensor a, Tensor b, Tensor *result);
FUNCTION_ATTRIBUTE const char *Sum(Tensor a, Tensor *result);
FUNCTION_ATTRIBUTE const char *SumByDim(Tensor a, int64_t dim, int8_t keepDim, Tensor *result);
const char *Relu(Tensor a, Tensor *result);
const char *LeakyRelu(Tensor a, double negative_slope, Tensor *result);
const char *Tanh(Tensor a, Tensor *result);
const char *Sigmoid(Tensor a, Tensor *result);
FUNCTION_ATTRIBUTE const char *Add(Tensor a, Tensor other, float alpha, Tensor *result);
FUNCTION_ATTRIBUTE const char *Add_(Tensor a, Tensor other, float alpha, Tensor *result);
FUNCTION_ATTRIBUTE const char *Sub(Tensor a, Tensor other, float alpha, Tensor *result);
FUNCTION_ATTRIBUTE const char *Sub_(Tensor a, Tensor other, float alpha, Tensor *result);
FUNCTION_ATTRIBUTE const char *Mul(Tensor a, Tensor other, Tensor *result);
FUNCTION_ATTRIBUTE const char *Mul_(Tensor a, Tensor other, Tensor *result);
FUNCTION_ATTRIBUTE const char *Div(Tensor a, Tensor other, Tensor *result);
FUNCTION_ATTRIBUTE const char *Div_(Tensor a, Tensor other, Tensor *result);
const char *Permute(Tensor a, int64_t *dims, int64_t dims_size, Tensor *result);
const char *AllClose(Tensor a, Tensor b, int64_t *result);
const char *Flatten(Tensor a, int64_t startDim, int64_t endDim, Tensor *result);
const char *TopK(Tensor a, int64_t k, int64_t dim, int8_t largest,
                 int8_t sorted, Tensor *values, Tensor *indices);
const char *Transpose(Tensor a, int64_t dim0, int64_t dim1, Tensor *result);
const char *ExpandAs(Tensor a, Tensor other, Tensor *result);
const char *Eq(Tensor a, Tensor other, Tensor *result);
const char *IndexSelect(Tensor a, int64_t dim, Tensor index, Tensor *result);
const char *View(Tensor a, Tensor *result, int64_t *size, int64_t size_len);
const char *LogSoftmax(Tensor a, int64_t dim, Tensor *result);
const char *Squeeze(Tensor a, Tensor *result);
const char *SqueezeWithDim(Tensor a, int64_t dim, Tensor *result);
const char *Argmin(Tensor a, int64_t *dim, int8_t keepdim, Tensor *result);
const char *Argmax(Tensor a, int64_t *dim, int8_t keepdim, Tensor *result);

const char *Mean(Tensor a, Tensor *result);
const char *Stack(Tensor *tensors, int64_t tensors_size, int64_t dim,
                  Tensor *result);
#ifdef __cplusplus
}
#endif
