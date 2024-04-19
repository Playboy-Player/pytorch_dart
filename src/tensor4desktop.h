//Some functions only available for desktop platforms
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

FUNCTION_ATTRIBUTE const char *Tensor_Save(Tensor tensor, const char *path);
FUNCTION_ATTRIBUTE const char *Tensor_Load(const char *path, Tensor *result);

#ifdef __cplusplus
}
#endif
