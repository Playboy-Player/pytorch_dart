// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#include "Utils.h"

#include <cstring>
#include <fstream>
#if _WINDOWS
#include <combaseapi.h>
#define TP_CoTaskMemAlloc(t) CoTaskMemAlloc(t)
#else
#define TP_CoTaskMemAlloc(t) malloc(t)
#endif

thread_local char * torch_last_err = nullptr;

const char * make_sharable_string(const std::string str)
{
    size_t n = str.length();
    char* result = (char *)TP_CoTaskMemAlloc(n + 1); 
    strncpy(result, str.c_str(), n);
    result[n] = '\0';
    return result;
}
