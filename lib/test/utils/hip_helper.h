#pragma once

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

#define CheckHipError(result)                                                  \
    if ((result) != hipSuccess) {                                              \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,        \
                hipGetErrorString(result));                                    \
        abort();                                                               \
    }