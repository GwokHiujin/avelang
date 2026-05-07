#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda.h>

#define CheckCuError(result)                                                   \
    do {                                                                       \
        if ((result) != CUDA_SUCCESS) {                                        \
            const char *error_string = nullptr;                                \
            cuGetErrorString(result, &error_string);                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    error_string);                                             \
            abort();                                                           \
        }                                                                      \
    } while (false)
