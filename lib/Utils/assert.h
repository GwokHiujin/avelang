#pragma once

#include <cassert>

#define SS_ASSERT(cond)                                                        \
    do {                                                                       \
        if (!(cond)) {                                                         \
            printf("Assertion '%s' failed at %s:%d\n", #cond, __FILE__,        \
                   __LINE__);                                                  \
            abort();                                                           \
        }                                                                      \
    } while (false)
