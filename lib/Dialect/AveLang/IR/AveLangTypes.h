//===- AveLangTypes.h - AveLang dialect type declarations --------------===//
//
// This file declares the types of the AveLang dialect.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"

// Include the generated type declarations
#define GET_TYPEDEF_CLASSES
#include "AveLangTypes.h.inc"

namespace causalflow::avelang::dialect {

// Forward declarations for generated types are included above

} // namespace causalflow::avelang::dialect
