#pragma once

#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

// Include the generated dialect declarations
#include "AveLangDialect.h.inc"

namespace causalflow::avelang::dialect {

// We'll use the TableGen generated dialect class

} // namespace causalflow::avelang::dialect