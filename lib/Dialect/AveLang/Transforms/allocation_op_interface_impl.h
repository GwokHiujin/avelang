#pragma once

#include <mlir/IR/DialectRegistry.h>

namespace causalflow::avelang::dialect {

void registerAllocationOpInterfaceExternalModels(
    mlir::DialectRegistry &registry);

} // namespace causalflow::avelang::dialect
