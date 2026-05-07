#pragma once

#include <mlir/Pass/Pass.h>

namespace causalflow::avelang::dialect {

/// Create a pass that lowers AveLang GPU operations to intrinsic function calls
std::unique_ptr<mlir::Pass> createLowerAveLangGPUToIntrinsicsPass();

} // namespace causalflow::avelang::dialect