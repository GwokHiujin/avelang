#pragma once

#include <mlir/Pass/Pass.h>

namespace causalflow::avelang::dialect {

/// Rewrite ave.return into a single-exit control-flow form before
/// lowering structured control flow.
std::unique_ptr<mlir::Pass> createNormalizeAveLangReturnPass();

} // namespace causalflow::avelang::dialect
