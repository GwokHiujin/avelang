#ifndef AVE_LANG_DIALECT_AVE_LANG_TRANSFORMS_HOIST_ALLOCA_PASS_H
#define AVE_LANG_DIALECT_AVE_LANG_TRANSFORMS_HOIST_ALLOCA_PASS_H

#include <memory>
#include <mlir/Pass/Pass.h>

namespace mlir {
class Pass;
} // namespace mlir

namespace causalflow::avelang::dialect {

std::unique_ptr<mlir::Pass> createHoistAllocaPass();

} // namespace causalflow::avelang::dialect

#endif // AVE_LANG_DIALECT_AVE_LANG_TRANSFORMS_HOIST_ALLOCA_PASS_H
