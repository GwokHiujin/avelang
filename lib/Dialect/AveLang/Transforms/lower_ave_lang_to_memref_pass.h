#ifndef AVE_LANG_DIALECT_AVE_LANG_TRANSFORMS_LOWER_AVE_LANG_TO_MEMREF_PASS_H
#define AVE_LANG_DIALECT_AVE_LANG_TRANSFORMS_LOWER_AVE_LANG_TO_MEMREF_PASS_H

#include <memory>
#include <mlir/Pass/Pass.h>

namespace mlir {
class Pass;
class ModuleOp;
} // namespace mlir

namespace causalflow::avelang::dialect {

/// Create a pass that lowers AveLang dialect operations to memref operations
/// with semi-affine layout maps. This pass lowers ave.memref.cast
/// operations into memref.view / memref.reinterpret_cast sequences with
/// appropriate strided layouts and semi-affine map attributes. It also adds
/// layout_map attributes to existing memref.reinterpret_cast operations for
/// analysis.
std::unique_ptr<mlir::Pass> createLowerAveLangToMemRefPass();

} // namespace causalflow::avelang::dialect

#endif // AVE_LANG_DIALECT_AVE_LANG_TRANSFORMS_LOWER_AVE_LANG_TO_MEMREF_PASS_H
