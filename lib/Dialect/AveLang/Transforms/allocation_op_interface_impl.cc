#include "allocation_op_interface_impl.h"

#include "Dialect/AveLang/IR/AveLangDialect.h"
#include "Dialect/AveLang/IR/AveLangOps.h"

#include <mlir/Dialect/Bufferization/IR/AllocationOpInterface.h>
#include <mlir/IR/Dialect.h>

namespace causalflow::avelang::dialect {
namespace {

// MLIR's memref.alloca only advertises HoistingKind::Loop as it argues that the
// stack variables have scoped lifetime. In the case of GPU kernels, it is more
// beneficial to hoist the alloca all the way to the entry blocks to enable
// LLVM's SROA to turn them into SSA values, eliminating stack allocations
// altogether.
struct AveLangAllocaAllocationInterface
    : public mlir::bufferization::AllocationOpInterface::ExternalModel<
          AveLangAllocaAllocationInterface, AveLangMemRefAllocaOp> {
    static ::mlir::HoistingKind getHoistingKind() {
        return mlir::HoistingKind::Loop | mlir::HoistingKind::Block;
    }
};

} // namespace

void registerAllocationOpInterfaceExternalModels(
    mlir::DialectRegistry &registry) {
    registry.addExtension(+[](mlir::MLIRContext *ctx, AveLangDialect *dialect) {
        (void)dialect;
        AveLangMemRefAllocaOp::attachInterface<
            AveLangAllocaAllocationInterface>(*ctx);
    });
}

} // namespace causalflow::avelang::dialect
