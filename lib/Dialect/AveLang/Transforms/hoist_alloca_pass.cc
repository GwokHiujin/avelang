#include "hoist_alloca_pass.h"

#include "Dialect/AveLang/IR/AveLangOps.h"
#include "Dialect/AveLang/IR/AveLangTypes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Pass/Pass.h>

namespace causalflow::avelang::dialect {
namespace {

bool isStaticAlloca(mlir::Operation *op) {
    if (auto avelangAlloca = mlir::dyn_cast<AveLangMemRefAllocaOp>(op)) {
        auto memrefType =
            mlir::dyn_cast<MemRefType>(avelangAlloca.getResult().getType());
        return memrefType && memrefType.hasStaticShape() &&
               avelangAlloca.getDynamicSizes().empty();
    }

    if (auto memrefAlloca = mlir::dyn_cast<mlir::memref::AllocaOp>(op)) {
        auto memrefType =
            mlir::dyn_cast<mlir::MemRefType>(memrefAlloca.getType());
        return memrefType && memrefType.hasStaticShape() &&
               memrefAlloca.getDynamicSizes().empty();
    }

    return false;
}

class HoistAllocaPass
    : public mlir::PassWrapper<HoistAllocaPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HoistAllocaPass)

    llvm::StringRef getArgument() const final { return "hoist-alloca"; }

    llvm::StringRef getDescription() const final {
        return "Hoist static allocas to the function entry block";
    }

    void runOnOperation() override {
        auto func = getOperation();
        if (func.isDeclaration() || func.getBody().empty()) {
            return;
        }
        auto &entryBlock = func.getBody().front();

        mlir::Operation *insertBefore = nullptr;
        for (mlir::Operation &op : entryBlock) {
            if (!isStaticAlloca(&op)) {
                insertBefore = &op;
                break;
            }
        }

        llvm::SmallVector<mlir::Operation *> allocasToHoist;
        func.walk([&](mlir::Operation *op) {
            if (isStaticAlloca(op)) {
                allocasToHoist.push_back(op);
            }
        });

        for (mlir::Operation *allocaOp : allocasToHoist) {
            if (insertBefore) {
                allocaOp->moveBefore(insertBefore);
            } else {
                allocaOp->moveBefore(&entryBlock, entryBlock.end());
            }
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createHoistAllocaPass() {
    return std::make_unique<HoistAllocaPass>();
}

} // namespace causalflow::avelang::dialect
