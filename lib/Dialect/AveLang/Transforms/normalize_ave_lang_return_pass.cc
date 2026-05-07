#include "normalize_ave_lang_return_pass.h"

#include "Dialect/AveLang/IR/AveLangOps.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/Pass.h>

#include <optional>
#include <type_traits>

namespace causalflow::avelang::dialect {

namespace {

std::optional<llvm::SmallVector<mlir::Value>>
createDefaultValues(mlir::OpBuilder &builder, mlir::Location loc,
                    llvm::ArrayRef<mlir::Type> types) {
    llvm::SmallVector<mlir::Value> values;
    values.reserve(types.size());
    for (auto type : types) {
        if (type.isIndex()) {
            values.push_back(
                mlir::arith::ConstantIndexOp::create(builder, loc, 0));
            continue;
        }
        if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
            values.push_back(
                mlir::arith::ConstantIntOp::create(builder, loc, intType, 0));
            continue;
        }
        if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
            auto attr = builder.getFloatAttr(floatType, 0.0);
            values.push_back(
                mlir::arith::ConstantOp::create(builder, loc, type, attr));
            continue;
        }
        if (auto vectorType = mlir::dyn_cast<mlir::VectorType>(type)) {
            auto elemType = vectorType.getElementType();
            mlir::Attribute zeroAttr;
            if (elemType.isIndex()) {
                zeroAttr = builder.getIndexAttr(0);
            } else if (auto intType =
                           mlir::dyn_cast<mlir::IntegerType>(elemType)) {
                zeroAttr = builder.getIntegerAttr(intType, 0);
            } else if (auto floatType =
                           mlir::dyn_cast<mlir::FloatType>(elemType)) {
                zeroAttr = builder.getFloatAttr(floatType, 0.0);
            } else {
                return std::nullopt;
            }
            auto denseAttr = mlir::DenseElementsAttr::get(vectorType, zeroAttr);
            values.push_back(
                mlir::arith::ConstantOp::create(builder, loc, type, denseAttr));
            continue;
        }
        return std::nullopt;
    }
    return values;
}

template <typename FuncLikeOp>
llvm::SmallVector<mlir::Type> getFunctionResultTypes(FuncLikeOp funcOp) {
    llvm::SmallVector<mlir::Type> resultTypes;
    llvm::append_range(resultTypes, funcOp.getFunctionType().getResults());
    return resultTypes;
}

template <typename FuncLikeOp>
void createFinalReturn(mlir::OpBuilder &builder, mlir::Location loc,
                       FuncLikeOp funcOp, mlir::ValueRange values) {
    if constexpr (std::is_same_v<FuncLikeOp, mlir::func::FuncOp>) {
        mlir::func::ReturnOp::create(builder, loc, values);
    } else {
        mlir::gpu::ReturnOp::create(builder, loc, values);
    }
}

struct BranchInfo {
    ReturnOp returnOp;
    mlir::Operation *placeholderReturn = nullptr;
    mlir::scf::YieldOp yieldOp;
};

template <typename FuncLikeOp>
bool isPlaceholderReturnOp(mlir::Operation *op, FuncLikeOp) {
    if constexpr (std::is_same_v<FuncLikeOp, mlir::func::FuncOp>) {
        return mlir::isa<mlir::func::ReturnOp>(op);
    } else {
        return mlir::isa<mlir::gpu::ReturnOp>(op);
    }
}

mlir::ValueRange getReturnOperands(mlir::Operation *op) {
    if (auto funcReturn = mlir::dyn_cast<mlir::func::ReturnOp>(op)) {
        return funcReturn.getOperands();
    }
    if (auto gpuReturn = mlir::dyn_cast<mlir::gpu::ReturnOp>(op)) {
        return gpuReturn.getOperands();
    }
    return {};
}

template <typename FuncLikeOp>
std::optional<BranchInfo> getBranchInfo(mlir::Region &region,
                                        FuncLikeOp funcOp) {
    if (region.empty()) {
        return BranchInfo{};
    }
    auto &block = region.front();
    if (auto yieldOp =
            mlir::dyn_cast<mlir::scf::YieldOp>(block.getTerminator())) {
        ReturnOp returnOp;
        mlir::Operation *placeholderReturn = nullptr;
        if (auto *prev = yieldOp->getPrevNode()) {
            if (auto marker = mlir::dyn_cast<ReturnOp>(prev)) {
                returnOp = marker;
            } else if (isPlaceholderReturnOp(prev, funcOp)) {
                placeholderReturn = prev;
                if (auto *prevPrev = prev->getPrevNode()) {
                    returnOp = mlir::dyn_cast<ReturnOp>(prevPrev);
                }
            }
        }
        return BranchInfo{returnOp, placeholderReturn, yieldOp};
    }
    return std::nullopt;
}

template <typename FuncLikeOp>
bool branchNeedsRewrite(mlir::Region &region, FuncLikeOp funcOp) {
    auto info = getBranchInfo(region, funcOp);
    return info && info->returnOp;
}

template <typename FuncLikeOp>
mlir::LogicalResult
rewriteIfWithEarlyReturn(mlir::scf::IfOp ifOp, FuncLikeOp funcOp,
                         llvm::ArrayRef<mlir::Type> resultTypes,
                         mlir::Block *exitBlock) {
    auto thenInfo = getBranchInfo(ifOp.getThenRegion(), funcOp);
    if (!thenInfo) {
        return ifOp.emitOpError()
               << "expected single-block then region terminated by scf.yield";
    }

    bool hasElse = !ifOp.getElseRegion().empty();
    std::optional<BranchInfo> elseInfo = BranchInfo{};
    if (hasElse) {
        elseInfo = getBranchInfo(ifOp.getElseRegion(), funcOp);
        if (!elseInfo) {
            return ifOp.emitOpError() << "expected single-block else region "
                                         "terminated by scf.yield";
        }
    }

    if (!thenInfo->returnOp && (!hasElse || !elseInfo->returnOp)) {
        return mlir::success();
    }

    auto *parentBlock = ifOp->getBlock();
    auto loc = ifOp.getLoc();
    mlir::OpBuilder builder(ifOp);

    auto falseValue = mlir::arith::ConstantIntOp::create(builder, loc, 0, 1);
    auto trueValue = mlir::arith::ConstantIntOp::create(builder, loc, 1, 1);

    auto defaultValues = createDefaultValues(builder, loc, resultTypes);
    if (!defaultValues) {
        return ifOp.emitOpError()
               << "cannot materialize placeholder return values";
    }

    llvm::SmallVector<mlir::Type> ifResultTypes;
    ifResultTypes.push_back(builder.getI1Type());
    ifResultTypes.append(resultTypes.begin(), resultTypes.end());

    mlir::Operation *nextOp = ifOp->getNextNode();
    mlir::Block *continueBlock = nullptr;
    if (nextOp) {
        continueBlock = parentBlock->splitBlock(mlir::Block::iterator(nextOp));
    } else {
        auto *parentRegion = parentBlock->getParent();
        continueBlock = builder.createBlock(
            parentRegion, std::next(mlir::Region::iterator(parentBlock)));
    }

    auto newIf = mlir::scf::IfOp::create(builder, loc, ifResultTypes,
                                         ifOp.getCondition(),
                                         /*withElseRegion=*/true);

    auto populateRegion = [&](mlir::Region &dst, mlir::Region &src,
                              BranchInfo info) {
        auto &dstBlock = dst.front();
        while (!dstBlock.empty()) {
            dstBlock.back().erase();
        }
        if (!src.empty()) {
            dstBlock.getOperations().splice(dstBlock.end(),
                                            src.front().getOperations());
        }

        builder.setInsertionPointToEnd(&dstBlock);
        if (info.returnOp) {
            if (!dstBlock.empty()) {
                if (auto trailingYield =
                        mlir::dyn_cast<mlir::scf::YieldOp>(&dstBlock.back())) {
                    trailingYield.erase();
                }
            }
            llvm::SmallVector<mlir::Value> yieldValues;
            yieldValues.push_back(trueValue);
            if (info.placeholderReturn) {
                llvm::append_range(yieldValues,
                                   getReturnOperands(info.placeholderReturn));
                info.placeholderReturn->erase();
            } else {
                llvm::append_range(yieldValues, info.returnOp.getOperands());
            }
            info.returnOp.erase();
            mlir::scf::YieldOp::create(builder, loc, yieldValues);
            return;
        }

        if (!dstBlock.empty()) {
            if (auto trailingYield =
                    mlir::dyn_cast<mlir::scf::YieldOp>(&dstBlock.back())) {
                trailingYield.erase();
            }
        }
        llvm::SmallVector<mlir::Value> yieldValues;
        yieldValues.push_back(falseValue);
        llvm::append_range(yieldValues, *defaultValues);
        mlir::scf::YieldOp::create(builder, loc, yieldValues);
    };

    populateRegion(newIf.getThenRegion(), ifOp.getThenRegion(), *thenInfo);
    if (hasElse) {
        populateRegion(newIf.getElseRegion(), ifOp.getElseRegion(), *elseInfo);
    } else {
        auto emptyElse = BranchInfo{};
        populateRegion(newIf.getElseRegion(), ifOp.getElseRegion(), emptyElse);
    }

    ifOp.erase();

    builder.setInsertionPointToEnd(parentBlock);
    llvm::SmallVector<mlir::Value> returnValues;
    for (unsigned i = 1; i < newIf.getNumResults(); ++i) {
        returnValues.push_back(newIf.getResult(i));
    }
    mlir::cf::CondBranchOp::create(builder, loc, newIf.getResult(0), exitBlock,
                                   returnValues, continueBlock,
                                   mlir::ValueRange{});

    return mlir::success();
}

template <typename FuncLikeOp>
mlir::LogicalResult normalizeFunction(FuncLikeOp funcOp) {
    if (funcOp.isExternal()) {
        return mlir::success();
    }

    auto resultTypes = getFunctionResultTypes(funcOp);
    auto &body = funcOp.getBody();
    auto loc = funcOp.getLoc();

    auto *exitBlock = new mlir::Block();
    body.push_back(exitBlock);
    for (auto type : resultTypes) {
        exitBlock->addArgument(type, loc);
    }

    mlir::OpBuilder builder(funcOp.getContext());
    builder.setInsertionPointToEnd(exitBlock);
    createFinalReturn(builder, loc, funcOp, exitBlock->getArguments());

    bool changed = true;
    while (changed) {
        changed = false;

        llvm::SmallVector<mlir::Block *> blocks;
        for (auto &block : body.getBlocks()) {
            if (&block != exitBlock) {
                blocks.push_back(&block);
            }
        }

        for (auto *block : blocks) {
            if (auto *terminator = block->getTerminator()) {
                if (isPlaceholderReturnOp(terminator, funcOp)) {
                    ReturnOp marker;
                    if (auto *prev = terminator->getPrevNode()) {
                        marker = mlir::dyn_cast<ReturnOp>(prev);
                    }
                    builder.setInsertionPoint(terminator);
                    mlir::cf::BranchOp::create(builder, terminator->getLoc(),
                                               exitBlock,
                                               getReturnOperands(terminator));
                    terminator->erase();
                    if (marker) {
                        marker.erase();
                    }
                    changed = true;
                    break;
                }
            }

            for (auto &op : llvm::make_early_inc_range(*block)) {
                auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op);
                if (!ifOp || ifOp.getNumResults() != 0) {
                    continue;
                }
                if (!branchNeedsRewrite(ifOp.getThenRegion(), funcOp) &&
                    !branchNeedsRewrite(ifOp.getElseRegion(), funcOp)) {
                    continue;
                }
                if (mlir::failed(rewriteIfWithEarlyReturn(
                        ifOp, funcOp, resultTypes, exitBlock))) {
                    return mlir::failure();
                }
                changed = true;
                break;
            }

            if (changed) {
                break;
            }
        }
    }

    for (auto &block : body.getBlocks()) {
        if (&block == exitBlock ||
            (!block.empty() &&
             block.back().template hasTrait<mlir::OpTrait::IsTerminator>())) {
            continue;
        }
        builder.setInsertionPointToEnd(&block);
        mlir::cf::BranchOp::create(builder, loc, exitBlock, mlir::ValueRange{});
    }

    bool hasResidualReturn = false;
    funcOp->walk([&](ReturnOp returnOp) { hasResidualReturn = true; });
    if (hasResidualReturn) {
        funcOp.emitOpError()
            << "normalize-ave-lang-return-pass could not eliminate all "
               "ave.return operations";
        return mlir::failure();
    }

    return mlir::success();
}

class NormalizeAveLangReturnPass
    : public mlir::PassWrapper<NormalizeAveLangReturnPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeAveLangReturnPass)

    void runOnOperation() override {
        auto module = getOperation();
        bool failed = false;
        module.walk([&](mlir::func::FuncOp funcOp) {
            if (failed) {
                return;
            }
            if (mlir::failed(normalizeFunction(funcOp))) {
                failed = true;
            }
        });
        if (failed) {
            signalPassFailure();
            return;
        }
        module.walk([&](mlir::gpu::GPUFuncOp gpuFunc) {
            if (mlir::failed(normalizeFunction(gpuFunc))) {
                signalPassFailure();
            }
        });
    }

    llvm::StringRef getArgument() const final {
        return "normalize-ave-lang-return";
    }

    llvm::StringRef getDescription() const final {
        return "Rewrite ave.return into a single-exit control-flow form";
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createNormalizeAveLangReturnPass() {
    return std::make_unique<NormalizeAveLangReturnPass>();
}

} // namespace causalflow::avelang::dialect
