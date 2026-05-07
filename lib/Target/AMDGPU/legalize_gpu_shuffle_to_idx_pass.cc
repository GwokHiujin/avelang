#include "legalize_gpu_shuffle_to_idx_pass.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/ROCDLDialect.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace causalflow::avelang::target::amdgpu {

namespace {

class AMDGPUShuffleToIDXLowering
    : public mlir::OpRewritePattern<mlir::gpu::ShuffleOp> {
  public:
    using mlir::OpRewritePattern<mlir::gpu::ShuffleOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(mlir::gpu::ShuffleOp op,
                    mlir::PatternRewriter &rewriter) const override {
        if (op.getMode() == mlir::gpu::ShuffleMode::IDX) {
            return mlir::failure();
        }

        auto loc = op.getLoc();
        auto lane =
            mlir::gpu::LaneIdOp::create(rewriter, loc, mlir::IntegerAttr{});
        auto laneI32 = mlir::arith::IndexCastOp::create(
            rewriter, loc, rewriter.getI32Type(), lane.getResult());
        auto width = op.getWidth();
        auto offset = op.getOffset();
        auto localLane =
            mlir::arith::RemUIOp::create(rewriter, loc, laneI32, width);

        mlir::Value targetLane;
        switch (op.getMode()) {
        case mlir::gpu::ShuffleMode::XOR:
            targetLane =
                mlir::arith::XOrIOp::create(rewriter, loc, localLane, offset);
            break;
        case mlir::gpu::ShuffleMode::UP: {
            auto shifted =
                mlir::arith::SubIOp::create(rewriter, loc, localLane, offset);
            auto underflow = mlir::arith::CmpIOp::create(
                rewriter, loc, mlir::arith::CmpIPredicate::ult, localLane,
                offset);
            targetLane = mlir::arith::SelectOp::create(
                rewriter, loc, underflow, localLane, shifted.getResult());
            break;
        }
        case mlir::gpu::ShuffleMode::DOWN: {
            auto shifted =
                mlir::arith::AddIOp::create(rewriter, loc, localLane, offset);
            auto overflow = mlir::arith::CmpIOp::create(
                rewriter, loc, mlir::arith::CmpIPredicate::uge,
                shifted.getResult(), width);
            targetLane = mlir::arith::SelectOp::create(
                rewriter, loc, overflow, localLane, shifted.getResult());
            break;
        }
        case mlir::gpu::ShuffleMode::IDX:
            llvm_unreachable("idx shuffle should not be rewritten");
        }

        auto legalizedShuffle = mlir::gpu::ShuffleOp::create(
            rewriter, loc, op.getValue(), targetLane, width,
            mlir::gpu::ShuffleMode::IDX);
        rewriter.replaceOp(op, legalizedShuffle->getResults());
        return mlir::success();
    }
};

class AMDGPUShuffleIDXToROCDLLowering
    : public mlir::OpRewritePattern<mlir::gpu::ShuffleOp> {
  public:
    using mlir::OpRewritePattern<mlir::gpu::ShuffleOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(mlir::gpu::ShuffleOp op,
                    mlir::PatternRewriter &rewriter) const override {
        if (op.getMode() != mlir::gpu::ShuffleMode::IDX) {
            return mlir::failure();
        }

        auto loc = op.getLoc();
        auto value = op.getValue();
        auto valueType = value.getType();
        auto i32Type = rewriter.getI32Type();

        mlir::Value src = value;
        if (valueType.isF32()) {
            src = mlir::arith::BitcastOp::create(rewriter, loc, i32Type, src);
        } else if (valueType != i32Type) {
            return mlir::failure();
        }

        if (op.getOffset().getType() != i32Type) {
            return mlir::failure();
        }

        auto lane =
            mlir::gpu::LaneIdOp::create(rewriter, loc, mlir::IntegerAttr{});
        auto laneI32 = mlir::arith::IndexCastOp::create(rewriter, loc, i32Type,
                                                        lane.getResult());
        auto localLane =
            mlir::arith::RemUIOp::create(rewriter, loc, laneI32, op.getWidth());
        auto waveBase =
            mlir::arith::SubIOp::create(rewriter, loc, laneI32, localLane);
        auto sourceLane = mlir::arith::AddIOp::create(
            rewriter, loc, waveBase.getResult(), op.getOffset());
        auto four = mlir::arith::ConstantIntOp::create(rewriter, loc, 4, 32);
        auto byteOffset = mlir::arith::MulIOp::create(
            rewriter, loc, sourceLane.getResult(), four);
        mlir::Value shuffled = mlir::ROCDL::DsBpermuteOp::create(
            rewriter, loc, i32Type, byteOffset, src);
        if (valueType.isF32()) {
            shuffled = mlir::arith::BitcastOp::create(rewriter, loc, valueType,
                                                      shuffled);
        }

        auto valid = mlir::arith::ConstantIntOp::create(
            rewriter, loc, 1, op.getValid().getType().getWidth());
        rewriter.replaceOp(op, mlir::ValueRange{shuffled, valid});
        return mlir::success();
    }
};

class LegalizeGPUShuffleToIDXPass
    : public mlir::PassWrapper<LegalizeGPUShuffleToIDXPass,
                               mlir::OperationPass<mlir::gpu::GPUModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeGPUShuffleToIDXPass)

    llvm::StringRef getArgument() const final {
        return "legalize-gpu-shuffle-to-idx";
    }

    llvm::StringRef getDescription() const final {
        return "Lower gpu.shuffle xor/up/down to idx form for AMDGPU";
    }

    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<AMDGPUShuffleToIDXLowering>(&getContext());
        patterns.add<AMDGPUShuffleIDXToROCDLLowering>(&getContext());

        if (mlir::failed(mlir::applyPatternsGreedily(getOperation(),
                                                     std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createLegalizeGPUShuffleToIDXPass() {
    return std::make_unique<LegalizeGPUShuffleToIDXPass>();
}

} // namespace causalflow::avelang::target::amdgpu
