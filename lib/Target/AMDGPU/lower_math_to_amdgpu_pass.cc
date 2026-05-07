#include "lower_math_to_amdgpu_pass.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace causalflow::avelang::target::amdgpu {

namespace {

template <typename OpTy>
class AMDGPUMathIntrinsicLowering : public mlir::OpRewritePattern<OpTy> {
  public:
    AMDGPUMathIntrinsicLowering(mlir::MLIRContext *context,
                                llvm::StringRef intrinsicName)
        : mlir::OpRewritePattern<OpTy>(context),
          intrinsicName_(intrinsicName.str()) {}

    mlir::LogicalResult
    matchAndRewrite(OpTy op, mlir::PatternRewriter &rewriter) const override {
        if (!op.getType().isF32()) {
            return mlir::failure();
        }

        auto intrinsic = mlir::LLVM::CallIntrinsicOp::create(
            rewriter, op.getLoc(), mlir::TypeRange{op.getType()},
            rewriter.getStringAttr(intrinsicName_),
            mlir::ValueRange{op.getOperand()},
            mlir::LLVM::FastmathFlagsAttr::get(
                rewriter.getContext(), mlir::LLVM::FastmathFlags::none));
        rewriter.replaceOp(op, intrinsic.getResult(0));
        return mlir::success();
    }

  private:
    std::string intrinsicName_;
};

template <typename OpTy>
class AMDGPUOCMLCallLowering : public mlir::OpRewritePattern<OpTy> {
  public:
    AMDGPUOCMLCallLowering(mlir::MLIRContext *context,
                           llvm::StringRef ocmlFunctionName)
        : mlir::OpRewritePattern<OpTy>(context),
          ocmlFunctionName_(ocmlFunctionName.str()) {}

    mlir::LogicalResult
    matchAndRewrite(OpTy op, mlir::PatternRewriter &rewriter) const override {
        if (!op.getType().isF32()) {
            return mlir::failure();
        }

        auto gpuModule = op->template getParentOfType<mlir::gpu::GPUModuleOp>();
        if (!gpuModule) {
            return mlir::failure();
        }

        auto ocmlFunc = gpuModule.template lookupSymbol<mlir::func::FuncOp>(
            ocmlFunctionName_);
        if (!ocmlFunc) {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(
                &gpuModule.getBodyRegion().front());
            auto funcType = rewriter.getFunctionType({rewriter.getF32Type()},
                                                     {rewriter.getF32Type()});
            ocmlFunc = mlir::func::FuncOp::create(rewriter, op.getLoc(),
                                                  ocmlFunctionName_, funcType);
            ocmlFunc.setPrivate();
        }

        auto call = mlir::func::CallOp::create(
            rewriter, op.getLoc(), ocmlFunc, mlir::ValueRange{op.getOperand()});
        rewriter.replaceOp(op, call.getResult(0));
        return mlir::success();
    }

  private:
    std::string ocmlFunctionName_;
};

class LowerMathToAMDGPUPass
    : public mlir::PassWrapper<LowerMathToAMDGPUPass,
                               mlir::OperationPass<mlir::gpu::GPUModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMathToAMDGPUPass)

    llvm::StringRef getArgument() const final { return "lower-math-to-amdgpu"; }

    llvm::StringRef getDescription() const final {
        return "Lower math operations to AMDGPU LLVM intrinsics";
    }

    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<AMDGPUMathIntrinsicLowering<mlir::math::AbsFOp>>(
            &getContext(), "llvm.fabs");
        patterns.add<AMDGPUMathIntrinsicLowering<mlir::math::ExpOp>>(
            &getContext(), "llvm.exp");
        patterns.add<AMDGPUMathIntrinsicLowering<mlir::math::Exp2Op>>(
            &getContext(), "llvm.amdgcn.exp2");
        patterns.add<AMDGPUOCMLCallLowering<mlir::math::TanhOp>>(
            &getContext(), "__ocml_tanh_f32");
        patterns.add<AMDGPUMathIntrinsicLowering<mlir::math::LogOp>>(
            &getContext(), "llvm.log");
        patterns.add<AMDGPUMathIntrinsicLowering<mlir::math::Log2Op>>(
            &getContext(), "llvm.log2");
        patterns.add<AMDGPUOCMLCallLowering<mlir::math::ErfOp>>(
            &getContext(), "__ocml_erf_f32");

        if (mlir::failed(mlir::applyPatternsGreedily(getOperation(),
                                                     std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerMathToAMDGPUPass() {
    return std::make_unique<LowerMathToAMDGPUPass>();
}

} // namespace causalflow::avelang::target::amdgpu
