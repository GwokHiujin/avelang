#pragma once

#include <memory>
#include <mlir/Pass/Pass.h>

namespace causalflow::avelang::target::gpu {

// GPU outlining pass that moves functions with ave.gpu_func ==
// kPrivateFunction or kGlobalKernel to a GPU module and converts them to
// gpu.func
std::unique_ptr<mlir::Pass> createGpuOutliningPass();

// Link intrinsic implementation pass that links intrinsic libraries
// into the main module
std::unique_ptr<mlir::Pass> createLinkIntrinsicImplementationPass();

// GPU attribute alloca pass that adds workgroup address space to
// memref.alloca operations without address space
std::unique_ptr<mlir::Pass> createGPUAttributeAllocaPass();

// Register all GPU target passes
void registerAllPasses();

} // namespace causalflow::avelang::target::gpu
