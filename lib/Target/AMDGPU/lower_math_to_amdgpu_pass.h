#pragma once

#include <memory>
#include <mlir/Pass/Pass.h>

namespace causalflow::avelang::target::amdgpu {

std::unique_ptr<mlir::Pass> createLowerMathToAMDGPUPass();

} // namespace causalflow::avelang::target::amdgpu
