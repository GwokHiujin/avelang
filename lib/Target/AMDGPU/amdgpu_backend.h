#pragma once

#include "Target/GPU/gpu_backend.h"
#include <memory>

namespace causalflow::avelang::target::amdgpu {

class AMDGPUBackend
    : public causalflow::avelang::target::gpu::GPUBackendInterface {
  public:
    void buildLoweringPipeline(
        mlir::OpPassManager &pm,
        const causalflow::avelang::target::gpu::GPUCompilationOptions &options)
        override;

    bool supportsTriple(llvm::StringRef triple) const override;

    std::string getName() const override;

    llvm::Expected<std::string>
    generateBinary(llvm::Module &module,
                   const causalflow::avelang::target::gpu::GPUCompilationOptions
                       &options) override;

    llvm::Expected<std::string> generateAssembly(
        llvm::Module &module,
        const causalflow::avelang::target::gpu::GPUCompilationOptions &options)
        override;

    void EnsureInitialized() override;
};

std::unique_ptr<causalflow::avelang::target::gpu::GPUBackendInterface>
CreateAMDGPUBackend();

} // namespace causalflow::avelang::target::amdgpu