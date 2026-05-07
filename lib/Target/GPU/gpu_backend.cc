#include "gpu_backend.h"
#include "avelang/config.h"

#ifdef WITH_AMDGPU
namespace causalflow::avelang::target::amdgpu {
std::unique_ptr<causalflow::avelang::target::gpu::GPUBackendInterface>
CreateAMDGPUBackend();
}
#endif

#ifdef WITH_CUDA
namespace causalflow::avelang::target::nvvm {
std::unique_ptr<causalflow::avelang::target::gpu::GPUBackendInterface>
CreateNVVMBackend();
}
#endif

namespace causalflow::avelang::target::gpu {

GPUBackendRegistry &GPUBackendRegistry::getInstance() {
    static GPUBackendRegistry instance;
    return instance;
}

GPUBackendRegistry::GPUBackendRegistry() {
#ifdef WITH_AMDGPU
    registerBackend(amdgpu::CreateAMDGPUBackend);
#endif

#ifdef WITH_CUDA
    registerBackend(nvvm::CreateNVVMBackend);
#endif
}

void GPUBackendRegistry::registerBackend(BackendFactory factory) {
    backends_.push_back(std::move(factory));
}

std::unique_ptr<GPUBackendInterface>
GPUBackendRegistry::createBackendForTriple(llvm::StringRef triple) {
    for (const auto &factory : backends_) {
        auto backend = factory();
        if (backend->supportsTriple(triple)) {
            return backend;
        }
    }
    return nullptr;
}

} // namespace causalflow::avelang::target::gpu