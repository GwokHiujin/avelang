#include "IR/ir_context.h"
#include "Target/GPU/lower_to_llvm.h"
#include "nvvm_backend.h"
#include "test/utils/cuda_helper.h"

#include <gtest/gtest.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>

#include <memory>
#include <random>
#include <vector>

using namespace llvm;
using namespace causalflow::avelang::target::nvvm;
using namespace causalflow::avelang::target::gpu;

static const std::string kAxpyMLIRCode = R"(
module {
  func.func @axpy(%arg0: i32 {llvm.name = "a"}, %arg1: i32 {llvm.name = "b"}, %arg2: memref<64xi32> {llvm.name = "x"}, %arg3: memref<64xi32> {llvm.name = "y"}) attributes {ave.gpu_func = 2 : i32} {
    %c0_i32 = arith.constant 0 : i32
    %block_id_x = gpu.block_id  x
    %c0_i32_0 = arith.constant 0 : i32
    %block_dim_x = gpu.block_dim  x
    %0 = arith.muli %block_id_x, %block_dim_x : index
    %c0_i32_1 = arith.constant 0 : i32
    %thread_id_x = gpu.thread_id  x
    %1 = arith.addi %0, %thread_id_x : index
    %2 = memref.load %arg2[%1] : memref<64xi32>
    %3 = arith.muli %arg0, %2 : i32
    %4 = arith.addi %3, %arg1 : i32
    memref.store %4, %arg3[%1] : memref<64xi32>
    return
  }
})";

class NVVMCodegenTest : public ::testing::Test {
  protected:
    void SetUp() override {
        backend = std::make_unique<NVVMBackend>();
        llvmContext = std::make_unique<LLVMContext>();
        ir_context = causalflow::avelang::ir::IRContext::Create();

        static std::once_flag initNVPTXFlag;
        std::call_once(initNVPTXFlag, []() {
            LLVMInitializeNVPTXTarget();
            LLVMInitializeNVPTXTargetInfo();
            LLVMInitializeNVPTXTargetMC();
            LLVMInitializeNVPTXAsmPrinter();
        });

        // Initialize compilation options once
        options.triple = "nvptx64-nvidia-cuda";
        options.chipset = "sm_80";
        options.optimization_level = 2;
    }

    void ParseMLIRString(const std::string &mlirCode) {
        LowerToLLVM compiler(ir_context.get());
        module = mlir::parseSourceString<mlir::ModuleOp>(mlirCode,
                                                         compiler.getContext());
        ASSERT_NE(module.get(), nullptr);
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    }

    void CompileToLLVMIR() {
        LowerToLLVM compiler(ir_context.get());
        llvmModule = compiler.compile(*module, *llvmContext, options);

        ASSERT_NE(llvmModule, nullptr);
        ASSERT_FALSE(llvmModule->empty());
    }

    // Compile LLVM IR to PTX using the NVVM backend
    void CompileToPTX() {
        auto ptxResult = backend->generateBinary(*llvmModule, options);
        ASSERT_TRUE((bool)ptxResult);

        ptx_code_ = *ptxResult;
        ASSERT_FALSE(ptx_code_.empty());
    }

    std::unique_ptr<NVVMBackend> backend;
    std::unique_ptr<LLVMContext> llvmContext;
    std::unique_ptr<causalflow::avelang::ir::IRContext> ir_context;
    GPUCompilationOptions options;
    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::unique_ptr<Module> llvmModule;
    std::string ptx_code_;
};

TEST_F(NVVMCodegenTest, ExecuteAxpyOnGPU) {
    static const int kN = 64;
    ParseMLIRString(kAxpyMLIRCode);
    CompileToLLVMIR();
    CompileToPTX();

    CUmodule cuModule;
    CUfunction axpyFunction;
    CUcontext cuContext;

    CUresult initResult = cuInit(0);
    if (initResult != CUDA_SUCCESS) {
        const char *error_string = nullptr;
        cuGetErrorString(initResult, &error_string);
        GTEST_SKIP() << "CUDA initialization failed: "
                     << (error_string ? error_string : "unknown error");
        return;
    }

    int deviceCount = 0;
    CUresult deviceCountResult = cuDeviceGetCount(&deviceCount);
    if (deviceCountResult != CUDA_SUCCESS || deviceCount == 0) {
        const char *error_string = nullptr;
        cuGetErrorString(deviceCountResult, &error_string);
        GTEST_SKIP() << "CUDA device unavailable: "
                     << (error_string ? error_string : "no devices");
        return;
    }
    CUdevice device;
    CheckCuError(cuDeviceGet(&device, 0));

#if CUDA_VERSION >= 13000
    // CUDA version >= 13.0, use new context creation API
    CUctxCreateParams ctxParams = {0};
    CheckCuError(cuCtxCreate(&cuContext, &ctxParams, 0, device));
#else
    // CUDA version < 13.0, use legacy context creation API
    CheckCuError(cuCtxCreate(&cuContext, 0, device));
#endif
    CheckCuError(cuModuleLoadData(&cuModule, ptx_code_.c_str()));
    CheckCuError(cuModuleGetFunction(&axpyFunction, cuModule, "axpy"));

    CUdeviceptr d_x, d_y;
    CheckCuError(cuMemAlloc(&d_x, kN * sizeof(int)));
    CheckCuError(cuMemAlloc(&d_y, kN * sizeof(int)));

    std::vector<int> h_x(kN), h_y(kN);
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(0, 100);
    for (int i = 0; i < kN; i++) {
        h_x[i] = dis(gen);
    }
    CheckCuError(cuMemcpyHtoD(d_x, h_x.data(), kN * sizeof(int)));
    CheckCuError(cuMemsetD32(d_y, 0, kN));

    struct {
        int a;
        int b;
        void *x;
        void *y;
    } params = {
        .a = 2,
        .b = 3,
        .x = reinterpret_cast<void *>(d_x),
        .y = reinterpret_cast<void *>(d_y),
    };
    static constexpr size_t kParamSize = sizeof(params);

    void *config[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &params,
                      CU_LAUNCH_PARAM_BUFFER_SIZE,
                      const_cast<size_t *>(&kParamSize), CU_LAUNCH_PARAM_END};

    CheckCuError(cuLaunchKernel(axpyFunction, 1, 1, 1, kN, 1, 1, 0, nullptr,
                                nullptr, config));
    CheckCuError(cuMemcpyDtoH(h_y.data(), d_y, kN * sizeof(int)));
    CheckCuError(cuCtxSynchronize());

    for (int i = 0; i < kN; i++) {
        ASSERT_EQ(h_y[i], 2 * h_x[i] + 3);
    }

    CheckCuError(cuMemFree(d_x));
    CheckCuError(cuMemFree(d_y));
    CheckCuError(cuModuleUnload(cuModule));
    CheckCuError(cuCtxDestroy(cuContext));
}
