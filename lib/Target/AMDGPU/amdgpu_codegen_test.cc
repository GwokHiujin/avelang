#include "IR/ir_context.h"
#include "Target/GPU/lower_to_llvm.h"
#include "amdgpu_backend.h"
#include "test/utils/hip_helper.h"

#include <gtest/gtest.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>

#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace llvm;
using namespace causalflow::avelang::target::amdgpu;
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

static std::string DetectHipChipset() {
    CheckHipError(hipInit(0));
    int device = 0;
    CheckHipError(hipGetDevice(&device));
    hipDeviceProp_t props;
    CheckHipError(hipGetDeviceProperties(&props, device));
    std::string arch(props.gcnArchName);
    auto delimiter = arch.find(':');
    if (delimiter != std::string::npos) {
        arch.resize(delimiter);
    }
    if (arch.empty()) {
        arch = "gfx90a";
    }
    return arch;
}

class AMDGPUCodegenTest : public ::testing::Test {
  protected:
    void SetUp() override {
        backend = std::make_unique<AMDGPUBackend>();
        llvmContext = std::make_unique<LLVMContext>();
        ir_context = causalflow::avelang::ir::IRContext::Create();

        backend->EnsureInitialized();
        // Initialize compilation options once
        options.triple = "amdgcn-amd-amdhsa";
        chipset_ = DetectHipChipset();
        options.chipset = chipset_;
        options.optimization_level = 3;
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

    // Compile LLVM IR to AMDGPU assembly using the AMDGPU backend
    void CompileToAMDGPUAssembly() {
        auto asmResult = backend->generateBinary(*llvmModule, options);
        ASSERT_TRUE((bool)asmResult);

        amdgpu_asm_code_ = *asmResult;
        ASSERT_FALSE(amdgpu_asm_code_.empty());
    }

    std::unique_ptr<AMDGPUBackend> backend;
    std::unique_ptr<LLVMContext> llvmContext;
    std::unique_ptr<causalflow::avelang::ir::IRContext> ir_context;
    GPUCompilationOptions options;
    std::string chipset_;
    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::unique_ptr<Module> llvmModule;
    std::string amdgpu_asm_code_;
};

TEST_F(AMDGPUCodegenTest, ExecuteAxpyOnGPU) {
    static const int kN = 64;
    ParseMLIRString(kAxpyMLIRCode);
    CompileToLLVMIR();
    CompileToAMDGPUAssembly();

    hipModule_t hipModule;
    hipFunction_t axpyFunction;

    CheckHipError(hipInit(0));
    CheckHipError(hipModuleLoadDataEx(&hipModule, amdgpu_asm_code_.data(), 0,
                                      nullptr, nullptr));

    CheckHipError(hipModuleGetFunction(&axpyFunction, hipModule, "axpy"));

    hipDeviceptr_t d_x, d_y;
    CheckHipError(hipMalloc(&d_x, kN * sizeof(int)));
    CheckHipError(hipMalloc(&d_y, kN * sizeof(int)));

    std::vector<int> h_x(kN), h_y(kN);
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(0, 100);
    for (int i = 0; i < kN; i++) {
        h_x[i] = dis(gen);
    }
    CheckHipError(hipMemcpyHtoD(d_x, h_x.data(), kN * sizeof(int)));
    CheckHipError(hipMemsetD32(d_y, 0, kN));

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
    size_t paramSize = sizeof(params);

    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &params,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &paramSize,
                      HIP_LAUNCH_PARAM_END};

    CheckHipError(hipModuleLaunchKernel(axpyFunction, 1, 1, 1, kN, 1, 1, 0,
                                        nullptr, nullptr, config));
    CheckHipError(hipMemcpyDtoH(h_y.data(), d_y, kN * sizeof(int)));
    CheckHipError(hipDeviceSynchronize());

    for (int i = 0; i < kN; i++) {
        ASSERT_EQ(h_y[i], 2 * h_x[i] + 3);
    }

    CheckHipError(hipFree(d_x));
    CheckHipError(hipFree(d_y));
    CheckHipError(hipModuleUnload(hipModule));
}
