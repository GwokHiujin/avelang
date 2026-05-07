#include "avelang_parser.h"

#include <gtest/gtest.h>

namespace causalflow::avelang::frontend {

class CaptureDiagnosticConsumer : public clang::DiagnosticConsumer {
    clang::SmallVector<clang::StoredDiagnostic, 4> Errors;

  public:
    void HandleDiagnostic(clang::DiagnosticsEngine::Level level,
                          const clang::Diagnostic &Info) override {
        if (level >= clang::DiagnosticsEngine::Error)
            Errors.push_back(clang::StoredDiagnostic(level, Info));
    }

    std::string
    GetErrorMessages(const clang::SourceManager *SM = nullptr) const {
        std::string messages;
        for (const auto &error : Errors) {
            messages += "Error";

            // Add location information if available
            if (error.getLocation().isValid() && SM) {
                clang::SourceLocation loc = error.getLocation();
                if (loc.isFileID()) {
                    // Get line and column information
                    clang::PresumedLoc ploc = SM->getPresumedLoc(loc);
                    if (ploc.isValid()) {
                        messages +=
                            " at line " + std::to_string(ploc.getLine()) +
                            ", column " + std::to_string(ploc.getColumn());
                    }
                }
            }

            messages += ": " + error.getMessage().str() + "\n";
        }
        return messages;
    }

    void Clear() { Errors.clear(); }
};

class AveLangParserTest : public ::testing::Test {
  protected:
    virtual void SetUp() override;
    void TestParseOkay(const std::string &source_code);

    llvm::IntrusiveRefCntPtr<ast::ASTContext> context_;
    CaptureDiagnosticConsumer diagConsumer_;
    llvm::IntrusiveRefCntPtr<basic::DiagnosticManager> diagnostics_;
};

void AveLangParserTest::SetUp() {
    context_ = new ast::ASTContext();
    diagConsumer_.Clear();
    diagnostics_ = new basic::DiagnosticManager(&diagConsumer_);
}

void AveLangParserTest::TestParseOkay(const std::string &source_code) {
    AveLangParser parser(context_, diagnostics_);
    parser.ParseImpl(clang::FileID(), "stdin", source_code);
    ast::ASTNode *module = parser.GetModule();

    const clang::SourceManager &SM = diagnostics_->GetSourceManager();
    ASSERT_NE(module, nullptr) << diagConsumer_.GetErrorMessages(&SM);
    ASSERT_FALSE(diagnostics_->GetEngine()->hasErrorOccurred())
        << diagConsumer_.GetErrorMessages(&SM);
}

TEST_F(AveLangParserTest, ParseModule) {
    static const std::string kSourceCode = R"""""(
@avelang.jit
def matmul(a: ss.Tensor, b: ss.Tensor) -> ss.Tensor:
    return a @ b
)""""";
    TestParseOkay(kSourceCode);
}

TEST_F(AveLangParserTest, ParseNaiveFP16Gemm) {
    static const std::string kSourceCode = R"""""(
@T.prim_func
def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
):
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.make_shared((block_M, block_K), dtype)
        B_shared = T.make_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * block_M, bx * block_N])

return gemm
)""""";
    TestParseOkay(kSourceCode);
}

TEST_F(AveLangParserTest, ParseInt4FP16Gemm) {
    static const std::string kSourceCode = R"""""(
@T.prim_func
def main(
        A: T.Buffer(A_shape, in_dtype),
        B: T.Buffer(B_shape, storage_dtype),
        C: T.Buffer((M, N), out_dtype),
):
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        A_shared = T.make_shared(A_shared_shape, in_dtype)
        B_shared = T.make_shared(B_shared_shape, storage_dtype)
        B_local = T.make_local([local_size_compressed], storage_dtype)
        B_dequantize_local = T.make_local([local_size], in_dtype)
        B_dequantize_shared = T.make_shared(B_dequantize_shared_shape, in_dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        tx = T.thread_binding(0, threads, thread="threadIdx.x")

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)

            for i in T.serial(block_N * block_K // num_elems_per_byte //
                              (threads * local_size_compressed)):
                for v in T.vectorized(0, local_size_compressed):
                    index = i * threads * local_size_compressed + tx * local_size_compressed + v
                    vi = index // (block_K // num_elems_per_byte)
                    vj = index % (block_K // num_elems_per_byte)
                    B_local[v] = B_shared[vi, vj]
                for v in T.serial(0, local_size):
                    B_dequantize_local[v] = _tir_packed_to_unsigned_convert(
                        storage_type, storage_nbit)(
                            num_bits,
                            B_local[v // num_elems_per_byte],
                            v % num_elems_per_byte,
                            dtype=in_dtype,
                        )
                for v in T.vectorized(0, local_size):
                    index = i * threads * local_size + tx * local_size + v
                    vi = index // block_K
                    vj = index % block_K
                    B_dequantize_shared[vi, vj] = B_dequantize_local[v]

            T.gemm(A_shared, B_dequantize_shared, C_local, transpose_B=True)

        T.copy(C_local, C[by * block_M, bx * block_N])
)""""";
    TestParseOkay(kSourceCode);
}

} // namespace causalflow::avelang::frontend
