#pragma once

#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/DiagnosticIDs.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/FileManager.h>
#include <clang/Basic/LangOptions.h>
#include <clang/Basic/SourceManager.h>
#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <memory>
#include <unordered_map>

namespace causalflow::avelang::basic {

/// Unified error codes for both AST parsing and MLIR generation
enum class DiagnosticCode {
    // AST Parser Error Codes
    kUnknownASTNode,
    kUnexpectedClass,

    // MLIR Generator Error Codes
    kUnexpectedReturnType,
    kUnimplemented,
    kIndexDimensionMismatch,
    kSymbolNotFound,
    kTypeMismatch,
    kInvalidArguments,
};

/// Diagnostic manager that owns and manages the DiagnosticsEngine lifecycle
class DiagnosticManager : public llvm::RefCountedBase<DiagnosticManager> {
  public:
    explicit DiagnosticManager(clang::DiagnosticConsumer *consumer);
    ~DiagnosticManager();
    clang::DiagnosticsEngine *GetEngine() { return diagnostics_engine_.get(); }
    clang::FileManager &GetFileManager() { return file_manager_; }
    clang::SourceManager &GetSourceManager() { return *source_manager_; }

    clang::DiagnosticBuilder
    Report(DiagnosticCode code,
           clang::SourceLocation loc = clang::SourceLocation());

  private:
    /// Initialize and register all diagnostic messages
    void Initialize();

    /// Register all diagnostic messages with clang
    void RegisterDiagnostics();

    // Owned diagnostic infrastructure
    llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagnostic_ids_;
    std::unique_ptr<clang::DiagnosticOptions> diagnostic_options_;
    clang::DiagnosticConsumer *diagnostic_consumer_;
    std::unique_ptr<clang::DiagnosticsEngine> diagnostics_engine_;

    // File and source management
    clang::FileManager file_manager_;
    llvm::IntrusiveRefCntPtr<clang::SourceManager> source_manager_;
    clang::LangOptions lang_options_;

    // Custom diagnostic ID mappings
    std::unordered_map<DiagnosticCode, unsigned> custom_diagnostic_ids_;
};

} // namespace causalflow::avelang::basic
