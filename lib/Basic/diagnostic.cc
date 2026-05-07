#include "diagnostic.h"
#include "Utils/assert.h"
#include "clang/Basic/DiagnosticIDs.h"
#include <clang/Basic/LangOptions.h>

namespace causalflow::avelang::basic {

DiagnosticManager::DiagnosticManager(clang::DiagnosticConsumer *consumer)
    : diagnostic_consumer_(consumer),
      file_manager_(clang::FileSystemOptions()) {
    Initialize();
    if (diagnostic_consumer_) {
        diagnostic_consumer_->BeginSourceFile(lang_options_, nullptr);
    }
}

DiagnosticManager::~DiagnosticManager() {
    if (diagnostic_consumer_) {
        diagnostic_consumer_->EndSourceFile();
    }
}

void DiagnosticManager::Initialize() {
    diagnostic_ids_ = new clang::DiagnosticIDs();
    diagnostic_options_ = std::make_unique<clang::DiagnosticOptions>();

    diagnostics_engine_ = std::make_unique<clang::DiagnosticsEngine>(
        diagnostic_ids_, *diagnostic_options_, diagnostic_consumer_, false);
    source_manager_ =
        new clang::SourceManager(*diagnostics_engine_, file_manager_);

    RegisterDiagnostics();
}

void DiagnosticManager::RegisterDiagnostics() {
    custom_diagnostic_ids_[DiagnosticCode::kUnknownASTNode] =
        diagnostics_engine_->getCustomDiagID(clang::DiagnosticsEngine::Error,
                                             "Unknown AST class: '%0'");

    custom_diagnostic_ids_[DiagnosticCode::kUnexpectedClass] =
        diagnostics_engine_->getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "Unexpected AST class: '%0', expecting: '%1'");

    custom_diagnostic_ids_[DiagnosticCode::kUnexpectedReturnType] =
        diagnostics_engine_->getCustomDiagID(clang::DiagnosticsEngine::Error,
                                             "Unexpected function return type");

    custom_diagnostic_ids_[DiagnosticCode::kUnimplemented] =
        diagnostics_engine_->getCustomDiagID(clang::DiagnosticsEngine::Error,
                                             "Unimplemented: '%0'");

    custom_diagnostic_ids_[DiagnosticCode::kIndexDimensionMismatch] =
        diagnostics_engine_->getCustomDiagID(clang::DiagnosticsEngine::Error,
                                             "Index dimension mismatch: '%0'");

    custom_diagnostic_ids_[DiagnosticCode::kSymbolNotFound] =
        diagnostics_engine_->getCustomDiagID(clang::DiagnosticsEngine::Error,
                                             "Symbol not found: '%0'");

    custom_diagnostic_ids_[DiagnosticCode::kTypeMismatch] =
        diagnostics_engine_->getCustomDiagID(clang::DiagnosticsEngine::Error,
                                             "Type mismatch: '%0'");

    custom_diagnostic_ids_[DiagnosticCode::kInvalidArguments] =
        diagnostics_engine_->getCustomDiagID(clang::DiagnosticsEngine::Error,
                                             "Invalid arguments: '%0'");
}

clang::DiagnosticBuilder DiagnosticManager::Report(DiagnosticCode code,
                                                   clang::SourceLocation loc) {
    auto it = custom_diagnostic_ids_.find(code);
    SS_ASSERT(it != custom_diagnostic_ids_.end());
    return diagnostics_engine_->Report(loc, it->second);
}

} // namespace causalflow::avelang::basic
