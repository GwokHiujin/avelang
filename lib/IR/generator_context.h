#pragma once

#include "AST/ast.h"
#include "Basic/diagnostic.h"
#include "ir_context.h"
#include "symbol_table.h"

#include <clang/Basic/SourceLocation.h>
#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <memory>
#include <mlir/IR/Location.h>
#include <utility>

namespace causalflow::avelang::ir {

class ExprGenerator;
class FunctionGenerator;
struct GeneratorContext {
    IRContext *ir_context;
    llvm::IntrusiveRefCntPtr<basic::DiagnosticManager> diagnostic_manager;
    std::unique_ptr<SymbolTable> syms;
    FunctionGenerator *current_function_generator = nullptr;
    clang::SourceLocation current_source_loc;

    explicit GeneratorContext(
        IRContext *ir_context,
        llvm::IntrusiveRefCntPtr<basic::DiagnosticManager> diagnostic_manager);

    friend class FunctionGeneratorGuard;

    struct FunctionGeneratorGuard {
        explicit FunctionGeneratorGuard(GeneratorContext *parent,
                                        FunctionGenerator *func_gen)
            : parent_(parent), previous_(parent->current_function_generator) {
            parent_->current_function_generator = func_gen;
        }
        ~FunctionGeneratorGuard() {
            if (parent_) {
                parent_->current_function_generator = previous_;
            }
        }

      private:
        FunctionGeneratorGuard(const FunctionGeneratorGuard &) = delete;
        FunctionGeneratorGuard &
        operator=(const FunctionGeneratorGuard &) = delete;

        GeneratorContext *parent_;
        FunctionGenerator *previous_ = nullptr;
    };

    struct SymbolTableGuard {
        SymbolTableGuard() = default;
        SymbolTableGuard(GeneratorContext *parent,
                         std::unique_ptr<SymbolTable> replacement)
            : parent_(parent), saved_(std::move(parent->syms)) {
            parent_->syms = std::move(replacement);
        }
        ~SymbolTableGuard() {
            if (parent_) {
                parent_->syms = std::move(saved_);
            }
        }

        SymbolTableGuard(SymbolTableGuard &&other) noexcept
            : parent_(other.parent_), saved_(std::move(other.saved_)) {
            other.parent_ = nullptr;
        }
        SymbolTableGuard &operator=(SymbolTableGuard &&other) noexcept {
            if (this != &other) {
                if (parent_) {
                    parent_->syms = std::move(saved_);
                }
                parent_ = other.parent_;
                saved_ = std::move(other.saved_);
                other.parent_ = nullptr;
            }
            return *this;
        }

      private:
        SymbolTableGuard(const SymbolTableGuard &) = delete;
        SymbolTableGuard &operator=(const SymbolTableGuard &) = delete;

        GeneratorContext *parent_ = nullptr;
        std::unique_ptr<SymbolTable> saved_;
    };

    FunctionGeneratorGuard
    GetFunctionGeneratorGuard(FunctionGenerator *func_gen) {
        return FunctionGeneratorGuard(this, func_gen);
    }

    SymbolTableGuard
    GetSymbolTableGuard(std::unique_ptr<SymbolTable> replacement) {
        return SymbolTableGuard(this, std::move(replacement));
    }

    FunctionGenerator *GetCurrentFunctionGenerator() const {
        return current_function_generator;
    }

    mlir::Location GetMLIRLocation(mlir::MLIRContext *mlir_context,
                                   clang::SourceLocation loc) const {
        if (!loc.isValid() || !diagnostic_manager) {
            return mlir::UnknownLoc::get(mlir_context);
        }

        auto &source_manager = diagnostic_manager->GetSourceManager();
        clang::SourceLocation file_loc = source_manager.getFileLoc(loc);
        if (file_loc.isInvalid()) {
            return mlir::UnknownLoc::get(mlir_context);
        }

        auto filename = source_manager.getFilename(file_loc);
        if (filename.empty()) {
            filename = source_manager.getBufferName(file_loc);
        }
        unsigned line = source_manager.getSpellingLineNumber(file_loc);
        unsigned column = source_manager.getSpellingColumnNumber(file_loc);
        if (filename.empty() || line == 0 || column == 0) {
            return mlir::UnknownLoc::get(mlir_context);
        }

        return mlir::FileLineColLoc::get(mlir_context, filename, line, column);
    }

    mlir::Location GetMLIRLocation(mlir::MLIRContext *mlir_context,
                                   const ast::ASTNode *node) const {
        return GetMLIRLocation(mlir_context,
                               node ? node->GetSourceRange().getBegin()
                                    : clang::SourceLocation());
    }

    ExprGenerator *GetExprGenerator() const;
};

} // namespace causalflow::avelang::ir
