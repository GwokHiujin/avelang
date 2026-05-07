#pragma once

#include "AST/ast.h"
#include "AST/ast_context.h"
#include "Basic/diagnostic.h"

#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/FileManager.h>
#include <clang/Basic/SourceManager.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <memory>
#include <string>

namespace pybind11 {
class object;
}

namespace llvm {
class MemoryBuffer;
}

namespace causalflow::avelang::frontend {

class AveLangParser {
  public:
    explicit AveLangParser(
        llvm::IntrusiveRefCntPtr<ast::ASTContext> context,
        llvm::IntrusiveRefCntPtr<basic::DiagnosticManager> diagnostic_manager);
    void Parse(const std::string &file_name);
    void ParseFromBuffer(const llvm::MemoryBuffer &buffer,
                         llvm::StringRef buffer_name);
    ast::ASTNode *ParsePythonAST(const pybind11::object &py_ast_node,
                                 llvm::StringRef file_name = "<input>");
    ast::FunctionDef *
    ParsePythonFunctionDef(const pybind11::object &py_function_def,
                           llvm::StringRef file_name = "<input>");
    std::unique_ptr<llvm::MemoryBuffer> ReleaseOwnedBuffer();

    // VISIBLE FOR TESTING
    void ParseImpl(const clang::FileID main_file_id,
                   const std::string &file_name,
                   const llvm::StringRef &source_code);

    ast::ASTNode *GetModule() const { return root_node_; }

  private:
    // AST node parsing dispatcher
    ast::ASTNode *ParseRoot(const pybind11::object &py_ast_node);

    // Specific parsing functions for each Python AST node type
    ast::Module *ParseModule(const pybind11::object &py_module);
    ast::FunctionDef *ParseFunctionDef(const pybind11::object &py_function_def);
    ast::Arguments *ParseArguments(const pybind11::object &py_arguments);
    ast::AttributeExpr *
    ParseAttributeExpr(const pybind11::object &py_attribute);
    ast::Arg *ParseArg(const pybind11::object &py_arg);
    ast::Name *ParseName(const pybind11::object &py_name);
    ast::Return *ParseReturn(const pybind11::object &py_return);
    ast::BinOp *ParseBinOp(const pybind11::object &py_binop);
    ast::UnaryOp *ParseUnaryOp(const pybind11::object &py_unaryop);
    ast::BoolOp *ParseBoolOp(const pybind11::object &py_boolop);
    ast::Lambda *ParseLambda(const pybind11::object &py_lambda);
    ast::IfExp *ParseIfExp(const pybind11::object &py_ifexp);
    ast::Dict *ParseDict(const pybind11::object &py_dict);
    ast::Compare *ParseCompare(const pybind11::object &py_compare);
    ast::Call *ParseCall(const pybind11::object &py_call);
    ast::Constant *ParseConstant(const pybind11::object &py_constant);
    ast::Subscript *ParseSubscript(const pybind11::object &py_subscript);
    ast::List *ParseList(const pybind11::object &py_list);
    ast::Tuple *ParseTuple(const pybind11::object &py_tuple);
    ast::Slice *ParseSlice(const pybind11::object &py_slice);

    // Additional statement parsing functions
    ast::Assign *ParseAssign(const pybind11::object &py_assign);
    ast::AugAssign *ParseAugAssign(const pybind11::object &py_augassign);
    ast::AnnAssign *ParseAnnAssign(const pybind11::object &py_annassign);
    ast::For *ParseFor(const pybind11::object &py_for);
    ast::While *ParseWhile(const pybind11::object &py_while);
    ast::If *ParseIf(const pybind11::object &py_if);
    ast::With *ParseWith(const pybind11::object &py_with);
    ast::Pass *ParsePass(const pybind11::object &py_pass);
    ast::Break *ParseBreak(const pybind11::object &py_break);
    ast::Continue *ParseContinue(const pybind11::object &py_continue);
    ast::ExprStmt *ParseExprStmt(const pybind11::object &py_expr);
    ast::Import *ParseImport(const pybind11::object &py_import);
    ast::ImportFrom *ParseImportFrom(const pybind11::object &py_import_from);

    // Expression and statement parsing functions
    ast::Expr *ParseExpr(const pybind11::object &py_expr);
    ast::Stmt *ParseStmt(const pybind11::object &py_stmt);

    // Utility functions
    clang::SourceRange ExtractSourceRange(const pybind11::object &py_ast_node);

    clang::SourceRange GetCurrentSourceRange() const;

    clang::DiagnosticBuilder Report(basic::DiagnosticCode error_code);

    llvm::IntrusiveRefCntPtr<ast::ASTContext> context_;
    llvm::IntrusiveRefCntPtr<basic::DiagnosticManager> diagnostic_manager_;

    ast::ASTNode *root_node_ = nullptr;
    clang::FileID current_file_id_;

    class ParsingContextGuard {
      public:
        ParsingContextGuard(AveLangParser *parser,
                            const clang::SourceRange &range);
        ~ParsingContextGuard();

        // Delete copy constructor and assignment operator
        ParsingContextGuard(const ParsingContextGuard &) = delete;
        ParsingContextGuard &operator=(const ParsingContextGuard &) = delete;

      private:
        AveLangParser *parent_;
    };

    friend class ParsingContextGuard;
    std::vector<clang::SourceRange> source_range_stack_;
    std::unique_ptr<llvm::MemoryBuffer> owned_buffer_;
};

} // namespace causalflow::avelang::frontend
