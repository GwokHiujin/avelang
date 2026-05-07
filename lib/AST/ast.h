#pragma once

#include <clang/Basic/SourceLocation.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>

#include <string>
#include <vector>

namespace causalflow::avelang::ast {

using SourceLocation = clang::SourceLocation;
using SourceRange = clang::SourceRange;

// Node kinds for LLVM-style casting
enum class NodeKind {
    // Base kinds
    ASTNodeKind,
    ExprKind,
    StmtKind,
    ExprContextKind,

    // Specific node kinds
    ModuleKind,
    ArgKind,
    ArgumentsKind,

    // Expression kinds
    AttributeExprKind,
    NameKind,
    BinOpKind,
    UnaryOpKind,
    BoolOpKind,
    LambdaKind,
    IfExpKind,
    DictKind,
    CompareKind,
    CallKind,
    ConstantKind,
    SubscriptKind,
    ListKind,
    TupleKind,
    SliceKind,

    // Statement kinds
    FunctionDefKind,
    ReturnKind,
    AssignKind,
    AugAssignKind,
    AnnAssignKind,
    ForKind,
    WhileKind,
    IfKind,
    WithKind,
    PassKind,
    BreakKind,
    ContinueKind,
    ExprStmtKind,
    ImportKind,
    ImportFromKind,
};

class ASTNode {
  public:
    using Children = llvm::SmallVector<ASTNode *, 4>;

    virtual ~ASTNode() = default;

    const SourceRange &GetSourceRange() const { return source_range_; }
    void SetSourceRange(const SourceRange &range) { source_range_ = range; }

    // Support for LLVM-style casting
    NodeKind getKind() const { return kind_; }

  protected:
    explicit ASTNode(NodeKind kind) : kind_(kind) {}

  private:
    const NodeKind kind_;
    SourceRange source_range_;
};

class Expr;

// Base classes for different AST node categories
class ExprContext : public ASTNode {
  public:
    ExprContext() : ASTNode(NodeKind::ExprContextKind) {}
};

// Function argument node
class Arg : public ASTNode {
  public:
    Arg(std::string arg_name, Expr *annotation = nullptr)
        : ASTNode(NodeKind::ArgKind), arg_name_(std::move(arg_name)),
          annotation_(annotation) {}

    const std::string &GetArgName() const { return arg_name_; }
    Expr *GetAnnotation() const { return annotation_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::ArgKind;
    }

  private:
    std::string arg_name_;
    Expr *annotation_;
};

// Function arguments container
class Arguments : public ASTNode {
  public:
    Arguments(llvm::SmallVector<Arg *, 4> posonlyargs = {},
              llvm::SmallVector<Arg *, 4> args = {},
              llvm::SmallVector<Arg *, 4> kwonlyargs = {},
              llvm::SmallVector<Expr *, 4> defaults = {},
              llvm::SmallVector<Expr *, 4> kw_defaults = {},
              Arg *vararg = nullptr, Arg *kwarg = nullptr)
        : ASTNode(NodeKind::ArgumentsKind),
          posonlyargs_(std::move(posonlyargs)), args_(std::move(args)),
          kwonlyargs_(std::move(kwonlyargs)), defaults_(std::move(defaults)),
          kw_defaults_(std::move(kw_defaults)), vararg_(vararg), kwarg_(kwarg) {
    }

    const llvm::SmallVector<Arg *, 4> &GetArgs() const { return args_; }
    const llvm::SmallVector<Arg *, 4> &GetPosOnlyArgs() const {
        return posonlyargs_;
    }
    const llvm::SmallVector<Arg *, 4> &GetKwOnlyArgs() const {
        return kwonlyargs_;
    }
    const llvm::SmallVector<Expr *, 4> &GetDefaults() const {
        return defaults_;
    }
    const llvm::SmallVector<Expr *, 4> &GetKwDefaults() const {
        return kw_defaults_;
    }

    Arg *GetVarArg() const { return vararg_; }
    Arg *GetKwArg() const { return kwarg_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::ArgumentsKind;
    }

  private:
    llvm::SmallVector<Arg *, 4> posonlyargs_;  // positional-only arguments
    llvm::SmallVector<Arg *, 4> args_;         // regular arguments
    llvm::SmallVector<Arg *, 4> kwonlyargs_;   // keyword-only arguments
    llvm::SmallVector<Expr *, 4> defaults_;    // default values for args
    llvm::SmallVector<Expr *, 4> kw_defaults_; // default values for kwonlyargs
    Arg *vararg_ = nullptr;                    // *args parameter
    Arg *kwarg_ = nullptr;                     // **kwargs parameter
};

// Forward declaration
class Stmt;

// Module node
class Module : public ASTNode {
  public:
    Module(llvm::SmallVector<Stmt *, 4> body = {})
        : ASTNode(NodeKind::ModuleKind), body_(std::move(body)) {}

    const llvm::SmallVector<Stmt *, 4> &GetBody() const { return body_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::ModuleKind;
    }

  private:
    llvm::SmallVector<Stmt *, 4> body_;
};

} // namespace causalflow::avelang::ast

#include "ast_nodes_expr.h"
#include "ast_nodes_stmt.h"