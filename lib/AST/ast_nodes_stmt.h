#pragma once

#include "ast.h"

#include <clang/Basic/SourceLocation.h>
#include <llvm/ADT/SmallVector.h>

#include <string>
#include <vector>

namespace causalflow::avelang::ast {

// Forward declarations
class Expr;
class Arguments;

class Stmt : public ASTNode {
  public:
    // LLVM-style casting support for all statements
    static bool classof(const ASTNode *node) {
        return node->getKind() >= NodeKind::FunctionDefKind &&
               node->getKind() <= NodeKind::ImportFromKind;
    }

  protected:
    explicit Stmt(NodeKind kind) : ASTNode(kind) {}
};

// Statement nodes
class FunctionDef : public Stmt {
  public:
    FunctionDef(std::string name, Arguments *args,
                llvm::SmallVector<Stmt *, 4> body,
                llvm::SmallVector<Expr *, 4> decorators = {},
                Expr *returns = nullptr)
        : Stmt(NodeKind::FunctionDefKind), name_(std::move(name)), args_(args),
          body_(std::move(body)), decorators_(std::move(decorators)),
          returns_(returns) {}

    const std::string &GetName() const { return name_; }
    Arguments *GetArguments() const { return args_; }
    const llvm::SmallVector<Stmt *, 4> &GetBody() const { return body_; }
    const llvm::SmallVector<Expr *, 4> &GetDecorators() const {
        return decorators_;
    }
    Expr *GetReturns() const { return returns_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::FunctionDefKind;
    }

  private:
    std::string name_;
    Arguments *args_;
    llvm::SmallVector<Stmt *, 4> body_;
    llvm::SmallVector<Expr *, 4> decorators_;
    Expr *returns_;
};

class Return : public Stmt {
  public:
    Return(Expr *value = nullptr) : Stmt(NodeKind::ReturnKind), value_(value) {}

    Expr *GetValue() const { return value_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::ReturnKind;
    }

  private:
    Expr *value_;
};

class Assign : public Stmt {
  public:
    Assign(llvm::SmallVector<Expr *, 4> targets, Expr *value)
        : Stmt(NodeKind::AssignKind), targets_(std::move(targets)),
          value_(value) {}

    const llvm::SmallVector<Expr *, 4> &GetTargets() const { return targets_; }
    Expr *GetValue() const { return value_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::AssignKind;
    }

  private:
    llvm::SmallVector<Expr *, 4> targets_;
    Expr *value_;
};

class AugAssign : public Stmt {
  public:
    AugAssign(Expr *target, std::string op, Expr *value)
        : Stmt(NodeKind::AugAssignKind), target_(target), op_(std::move(op)),
          value_(value) {}

    Expr *GetTarget() const { return target_; }
    const std::string &GetOp() const { return op_; }
    Expr *GetValue() const { return value_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::AugAssignKind;
    }

  private:
    Expr *target_;
    std::string op_;
    Expr *value_;
};

class AnnAssign : public Stmt {
  public:
    AnnAssign(Expr *target, Expr *annotation, Expr *value = nullptr,
              bool simple = true)
        : Stmt(NodeKind::AnnAssignKind), target_(target),
          annotation_(annotation), value_(value), simple_(simple) {}

    Expr *GetTarget() const { return target_; }
    Expr *GetAnnotation() const { return annotation_; }
    Expr *GetValue() const { return value_; }
    bool GetSimple() const { return simple_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::AnnAssignKind;
    }

  private:
    Expr *target_;
    Expr *annotation_;
    Expr *value_;
    bool simple_;
};

class For : public Stmt {
  public:
    For(Expr *target, Expr *iter, llvm::SmallVector<Stmt *, 4> body,
        llvm::SmallVector<Stmt *, 4> orelse = {})
        : Stmt(NodeKind::ForKind), target_(target), iter_(iter),
          body_(std::move(body)), orelse_(std::move(orelse)) {}

    Expr *GetTarget() const { return target_; }
    Expr *GetIter() const { return iter_; }
    const llvm::SmallVector<Stmt *, 4> &GetBody() const { return body_; }
    const llvm::SmallVector<Stmt *, 4> &GetOrelse() const { return orelse_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::ForKind;
    }

  private:
    Expr *target_;
    Expr *iter_;
    llvm::SmallVector<Stmt *, 4> body_;
    llvm::SmallVector<Stmt *, 4> orelse_;
};

class While : public Stmt {
  public:
    While(Expr *test, llvm::SmallVector<Stmt *, 4> body,
          llvm::SmallVector<Stmt *, 4> orelse = {})
        : Stmt(NodeKind::WhileKind), test_(test), body_(std::move(body)),
          orelse_(std::move(orelse)) {}

    Expr *GetTest() const { return test_; }
    const llvm::SmallVector<Stmt *, 4> &GetBody() const { return body_; }
    const llvm::SmallVector<Stmt *, 4> &GetOrelse() const { return orelse_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::WhileKind;
    }

  private:
    Expr *test_;
    llvm::SmallVector<Stmt *, 4> body_;
    llvm::SmallVector<Stmt *, 4> orelse_;
};

class If : public Stmt {
  public:
    If(Expr *test, llvm::SmallVector<Stmt *, 4> body,
       llvm::SmallVector<Stmt *, 4> orelse = {})
        : Stmt(NodeKind::IfKind), test_(test), body_(std::move(body)),
          orelse_(std::move(orelse)) {}

    Expr *GetTest() const { return test_; }
    const llvm::SmallVector<Stmt *, 4> &GetBody() const { return body_; }
    const llvm::SmallVector<Stmt *, 4> &GetOrelse() const { return orelse_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::IfKind;
    }

  private:
    Expr *test_;
    llvm::SmallVector<Stmt *, 4> body_;
    llvm::SmallVector<Stmt *, 4> orelse_;
};

class With : public Stmt {
  public:
    struct WithItem {
        Expr *context_expr;
        Expr *optional_vars;

        WithItem(Expr *ctx, Expr *vars = nullptr)
            : context_expr(ctx), optional_vars(vars) {}
    };

    With(llvm::SmallVector<WithItem, 4> items,
         llvm::SmallVector<Stmt *, 4> body)
        : Stmt(NodeKind::WithKind), items_(std::move(items)),
          body_(std::move(body)) {}

    const llvm::SmallVector<WithItem, 4> &GetItems() const { return items_; }
    const llvm::SmallVector<Stmt *, 4> &GetBody() const { return body_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::WithKind;
    }

  private:
    llvm::SmallVector<WithItem, 4> items_;
    llvm::SmallVector<Stmt *, 4> body_;
};

class Pass : public Stmt {
  public:
    Pass() : Stmt(NodeKind::PassKind) {}

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::PassKind;
    }
};

class Break : public Stmt {
  public:
    Break() : Stmt(NodeKind::BreakKind) {}

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::BreakKind;
    }
};

class Continue : public Stmt {
  public:
    Continue() : Stmt(NodeKind::ContinueKind) {}

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::ContinueKind;
    }
};

class ExprStmt : public Stmt {
  public:
    ExprStmt(Expr *value) : Stmt(NodeKind::ExprStmtKind), value_(value) {}

    Expr *GetValue() const { return value_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::ExprStmtKind;
    }

  private:
    Expr *value_;
};

// Alias represents a single import alias (e.g., "foo as bar")
struct Alias {
    std::string name;   // The original name
    std::string asname; // The alias name (empty if no alias)

    Alias(std::string name, std::string asname = "")
        : name(std::move(name)), asname(std::move(asname)) {}
};

class Import : public Stmt {
  public:
    Import(llvm::SmallVector<Alias, 4> names)
        : Stmt(NodeKind::ImportKind), names_(std::move(names)) {}

    const llvm::SmallVector<Alias, 4> &GetNames() const { return names_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::ImportKind;
    }

  private:
    llvm::SmallVector<Alias, 4> names_;
};

class ImportFrom : public Stmt {
  public:
    ImportFrom(std::string module, llvm::SmallVector<Alias, 4> names,
               int level = 0)
        : Stmt(NodeKind::ImportFromKind), module_(std::move(module)),
          names_(std::move(names)), level_(level) {}

    const std::string &GetModule() const { return module_; }
    const llvm::SmallVector<Alias, 4> &GetNames() const { return names_; }
    int GetLevel() const { return level_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::ImportFromKind;
    }

  private:
    std::string module_;
    llvm::SmallVector<Alias, 4> names_;
    int level_; // For relative imports (e.g., "from ..module import foo")
};

} // namespace causalflow::avelang::ast