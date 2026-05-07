#pragma once

#include "ast.h"
#include <clang/Basic/SourceLocation.h>
#include <llvm/ADT/SmallVector.h>
#include <string>
#include <vector>

namespace causalflow::avelang::ast {

class Expr : public ASTNode {
  public:
    // LLVM-style casting support for all expressions
    static bool classof(const ASTNode *node) {
        return node->getKind() >= NodeKind::AttributeExprKind &&
               node->getKind() <= NodeKind::SliceKind;
    }

  protected:
    explicit Expr(NodeKind kind) : ASTNode(kind) {}
};

// Forward declarations
class ASTNode;
class Expr;
class ExprContext;
class Arguments;

// Expression nodes
class AttributeExpr : public Expr {
  public:
    AttributeExpr(Expr *value, std::string attr)
        : Expr(NodeKind::AttributeExprKind), value_(value),
          attr_(std::move(attr)) {}

    Expr *GetValue() const { return value_; }
    const std::string &GetAttr() const { return attr_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::AttributeExprKind;
    }

  private:
    Expr *value_;
    std::string attr_;
};

class Name : public Expr {
  public:
    Name(std::string id, ExprContext *ctx = nullptr)
        : Expr(NodeKind::NameKind), id_(std::move(id)), ctx_(ctx) {}

    const std::string &GetId() const { return id_; }
    ExprContext *GetCtx() const { return ctx_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::NameKind;
    }

  private:
    std::string id_;
    ExprContext *ctx_;
};

class BinOp : public Expr {
  public:
    BinOp(Expr *left, Expr *right, std::string op)
        : Expr(NodeKind::BinOpKind), left_(left), right_(right),
          op_(std::move(op)) {}

    Expr *GetLeft() const { return left_; }
    Expr *GetRight() const { return right_; }
    const std::string &GetOp() const { return op_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::BinOpKind;
    }

  private:
    Expr *left_;
    Expr *right_;
    std::string op_;
};

class UnaryOp : public Expr {
  public:
    UnaryOp(Expr *operand, std::string op)
        : Expr(NodeKind::UnaryOpKind), operand_(operand), op_(std::move(op)) {}

    Expr *GetOperand() const { return operand_; }
    const std::string &GetOp() const { return op_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::UnaryOpKind;
    }

  private:
    Expr *operand_;
    std::string op_;
};

class BoolOp : public Expr {
  public:
    BoolOp(llvm::SmallVector<Expr *, 4> values, std::string op)
        : Expr(NodeKind::BoolOpKind), values_(std::move(values)),
          op_(std::move(op)) {}

    const llvm::SmallVector<Expr *, 4> &GetValues() const { return values_; }
    const std::string &GetOp() const { return op_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::BoolOpKind;
    }

  private:
    llvm::SmallVector<Expr *, 4> values_;
    std::string op_;
};

class Lambda : public Expr {
  public:
    Lambda(ast::Arguments *args, Expr *body)
        : Expr(NodeKind::LambdaKind), args_(args), body_(body) {}

    ast::Arguments *GetArgs() const { return args_; }
    Expr *GetBody() const { return body_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::LambdaKind;
    }

  private:
    ast::Arguments *args_;
    Expr *body_;
};

class IfExp : public Expr {
  public:
    IfExp(Expr *test, Expr *body, Expr *orelse)
        : Expr(NodeKind::IfExpKind), test_(test), body_(body), orelse_(orelse) {
    }

    Expr *GetTest() const { return test_; }
    Expr *GetBody() const { return body_; }
    Expr *GetOrelse() const { return orelse_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::IfExpKind;
    }

  private:
    Expr *test_;
    Expr *body_;
    Expr *orelse_;
};

class Dict : public Expr {
  public:
    Dict(llvm::SmallVector<Expr *, 4> keys, llvm::SmallVector<Expr *, 4> values)
        : Expr(NodeKind::DictKind), keys_(std::move(keys)),
          values_(std::move(values)) {}

    const llvm::SmallVector<Expr *, 4> &GetKeys() const { return keys_; }
    const llvm::SmallVector<Expr *, 4> &GetValues() const { return values_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::DictKind;
    }

  private:
    llvm::SmallVector<Expr *, 4> keys_;
    llvm::SmallVector<Expr *, 4> values_;
};

class Compare : public Expr {
  public:
    Compare(Expr *left, llvm::SmallVector<std::string, 4> ops,
            llvm::SmallVector<Expr *, 4> comparators)
        : Expr(NodeKind::CompareKind), left_(left), ops_(std::move(ops)),
          comparators_(std::move(comparators)) {}

    Expr *GetLeft() const { return left_; }
    const llvm::SmallVector<std::string, 4> &GetOps() const { return ops_; }
    const llvm::SmallVector<Expr *, 4> &GetComparators() const {
        return comparators_;
    }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::CompareKind;
    }

  private:
    Expr *left_;
    llvm::SmallVector<std::string, 4> ops_;
    llvm::SmallVector<Expr *, 4> comparators_;
};

class Call : public Expr {
  public:
    Call(Expr *func, llvm::SmallVector<Expr *, 4> args,
         llvm::SmallVector<std::string, 4> keywords = {})
        : Expr(NodeKind::CallKind), func_(func), args_(std::move(args)),
          keywords_(std::move(keywords)) {}

    Expr *GetFunc() const { return func_; }
    const llvm::SmallVector<Expr *, 4> &GetArgs() const { return args_; }
    const llvm::SmallVector<std::string, 4> &GetKeywords() const {
        return keywords_;
    }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::CallKind;
    }

  private:
    Expr *func_;
    llvm::SmallVector<Expr *, 4> args_;
    llvm::SmallVector<std::string, 4> keywords_;
};

class Constant : public Expr {
  public:
    Constant(std::string value)
        : Expr(NodeKind::ConstantKind), value_(std::move(value)) {}

    const std::string &GetValue() const { return value_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::ConstantKind;
    }

  private:
    std::string value_;
};

class Subscript : public Expr {
  public:
    Subscript(Expr *value, Expr *slice, ExprContext *ctx = nullptr)
        : Expr(NodeKind::SubscriptKind), value_(value), slice_(slice),
          ctx_(ctx) {}

    Expr *GetValue() const { return value_; }
    Expr *GetSlice() const { return slice_; }
    ExprContext *GetCtx() const { return ctx_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::SubscriptKind;
    }

  private:
    Expr *value_;
    Expr *slice_;
    ExprContext *ctx_;
};

class List : public Expr {
  public:
    List(llvm::SmallVector<Expr *, 4> elts, ExprContext *ctx = nullptr)
        : Expr(NodeKind::ListKind), elts_(std::move(elts)), ctx_(ctx) {}

    const llvm::SmallVector<Expr *, 4> &GetElts() const { return elts_; }
    ExprContext *GetCtx() const { return ctx_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::ListKind;
    }

  private:
    llvm::SmallVector<Expr *, 4> elts_;
    ExprContext *ctx_;
};

class Tuple : public Expr {
  public:
    Tuple(llvm::SmallVector<Expr *, 4> elts, ExprContext *ctx = nullptr)
        : Expr(NodeKind::TupleKind), elts_(std::move(elts)), ctx_(ctx) {}

    const llvm::SmallVector<Expr *, 4> &GetElts() const { return elts_; }
    ExprContext *GetCtx() const { return ctx_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::TupleKind;
    }

  private:
    llvm::SmallVector<Expr *, 4> elts_;
    ExprContext *ctx_;
};

class Slice : public Expr {
  public:
    Slice(Expr *lower = nullptr, Expr *upper = nullptr, Expr *step = nullptr)
        : Expr(NodeKind::SliceKind), lower_(lower), upper_(upper), step_(step) {
    }

    Expr *GetLower() const { return lower_; }
    Expr *GetUpper() const { return upper_; }
    Expr *GetStep() const { return step_; }

    // LLVM-style casting support
    static bool classof(const ASTNode *node) {
        return node->getKind() == NodeKind::SliceKind;
    }

  private:
    Expr *lower_;
    Expr *upper_;
    Expr *step_;
};

} // namespace causalflow::avelang::ast
