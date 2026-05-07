#pragma once

#include "ast.h"

namespace causalflow::avelang::ast {

// Forward declarations
class Module;
class ASTNode;
class Stmt;
class Expr;
class FunctionDef;
class Return;
class Assign;
class AugAssign;
class AnnAssign;
class For;
class While;
class If;
class With;
class Pass;
class Break;
class Continue;
class ExprStmt;

// Expression forward declarations
class AttributeExpr;
class Name;
class BinOp;
class UnaryOp;
class BoolOp;
class Lambda;
class IfExp;
class Dict;
class Compare;
class Call;
class Constant;
class Subscript;
class List;
class Tuple;
class Slice;

/**
 * Template-based AST visitor for traversing AST nodes.
 *
 * This visitor uses the Curiously Recurring Template Pattern (CRTP)
 * to avoid virtual function call overhead while still allowing
 * derived classes to customize behavior.
 *
 * Usage:
 *   class MyVisitor : public ASTVisitor<MyVisitor> {
 *   public:
 *     void VisitASTNode(ASTNode *node) {
 *       // Custom implementation
 *     }
 *     void VisitStmt(Stmt *stmt) {
 *       // Custom statement handling
 *     }
 *   };
 */
template <typename Derived, typename R = void> class ASTVisitor {
  public:
    // Default implementation for visiting any AST node
    R Dispatch(ASTNode *node);
    R DispatchStmt(Stmt *stmt);
    R DispatchExpr(Expr *expr);

    // Module visitor methods
    R VisitModule(Module *module) { return R(); }

    // Individual statement visitor methods
    R VisitFunctionDef(FunctionDef *func) { return R(); }
    R VisitReturn(Return *ret) { return R(); }
    R VisitAssign(Assign *assign) { return R(); }
    R VisitAugAssign(AugAssign *aug_assign) { return R(); }
    R VisitAnnAssign(AnnAssign *ann_assign) { return R(); }
    R VisitFor(For *for_loop) { return R(); }
    R VisitWhile(While *while_loop) { return R(); }
    R VisitIf(If *if_stmt) { return R(); }
    R VisitWith(With *with_stmt) { return R(); }
    R VisitPass(Pass *pass) { return R(); }
    R VisitBreak(Break *break_stmt) { return R(); }
    R VisitContinue(Continue *continue_stmt) { return R(); }
    R VisitExprStmt(ExprStmt *expr_stmt) { return R(); }

    // Individual expression visitor methods
    R VisitAttributeExpr(AttributeExpr *attr) { return R(); }
    R VisitName(Name *name) { return R(); }
    R VisitBinOp(BinOp *binop) { return R(); }
    R VisitUnaryOp(UnaryOp *unaryop) { return R(); }
    R VisitBoolOp(BoolOp *boolop) { return R(); }
    R VisitLambda(Lambda *lambda) { return R(); }
    R VisitIfExp(IfExp *ifexp) { return R(); }
    R VisitDict(Dict *dict) { return R(); }
    R VisitCompare(Compare *compare) { return R(); }
    R VisitCall(Call *call) { return R(); }
    R VisitConstant(Constant *constant) { return R(); }
    R VisitSubscript(Subscript *subscript) { return R(); }
    R VisitList(List *list) { return R(); }
    R VisitTuple(Tuple *tuple) { return R(); }
    R VisitSlice(Slice *slice) { return R(); }
};

} // namespace causalflow::avelang::ast

// Include implementation
#include "visitor.inl"