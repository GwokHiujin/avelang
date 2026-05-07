#pragma once

#include "ast.h"
#include "ast_casting.h"
#include "visitor.h"

namespace causalflow::avelang::ast {

using llvm::cast;

template <typename Derived, typename R> R ASTVisitor<Derived, R>::Dispatch(ASTNode *node) {
    if (llvm::isa<Module>(node)) {
        return static_cast<Derived *>(this)->VisitModule(cast<Module>(node));
    } else if (llvm::isa<Stmt>(node)) {
        return DispatchStmt(cast<Stmt>(node));
    } else if (llvm::isa<Expr>(node)) {
        return DispatchExpr(cast<Expr>(node));
    } else {
        __builtin_unreachable();
    }
}

template <typename Derived, typename R> R ASTVisitor<Derived, R>::DispatchExpr(Expr *expr) {
    // Dispatch to specific expression visitor based on type
    if (llvm::isa<AttributeExpr>(expr)) {
        return static_cast<Derived *>(this)->VisitAttributeExpr(cast<AttributeExpr>(expr));
    } else if (llvm::isa<Name>(expr)) {
        return static_cast<Derived *>(this)->VisitName(cast<Name>(expr));
    } else if (llvm::isa<BinOp>(expr)) {
        return static_cast<Derived *>(this)->VisitBinOp(cast<BinOp>(expr));
    } else if (llvm::isa<UnaryOp>(expr)) {
        return static_cast<Derived *>(this)->VisitUnaryOp(cast<UnaryOp>(expr));
    } else if (llvm::isa<BoolOp>(expr)) {
        return static_cast<Derived *>(this)->VisitBoolOp(cast<BoolOp>(expr));
    } else if (llvm::isa<Lambda>(expr)) {
        return static_cast<Derived *>(this)->VisitLambda(cast<Lambda>(expr));
    } else if (llvm::isa<IfExp>(expr)) {
        return static_cast<Derived *>(this)->VisitIfExp(cast<IfExp>(expr));
    } else if (llvm::isa<Dict>(expr)) {
        return static_cast<Derived *>(this)->VisitDict(cast<Dict>(expr));
    } else if (llvm::isa<Compare>(expr)) {
        return static_cast<Derived *>(this)->VisitCompare(cast<Compare>(expr));
    } else if (llvm::isa<Call>(expr)) {
        return static_cast<Derived *>(this)->VisitCall(cast<Call>(expr));
    } else if (llvm::isa<Constant>(expr)) {
        return static_cast<Derived *>(this)->VisitConstant(cast<Constant>(expr));
    } else if (llvm::isa<Subscript>(expr)) {
        return static_cast<Derived *>(this)->VisitSubscript(cast<Subscript>(expr));
    } else if (llvm::isa<List>(expr)) {
        return static_cast<Derived *>(this)->VisitList(cast<List>(expr));
    } else if (llvm::isa<Tuple>(expr)) {
        return static_cast<Derived *>(this)->VisitTuple(cast<Tuple>(expr));
    } else if (llvm::isa<Slice>(expr)) {
        return static_cast<Derived *>(this)->VisitSlice(cast<Slice>(expr));
    } else {
        __builtin_unreachable();
    }
}

template <typename Derived, typename R> R ASTVisitor<Derived, R>::DispatchStmt(Stmt *stmt) {
    // Dispatch to specific statement visitor based on type
    if (llvm::isa<FunctionDef>(stmt)) {
        return static_cast<Derived *>(this)->VisitFunctionDef(cast<FunctionDef>(stmt));
    } else if (llvm::isa<Return>(stmt)) {
        return static_cast<Derived *>(this)->VisitReturn(cast<Return>(stmt));
    } else if (llvm::isa<Assign>(stmt)) {
        return static_cast<Derived *>(this)->VisitAssign(cast<Assign>(stmt));
    } else if (llvm::isa<AugAssign>(stmt)) {
        return static_cast<Derived *>(this)->VisitAugAssign(cast<AugAssign>(stmt));
    } else if (llvm::isa<AnnAssign>(stmt)) {
        return static_cast<Derived *>(this)->VisitAnnAssign(cast<AnnAssign>(stmt));
    } else if (llvm::isa<For>(stmt)) {
        return static_cast<Derived *>(this)->VisitFor(cast<For>(stmt));
    } else if (llvm::isa<While>(stmt)) {
        return static_cast<Derived *>(this)->VisitWhile(cast<While>(stmt));
    } else if (llvm::isa<If>(stmt)) {
        return static_cast<Derived *>(this)->VisitIf(cast<If>(stmt));
    } else if (llvm::isa<With>(stmt)) {
        return static_cast<Derived *>(this)->VisitWith(cast<With>(stmt));
    } else if (llvm::isa<Pass>(stmt)) {
        return static_cast<Derived *>(this)->VisitPass(cast<Pass>(stmt));
    } else if (llvm::isa<Break>(stmt)) {
        return static_cast<Derived *>(this)->VisitBreak(cast<Break>(stmt));
    } else if (llvm::isa<Continue>(stmt)) {
        return static_cast<Derived *>(this)->VisitContinue(cast<Continue>(stmt));
    } else if (llvm::isa<ExprStmt>(stmt)) {
        return static_cast<Derived *>(this)->VisitExprStmt(cast<ExprStmt>(stmt));
    } else if (llvm::isa<Import>(stmt)) {
        return static_cast<Derived *>(this)->VisitImport(cast<Import>(stmt));
    } else if (llvm::isa<ImportFrom>(stmt)) {
        return static_cast<Derived *>(this)->VisitImportFrom(cast<ImportFrom>(stmt));
    } else {
        __builtin_unreachable();
    }
}


} // namespace causalflow::avelang::ast