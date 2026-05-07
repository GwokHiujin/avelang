#pragma once

#include "ast.h"
#include "ast_nodes_expr.h"
#include "ast_nodes_stmt.h"
#include <llvm/Support/Casting.h>

// LLVM casting specializations for AST nodes
namespace llvm {

// Module casting
template <>
struct isa_impl<causalflow::avelang::ast::Module,
                causalflow::avelang::ast::ASTNode> {
    static inline bool doit(const causalflow::avelang::ast::ASTNode &node) {
        return causalflow::avelang::ast::Module::classof(&node);
    }
};

// Expression casting
template <>
struct isa_impl<causalflow::avelang::ast::Expr,
                causalflow::avelang::ast::ASTNode> {
    static inline bool doit(const causalflow::avelang::ast::ASTNode &node) {
        return causalflow::avelang::ast::Expr::classof(&node);
    }
};

// Statement casting
template <>
struct isa_impl<causalflow::avelang::ast::Stmt,
                causalflow::avelang::ast::ASTNode> {
    static inline bool doit(const causalflow::avelang::ast::ASTNode &node) {
        return causalflow::avelang::ast::Stmt::classof(&node);
    }
};

// Specific expression node casting
template <>
struct isa_impl<causalflow::avelang::ast::AttributeExpr,
                causalflow::avelang::ast::ASTNode> {
    static inline bool doit(const causalflow::avelang::ast::ASTNode &node) {
        return causalflow::avelang::ast::AttributeExpr::classof(&node);
    }
};

template <>
struct isa_impl<causalflow::avelang::ast::Name,
                causalflow::avelang::ast::ASTNode> {
    static inline bool doit(const causalflow::avelang::ast::ASTNode &node) {
        return causalflow::avelang::ast::Name::classof(&node);
    }
};

// Specific statement node casting
template <>
struct isa_impl<causalflow::avelang::ast::FunctionDef,
                causalflow::avelang::ast::ASTNode> {
    static inline bool doit(const causalflow::avelang::ast::ASTNode &node) {
        return causalflow::avelang::ast::FunctionDef::classof(&node);
    }
};

template <>
struct isa_impl<causalflow::avelang::ast::Return,
                causalflow::avelang::ast::ASTNode> {
    static inline bool doit(const causalflow::avelang::ast::ASTNode &node) {
        return causalflow::avelang::ast::Return::classof(&node);
    }
};

// Argument nodes casting
template <>
struct isa_impl<causalflow::avelang::ast::Arg,
                causalflow::avelang::ast::ASTNode> {
    static inline bool doit(const causalflow::avelang::ast::ASTNode &node) {
        return causalflow::avelang::ast::Arg::classof(&node);
    }
};

template <>
struct isa_impl<causalflow::avelang::ast::Arguments,
                causalflow::avelang::ast::ASTNode> {
    static inline bool doit(const causalflow::avelang::ast::ASTNode &node) {
        return causalflow::avelang::ast::Arguments::classof(&node);
    }
};

} // namespace llvm