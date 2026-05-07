#pragma once

#include "AST/ast_nodes_expr.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wambiguous-reversed-operator"
#include <mlir/IR/Value.h>
#pragma clang diagnostic pop

#include <optional>

namespace causalflow::avelang::ir {

struct TypeInfo {
    std::optional<bool> is_unsigned_integer;
};

bool IsIntegerValueOrVectorOfIntegers(mlir::Type type);
bool IsMemRefWithIntegerElements(mlir::Type type);

TypeInfo GetTypeInfo(mlir::Value value);
TypeInfo GetTypeInfo(ast::Expr *expr);
void SetTypeInfo(mlir::Value value, const TypeInfo &typeInfo);

} // namespace causalflow::avelang::ir
