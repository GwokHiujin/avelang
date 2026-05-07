#pragma once

#include <cstdint>
#include <optional>

#include <mlir/IR/Value.h>

namespace causalflow::avelang {
namespace ast {
class Expr;
} // namespace ast

namespace ir {

struct GeneratorContext;

class ConstantFolder {
  public:
    explicit ConstantFolder(GeneratorContext *ctx) : ctx_(ctx) {}

    std::optional<int64_t> Evaluate(ast::Expr *expr) const;
    static std::optional<int64_t> FoldIntValue(mlir::Value value);
    static std::optional<bool> FoldBoolValue(mlir::Value value);

  private:
    static std::optional<int64_t> GetConstantIntValue(mlir::Value value);
    std::optional<int64_t> ResolveConstantReference(ast::Expr *expr) const;

    GeneratorContext *ctx_;
};

} // namespace ir
} // namespace causalflow::avelang
