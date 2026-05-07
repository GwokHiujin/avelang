#pragma once

#include <memory>
#include <mlir/IR/MLIRContext.h>

namespace causalflow::avelang::ir {

class IRContext {
  public:
    static std::unique_ptr<IRContext> Create();
    mlir::MLIRContext *GetMLIRContext() const;

  protected:
    explicit IRContext();
    std::unique_ptr<mlir::MLIRContext> mlir_context_;
};
} // namespace causalflow::avelang::ir