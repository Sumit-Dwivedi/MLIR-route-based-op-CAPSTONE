#ifndef ADAPTIVE_MATMUL_ROUTER_H
#define ADAPTIVE_MATMUL_ROUTER_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <memory>

namespace adaptive_matmul {

class RouterPass : public mlir::PassWrapper<RouterPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RouterPass)

  llvm::StringRef getArgument() const final { return "adaptive-router"; }
  llvm::StringRef getDescription() const final { return "Routes matrix multiplications to specialized pipelines."; }

  void runOnOperation() override;
};

std::unique_ptr<mlir::Pass> createRouterPass();

} // namespace adaptive_matmul

#endif // ADAPTIVE_MATMUL_ROUTER_H
