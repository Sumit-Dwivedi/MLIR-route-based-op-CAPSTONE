//===- KernelFusionPass.cpp - Linalg elementwise fusion -------------------===//
//
// Fuses linalg.matmul with consumer elementwise operations (ReLU, bias add)
// using MLIR's populateElementwiseOpsFusionPatterns. Runs BEFORE tiling and
// vectorization so the fused operation is tiled as a single micro-kernel.
//
// Fusion candidates:
//   matmul -> linalg.generic (ReLU, bias_add, activation)
//   matmul -> linalg.elemwise_binary (add)
//   matmul -> linalg.elemwise_unary  (exp, tanh, etc.)
//
//===----------------------------------------------------------------------===//

#include "AdaptiveMatmul/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace adaptive_matmul {

/// Returns true if the producer-consumer edge should be fused.
/// We fuse when the producer is a contraction (matmul) and the consumer
/// is elementwise — this is the standard XLA/IREE epilogue fusion pattern.
static bool controlFusionFn(mlir::OpOperand *fusedOperand) {
  auto producer = fusedOperand->get().getDefiningOp<mlir::linalg::LinalgOp>();
  if (!producer)
    return false;

  auto consumer =
      mlir::dyn_cast<mlir::linalg::LinalgOp>(fusedOperand->getOwner());
  if (!consumer)
    return false;

  // Fuse contraction (matmul) producers into elementwise consumers.
  if (mlir::linalg::isaContractionOpInterface(producer) &&
      mlir::linalg::isElementwise(consumer))
    return true;

  // Also fuse elementwise -> elementwise chains (e.g., bias_add -> relu).
  if (mlir::linalg::isElementwise(producer) &&
      mlir::linalg::isElementwise(consumer))
    return true;

  return false;
}

void KernelFusionPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();

  mlir::RewritePatternSet patterns(&getContext());
  mlir::linalg::ControlFusionFn control = controlFusionFn;
  mlir::linalg::populateElementwiseOpsFusionPatterns(patterns, control);

  mlir::GreedyRewriteConfig config;
  config.setMaxIterations(10);
  config.setUseTopDownTraversal(true);

  if (mlir::failed(
          mlir::applyPatternsGreedily(module, std::move(patterns), config))) {
    llvm::errs() << "[KernelFusion] Pattern application failed.\n";
    signalPassFailure();
    return;
  }

  // Count remaining ops to report fusion stats.
  unsigned matmuls = 0, generics = 0;
  module.walk([&](mlir::linalg::LinalgOp op) {
    if (mlir::linalg::isaContractionOpInterface(op))
      ++matmuls;
    else if (mlir::linalg::isElementwise(op))
      ++generics;
  });

  llvm::outs() << "[KernelFusion] Done. Remaining: " << matmuls
               << " contraction(s), " << generics << " elementwise op(s).\n";
}

} // namespace adaptive_matmul
