//===- KernelFusionPass.cpp - Elementwise chain fusion --------------------===//
//
// Fuses elementwise consumer chains (bias_add → relu, etc.) into single
// linalg.generic ops. Does NOT fuse elementwise ops into contractions —
// that would incorrectly apply epilogue ops (ReLU, bias) per-reduction-step
// instead of after the full reduction. Contraction+epilogue fusion is
// handled by tile-and-fuse in the tiling passes.
//
// Strategy:
//   1. Fuse elementwise → elementwise chains via MLIR's
//      populateElementwiseOpsFusionPatterns (bias_add + relu → one generic).
//
// Runs BEFORE tiling so the fused elementwise epilogue is tiled as one unit.
//
//===----------------------------------------------------------------------===//

#include "AdaptiveMatmul/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace adaptive_matmul {

void KernelFusionPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();

  // ---- Elementwise → elementwise chain fusion ----
  // Fuses chains like bias_add → relu into a single linalg.generic.
  // Contractions (matmul) are left untouched — their epilogues will be
  // fused into the tile loops by the tiling pass (tile-and-fuse).
  {
    mlir::RewritePatternSet patterns(&getContext());
    mlir::linalg::ControlFusionFn control = [](mlir::OpOperand *) {
      return true;
    };
    mlir::linalg::populateElementwiseOpsFusionPatterns(patterns, control);

    mlir::GreedyRewriteConfig config;
    config.setMaxIterations(10);
    config.setUseTopDownTraversal(true);

    if (mlir::failed(
            mlir::applyPatternsGreedily(module, std::move(patterns), config))) {
      llvm::errs() << "[KernelFusion] Elementwise fusion failed.\n";
      signalPassFailure();
      return;
    }
  }

  // ---- Report ----
  unsigned contractions = 0, elementwise = 0;
  module.walk([&](mlir::linalg::LinalgOp op) {
    if (mlir::linalg::isaContractionOpInterface(op))
      ++contractions;
    else if (mlir::linalg::isElementwise(op))
      ++elementwise;
  });

  llvm::outs() << "[KernelFusion] Elementwise chain fusion done. Remaining: "
               << contractions << " contraction(s), " << elementwise
               << " elementwise epilogue(s).\n";
}

} // namespace adaptive_matmul
