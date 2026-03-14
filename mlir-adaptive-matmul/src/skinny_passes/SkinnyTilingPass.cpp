#include "AdaptiveMatmul/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

namespace adaptive_matmul {

class LinalgTilingPattern : public mlir::OpRewritePattern<mlir::linalg::MatmulOp> {
public:
  using OpRewritePattern<mlir::linalg::MatmulOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::linalg::MatmulOp op, mlir::PatternRewriter &rewriter) const override {
    auto strategyAttr = op->getAttrOfType<mlir::StringAttr>("optimization_strategy");
    if (!strategyAttr || strategyAttr.getValue() != "skinny") {
      return mlir::failure();
    }

    if (op->hasAttr("tiled")) {
      return mlir::failure();
    }

    // Placeholder to avoid crash in this research iteration
    op->setAttr("tiled", rewriter.getUnitAttr());
    return mlir::success();
  }
};

void SkinnyTilingPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  
  bool has_skinny = false;
  module.walk([&](mlir::linalg::LinalgOp op) {
    auto strategyAttr = op->getAttrOfType<mlir::StringAttr>("optimization_strategy");
    if (strategyAttr && strategyAttr.getValue() == "skinny") has_skinny = true;
  });
  if (!has_skinny) return;

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<LinalgTilingPattern>(&getContext());
  
  if (mlir::failed(mlir::applyPatternsGreedily(module, std::move(patterns)))) {
    signalPassFailure();
  }
  llvm::outs() << "Applied Skinny Tiling (Research Placeholder)\n";
}

} // namespace adaptive_matmul
