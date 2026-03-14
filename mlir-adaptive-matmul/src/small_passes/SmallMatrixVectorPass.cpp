#include "AdaptiveMatmul/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

namespace adaptive_matmul {

class SmallMatrixVectorPattern : public mlir::OpRewritePattern<mlir::linalg::MatmulOp> {
public:
  using OpRewritePattern<mlir::linalg::MatmulOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::linalg::MatmulOp op, mlir::PatternRewriter &rewriter) const override {
    auto strategyAttr = op->getAttrOfType<mlir::StringAttr>("optimization_strategy");
    if (!strategyAttr || strategyAttr.getValue() != "small") {
      return mlir::failure();
    }

    if (op->hasAttr("vectorized")) {
      return mlir::failure();
    }

    // Just mark as vectorized for now as a placeholder
    op->setAttr("vectorized", rewriter.getUnitAttr());
    llvm::outs() << "Simulating Small Matrix Vectorization\n";

    return mlir::success();
  }
};

void SmallMatrixVectorPass::runOnOperation() {
  bool has_small = false;
  getOperation().walk([&](mlir::linalg::LinalgOp op) {
    auto strategyAttr = op->getAttrOfType<mlir::StringAttr>("optimization_strategy");
    if (strategyAttr && strategyAttr.getValue() == "small") has_small = true;
  });
  if (!has_small) return;
  mlir::ModuleOp module = getOperation();
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<SmallMatrixVectorPattern>(&getContext());
  
  if (mlir::failed(mlir::applyPatternsGreedily(module, std::move(patterns)))) {
    signalPassFailure();
  }
  llvm::outs() << "Applied Small Matrix Vectorization (Simulated)\n";
}

} // namespace adaptive_matmul
