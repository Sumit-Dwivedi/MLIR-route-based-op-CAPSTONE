#include "AdaptiveMatmul/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

namespace adaptive_matmul {

class GenericTilingPattern : public mlir::OpRewritePattern<mlir::linalg::MatmulOp> {
public:
  using OpRewritePattern<mlir::linalg::MatmulOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::linalg::MatmulOp op, mlir::PatternRewriter &rewriter) const override {
    auto strategyAttr = op->getAttrOfType<mlir::StringAttr>("optimization_strategy");
    if (!strategyAttr || strategyAttr.getValue() != "square") {
      return mlir::failure();
    }

    if (op->hasAttr("tiled")) {
      return mlir::failure();
    }

    // Square tiling for square matrices (e.g., 32x32 tiles)
    mlir::linalg::LinalgTilingOptions options;
    options.setTileSizes({32, 32, 32});

    auto result = mlir::linalg::tileLinalgOp(rewriter, op, options);
    if (mlir::failed(result)) {
      return mlir::failure();
    }

    result->op->setAttr("tiled", rewriter.getUnitAttr());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

void GenericTilingPass::runOnOperation() {
  bool has_square = false;
  getOperation().walk([&](mlir::linalg::LinalgOp op) {
    auto strategyAttr = op->getAttrOfType<mlir::StringAttr>("optimization_strategy");
    if (strategyAttr && strategyAttr.getValue() == "square") has_square = true;
  });
  if (!has_square) return;
  mlir::ModuleOp module = getOperation();
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<GenericTilingPattern>(&getContext());
  
  if (mlir::failed(mlir::applyPatternsGreedily(module, std::move(patterns)))) {
    signalPassFailure();
  }
  llvm::outs() << "Applied Generic Square Tiling\n";
}

} // namespace adaptive_matmul
