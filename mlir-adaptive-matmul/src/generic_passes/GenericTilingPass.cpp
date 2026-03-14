#include "AdaptiveMatmul/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace adaptive_matmul {

void GenericTilingPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();

  llvm::SmallVector<mlir::linalg::MatmulOp> targets;
  module.walk([&](mlir::linalg::MatmulOp op) {
    auto s = op->getAttrOfType<mlir::StringAttr>("optimization_strategy");
    if (s && s.getValue() == "square" && !op->hasAttr("tiled"))
      targets.push_back(op);
  });

  if (targets.empty())
    return;

  mlir::IRRewriter rewriter(&getContext());

  for (auto op : targets) {
    mlir::linalg::LinalgTilingOptions opts;
    opts.setTileSizes({32, 32, 32});

    rewriter.setInsertionPoint(op);
    auto result = mlir::linalg::tileLinalgOp(rewriter, op, opts);
    if (mlir::failed(result))
      continue;

    result->op->setAttr("tiled", rewriter.getUnitAttr());
    rewriter.replaceOp(op, result->tensorResults);
  }

  llvm::outs() << "Applied Generic Square Tiling\n";
}

} // namespace adaptive_matmul
