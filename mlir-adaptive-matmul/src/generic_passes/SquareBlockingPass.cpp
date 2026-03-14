//===- SquareBlockingPass.cpp - Cache-aware blocking for square matrices --===//
//
// Two-level blocking strategy for near-square matrices:
//
//   Level 0 (L2 blocking):  Tile M, N, K into blocks that fit in L2 cache.
//   Level 1 (Register tile): Tile M, N to small MR x NR micro-tiles sized
//                            for SIMD register files.
//
//===----------------------------------------------------------------------===//

#include "AdaptiveMatmul/Passes.h"
#include "AdaptiveMatmul/HardwareInfo.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace adaptive_matmul {

// ---------------------------------------------------------------------------
// Blocking constants — tuned for server-class x86 (Zen4 / SPR).
// ---------------------------------------------------------------------------
struct BlockingSizes {
  int64_t l2M, l2N, l1K;
  int64_t regM, regN;
};

static BlockingSizes computeBlockingSizes(const HardwareFeatures &hw) {
  BlockingSizes bs;
  if (hw.has_avx512) {
    bs.l2M = 128; bs.l2N = 128;
    bs.regM = 16; bs.regN = 16;
  } else {
    bs.l2M = 64;  bs.l2N = 64;
    bs.regM = 8;  bs.regN = 8;
  }
  bs.l1K = 64;

  if (hw.cache_size_kb < 4096) {
    bs.l2M = std::max<int64_t>(32, bs.l2M / 2);
    bs.l2N = std::max<int64_t>(32, bs.l2N / 2);
  }
  return bs;
}

// ---------------------------------------------------------------------------
// Pass entry point.
// ---------------------------------------------------------------------------
void SquareBlockingPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();

  auto hw = HardwareInfo::detect();
  auto bs = computeBlockingSizes(hw);

  // --- Level 0: L2 blocking ---
  {
    llvm::SmallVector<mlir::linalg::MatmulOp> targets;
    module.walk([&](mlir::linalg::MatmulOp op) {
      auto s = op->getAttrOfType<mlir::StringAttr>("optimization_strategy");
      if (s && s.getValue() == "square" && !op->hasAttr("square_l2_blocked"))
        targets.push_back(op);
    });

    if (targets.empty())
      return;

    mlir::IRRewriter rewriter(&getContext());

    for (auto op : targets) {
      auto shapes = op.getStaticLoopRanges();
      if (shapes.size() < 3) continue;

      int64_t M = shapes[0], N = shapes[1], K = shapes[2];

      if (M == mlir::ShapedType::kDynamic || N == mlir::ShapedType::kDynamic) {
        op->setAttr("square_l2_blocked", rewriter.getUnitAttr());
        op->setAttr("square_dynamic_fallback", rewriter.getUnitAttr());
        continue;
      }

      int64_t tM = std::min(bs.l2M, M);
      int64_t tN = std::min(bs.l2N, N);
      int64_t tK = std::min(bs.l1K, K > 0 ? K : bs.l1K);

      mlir::linalg::LinalgTilingOptions opts;
      opts.setTileSizes({tM, tN, tK});

      rewriter.setInsertionPoint(op);
      auto result = mlir::linalg::tileLinalgOp(rewriter, op, opts);
      if (mlir::failed(result)) continue;

      auto strategy = op->getAttrOfType<mlir::StringAttr>("optimization_strategy");
      result->op->setAttr("optimization_strategy", strategy);
      result->op->setAttr("square_l2_blocked", rewriter.getUnitAttr());
      rewriter.replaceOp(op, result->tensorResults);
    }
  }

  // --- Level 1: Register tiling ---
  {
    llvm::SmallVector<mlir::linalg::MatmulOp> targets;
    module.walk([&](mlir::linalg::MatmulOp op) {
      auto s = op->getAttrOfType<mlir::StringAttr>("optimization_strategy");
      if (s && s.getValue() == "square" &&
          op->hasAttr("square_l2_blocked") &&
          !op->hasAttr("square_reg_tiled"))
        targets.push_back(op);
    });

    mlir::IRRewriter rewriter(&getContext());

    for (auto op : targets) {
      auto shapes = op.getStaticLoopRanges();
      if (shapes.size() < 3) continue;

      int64_t M = shapes[0], N = shapes[1];
      if (M == mlir::ShapedType::kDynamic || N == mlir::ShapedType::kDynamic ||
          (M <= bs.regM && N <= bs.regN)) {
        op->setAttr("square_reg_tiled", rewriter.getUnitAttr());
        continue;
      }

      int64_t tM = std::min(bs.regM, M);
      int64_t tN = std::min(bs.regN, N);

      mlir::linalg::LinalgTilingOptions opts;
      opts.setTileSizes({tM, tN, 0});

      rewriter.setInsertionPoint(op);
      auto result = mlir::linalg::tileLinalgOp(rewriter, op, opts);
      if (mlir::failed(result)) continue;

      auto strategy = op->getAttrOfType<mlir::StringAttr>("optimization_strategy");
      result->op->setAttr("optimization_strategy", strategy);
      result->op->setAttr("square_l2_blocked", rewriter.getUnitAttr());
      result->op->setAttr("square_reg_tiled", rewriter.getUnitAttr());
      rewriter.replaceOp(op, result->tensorResults);
    }
  }

  llvm::outs() << "[SquareBlocking] Two-level blocking complete "
                  "(L2 + register).\n";
}

} // namespace adaptive_matmul
