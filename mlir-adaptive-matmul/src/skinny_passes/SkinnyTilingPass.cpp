//===- SkinnyTilingPass.cpp - IREE-dispatch-aware tiling for skinny mats --===//
//
// Two-level tiling for skinny (tall-and-thin or short-and-wide) matrices:
//
//   Level 0 (Workgroup): Tile the dominant parallel dimension with scf.for.
//            IREE's dispatch region formation recovers workgroup parallelism
//            from the tile structure after bufferization.
//
//   Level 1 (UKernel / L1): Tile the reduction dimension K with sequential
//            scf.for loops sized for L1 residency.  Inner tiles of 32–64
//            along K maximise data reuse inside a micro-kernel call.
//
//   Dynamic-shape guard: When M or N is dynamic at compile time, the op is
//            marked for runtime dispatch and left untiled.
//
//===----------------------------------------------------------------------===//

#include "AdaptiveMatmul/Passes.h"
#include "AdaptiveMatmul/HardwareInfo.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace adaptive_matmul {

// ---------------------------------------------------------------------------
// Constants derived from micro-architecture modelling.
// ---------------------------------------------------------------------------
static constexpr int64_t kWorkgroupTileM = 128; // L2-friendly workgroup tile
static constexpr int64_t kWorkgroupTileN = 128;
static constexpr int64_t kUKernelTileK   = 32;  // L1-friendly reduction tile

// ---------------------------------------------------------------------------
// Helper: decide tile sizes for a skinny matmul based on static shape.
// ---------------------------------------------------------------------------
struct SkinnyTileSizes {
  int64_t tileM;
  int64_t tileN;
  int64_t tileK;
};

static SkinnyTileSizes computeSkinnyTiles(int64_t M, int64_t N, int64_t K,
                                          const HardwareFeatures &hw) {
  SkinnyTileSizes ts;
  ts.tileK = std::min(kUKernelTileK, K > 0 ? K : kUKernelTileK);

  if (M >= N) {
    // Tall-and-thin: tile M (dominant), leave N untiled.
    ts.tileM = hw.has_avx512 ? 256 : kWorkgroupTileM;
    ts.tileN = 0;
  } else {
    // Short-and-wide: tile N (dominant), leave M untiled.
    ts.tileN = hw.has_avx512 ? 256 : kWorkgroupTileN;
    ts.tileM = 0;
  }

  return ts;
}

// ---------------------------------------------------------------------------
// Pass entry point — walk-based rewriting (avoids greedy driver issues).
// ---------------------------------------------------------------------------
void SkinnyTilingPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  auto hw = HardwareInfo::detect();

  // Collect skinny matmul ops first (can't modify while walking).
  llvm::SmallVector<mlir::linalg::MatmulOp> skinnyOps;
  module.walk([&](mlir::linalg::MatmulOp op) {
    auto s = op->getAttrOfType<mlir::StringAttr>("optimization_strategy");
    if (s && s.getValue() == "skinny" && !op->hasAttr("tiled"))
      skinnyOps.push_back(op);
  });

  if (skinnyOps.empty())
    return;

  mlir::IRRewriter rewriter(&getContext());

  for (auto op : skinnyOps) {
    auto shapes = op.getStaticLoopRanges();
    if (shapes.size() < 3)
      continue;

    int64_t M = shapes[0], N = shapes[1], K = shapes[2];

    // Dynamic shape guard — mark and skip.
    if (M == mlir::ShapedType::kDynamic || N == mlir::ShapedType::kDynamic) {
      op->setAttr("tiled", rewriter.getUnitAttr());
      op->setAttr("skinny_dynamic_fallback", rewriter.getUnitAttr());
      llvm::outs() << "[SkinnyTiling] Dynamic shape — deferring.\n";
      continue;
    }

    auto tiles = computeSkinnyTiles(M, N, K, hw);

    // Level 0: Workgroup tiling.
    mlir::linalg::LinalgTilingOptions opts;
    opts.setTileSizes({tiles.tileM, tiles.tileN, tiles.tileK});

    rewriter.setInsertionPoint(op);
    auto result = mlir::linalg::tileLinalgOp(rewriter, op, opts);
    if (mlir::failed(result))
      continue;

    auto strategy =
        op->getAttrOfType<mlir::StringAttr>("optimization_strategy");
    result->op->setAttr("optimization_strategy", strategy);
    result->op->setAttr("tiled", rewriter.getUnitAttr());
    rewriter.replaceOp(op, result->tensorResults);
  }

  llvm::outs() << "[SkinnyTiling] Two-level tiling complete.\n";
}

} // namespace adaptive_matmul
