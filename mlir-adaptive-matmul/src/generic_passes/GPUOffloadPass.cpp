//===- GPUOffloadPass.cpp - Map massive matmuls to GPU blocks -------------===//
//
// Tiles operations tagged with optimization_strategy = "gpu" into
// scf.forall loops with gpu.block mapping attributes.  This prepares
// the IR for the standard convert-scf-forall-to-gpu pass that creates
// gpu.launch ops.
//
// Tiling: M,N → 64x64 GPU blocks.  K is not tiled (full reduction per block).
//
//===----------------------------------------------------------------------===//

#include "AdaptiveMatmul/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

namespace adaptive_matmul {

static constexpr int64_t kGPUBlockM = 64;
static constexpr int64_t kGPUBlockN = 64;

void GPUOffloadPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  auto *ctx = module.getContext();

  // Collect GPU-targeted ops.
  llvm::SmallVector<mlir::linalg::LinalgOp> targets;
  module.walk([&](mlir::linalg::LinalgOp op) {
    auto s = op->getAttrOfType<mlir::StringAttr>("optimization_strategy");
    if (s && s.getValue() == "gpu")
      targets.push_back(op);
  });

  if (targets.empty()) {
    llvm::outs() << "[GPUOffload] No GPU-targeted ops found.\n";
    return;
  }

  mlir::IRRewriter rewriter(ctx);
  unsigned offloaded = 0;

  for (auto op : targets) {
    auto shapes = op.getStaticLoopRanges();
    if (shapes.size() < 3)
      continue;

    int64_t M = shapes[0], N = shapes[1];
    if (M == mlir::ShapedType::kDynamic || N == mlir::ShapedType::kDynamic)
      continue;

    int64_t tileM = std::min(kGPUBlockM, M);
    int64_t tileN = std::min(kGPUBlockN, N);

    // Tile M and N into scf.forall with GPU block mapping.
    mlir::scf::SCFTilingOptions tilingOpts;
    llvm::SmallVector<mlir::OpFoldResult> tileSizes = {
        rewriter.getIndexAttr(tileM),
        rewriter.getIndexAttr(tileN),
        rewriter.getIndexAttr(0) // don't tile K — full reduction per block
    };
    tilingOpts.setTileSizes(tileSizes);
    tilingOpts.setLoopType(
        mlir::scf::SCFTilingOptions::LoopType::ForallOp);

    // Map parallel dims: M → blockIdx.y, N → blockIdx.x.
    llvm::SmallVector<mlir::Attribute> mapping;
    mapping.push_back(mlir::gpu::GPUBlockMappingAttr::get(
        ctx, mlir::gpu::MappingId::DimY));
    mapping.push_back(mlir::gpu::GPUBlockMappingAttr::get(
        ctx, mlir::gpu::MappingId::DimX));
    tilingOpts.setMapping(mapping);

    auto tilingInterface =
        mlir::cast<mlir::TilingInterface>(op.getOperation());
    rewriter.setInsertionPoint(op);
    auto result =
        mlir::scf::tileUsingSCF(rewriter, tilingInterface, tilingOpts);

    if (mlir::failed(result)) {
      llvm::errs() << "[GPUOffload] Failed to tile op for GPU.\n";
      continue;
    }

    rewriter.replaceOp(op, result->replacements);
    ++offloaded;

    llvm::outs() << "[GPUOffload] Tiled " << M << "x" << N << " -> "
                 << tileM << "x" << tileN << " GPU blocks ("
                 << ((M + tileM - 1) / tileM) << "x"
                 << ((N + tileN - 1) / tileN) << " grid).\n";
  }

  if (offloaded > 0)
    llvm::outs() << "[GPUOffload] Offloaded " << offloaded << " op(s).\n";
}

} // namespace adaptive_matmul
