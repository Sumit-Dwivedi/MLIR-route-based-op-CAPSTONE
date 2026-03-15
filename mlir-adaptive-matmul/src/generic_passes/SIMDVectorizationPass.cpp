//===- SIMDVectorizationPass.cpp - Hardware-aware vectorization -----------===//
//
// Vectorizes innermost linalg.matmul ops into the vector dialect.
// Uses walk-based IRRewriter (not greedy pattern driver) to match
// the codepath that the transform dialect uses successfully.
//
//===----------------------------------------------------------------------===//

#include "AdaptiveMatmul/Passes.h"
#include "AdaptiveMatmul/HardwareInfo.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace adaptive_matmul {

static int64_t computeVectorLen(int simdWidthBits, mlir::Type elementType) {
  unsigned bitwidth = 32;
  if (elementType.isF64())
    bitwidth = 64;
  else if (elementType.isF16() || elementType.isBF16())
    bitwidth = 16;
  else if (elementType.isInteger(8))
    bitwidth = 8;
  return std::max<int64_t>(1, simdWidthBits / bitwidth);
}

static mlir::Type getOutputElementType(mlir::linalg::LinalgOp op) {
  if (op.getNumDpsInits() == 0)
    return mlir::Float32Type::get(op.getContext());
  mlir::Value init = op.getDpsInits()[0];
  if (auto shapedTy = mlir::dyn_cast<mlir::ShapedType>(init.getType()))
    return shapedTy.getElementType();
  return mlir::Float32Type::get(op.getContext());
}

void SIMDVectorizationPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  auto hw = HardwareInfo::detectFromModule(module);

  // Collect targets first — can't modify while walking.
  llvm::SmallVector<mlir::Operation *> targets;
  module.walk([&](mlir::linalg::LinalgOp op) {
    if (!mlir::linalg::hasVectorizationImpl(op))
      return;
    targets.push_back(op);
  });

  if (targets.empty())
    return;

  mlir::IRRewriter rewriter(&getContext());
  unsigned vectorized = 0;

  for (auto *op : targets) {
    auto linalgOp = mlir::cast<mlir::linalg::LinalgOp>(op);
    auto shapes = linalgOp.getStaticLoopRanges();
    if (shapes.size() < 3)
      continue;

    int64_t M = shapes[0], N = shapes[1], K = shapes[2];

    // Skip dynamic shapes — runtime dispatch handles them.
    if (M == mlir::ShapedType::kDynamic ||
        N == mlir::ShapedType::kDynamic ||
        K == mlir::ShapedType::kDynamic)
      continue;

    // Skip strategies that have their own vectorization or are GPU-targeted.
    if (auto s = op->getAttrOfType<mlir::StringAttr>("optimization_strategy")) {
      llvm::StringRef sv = s.getValue();
      if (sv == "small" || sv == "gpu")
        continue;
    }

    // Determine vector tile scaling from datatype attribute.
    // INT8 inputs pack 4x more elements per SIMD register than f32.
    int64_t vecScale = 1;
    if (auto dtAttr =
            op->getAttrOfType<mlir::StringAttr>("adaptive.datatype")) {
      if (dtAttr.getValue() == "int8")
        vecScale = 4;
    }

    llvm::SmallVector<int64_t> vecSizes(shapes.begin(), shapes.end());
    // Scale the innermost dimension (K / reduction) by the datatype factor
    // to saturate integer SIMD registers (e.g., 64 i8 vs 16 f32 per 512-bit).
    if (vecScale > 1 && vecSizes.size() >= 3) {
      vecSizes[2] = std::min(vecSizes[2] * vecScale, K);
      llvm::outs() << "[SIMDVectorization] INT8 mode: K vector tile "
                   << shapes[2] << " -> " << vecSizes[2] << "\n";
    }
    llvm::SmallVector<bool> scalableDims(vecSizes.size(), false);

    // Match the transform dialect's codepath exactly:
    // 1. Set insertion point
    // 2. Call vectorize
    // 3. Replace op with results
    rewriter.setInsertionPoint(op);
    auto result = mlir::linalg::vectorize(
        rewriter, op, vecSizes, scalableDims);

    if (mlir::succeeded(result)) {
      rewriter.replaceOp(op, result->replacements);
      ++vectorized;
    }
  }

  if (vectorized > 0) {
    llvm::outs() << "[SIMDVectorization] Vectorized " << vectorized
                 << " op(s), SIMD width " << hw.sim_width << " bits.\n";
  }
}

} // namespace adaptive_matmul
