#include "AdaptiveMatmul/MatrixAnalyzer.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <algorithm>

namespace adaptive_matmul {

std::optional<MatrixInfo> MatrixAnalyzer::analyze(mlir::linalg::LinalgOp op) {
  if (!llvm::isa<mlir::linalg::MatmulOp, mlir::linalg::BatchMatmulOp>(op.getOperation())) {
    return std::nullopt;
  }

  auto shapes = op.getStaticLoopRanges();
  if (shapes.size() < 3) return std::nullopt;

  MatrixInfo info;
  // For matmul: M=row(A), N=col(B), K=col(A)/row(B)
  // For batch_matmul: batch, M, N, K
  if (llvm::isa<mlir::linalg::MatmulOp>(op.getOperation())) {
      info.M = shapes[0];
      info.N = shapes[1];
      info.K = shapes[2];
  } else if (llvm::isa<mlir::linalg::BatchMatmulOp>(op.getOperation())) {
      info.M = shapes[1];
      info.N = shapes[2];
      info.K = shapes[3];
  }

  if (info.M == mlir::ShapedType::kDynamic || info.N == mlir::ShapedType::kDynamic) {
      info.aspect_ratio = 1.0; // Assume square for dynamic shapes for now
  } else {
      info.aspect_ratio = static_cast<double>(std::max(info.M, info.N)) / 
                         std::max(static_cast<int64_t>(1), std::min(info.M, info.N));
  }

  if (info.aspect_ratio > 8.0) {
    info.shape = MatrixShape::Skinny;
  } else if (info.aspect_ratio < 1.5 && info.M > 128) {
    info.shape = MatrixShape::Square;
  } else if (info.M > 0 && info.M < 128 && info.N > 0 && info.N < 128) {
    info.shape = MatrixShape::Small;
  } else {
    info.shape = MatrixShape::Generic;
  }

  return info;
}

} // namespace adaptive_matmul
