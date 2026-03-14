#ifndef ADAPTIVE_MATMUL_MATRIX_ANALYZER_H
#define ADAPTIVE_MATMUL_MATRIX_ANALYZER_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <optional>

namespace adaptive_matmul {

enum class MatrixShape {
  Skinny,
  Square,
  Small,
  Generic
};

struct MatrixInfo {
  int64_t M;
  int64_t N;
  int64_t K;
  double aspect_ratio;
  MatrixShape shape;
};

class MatrixAnalyzer {
public:
  static std::optional<MatrixInfo> analyze(mlir::linalg::LinalgOp op);
};

} // namespace adaptive_matmul

#endif // ADAPTIVE_MATMUL_MATRIX_ANALYZER_H
