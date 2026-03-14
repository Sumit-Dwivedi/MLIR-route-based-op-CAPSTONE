#include "AdaptiveMatmul/CostModel.h"
#include <cmath>

namespace adaptive_matmul {

double CostModel::computeScore(const MatrixInfo& info, const HardwareFeatures& hardware) {
  double alpha = 1.5; // weight for aspect ratio
  double beta = 0.0001; // weight for size
  double gamma = 2.0; // weight for hardware

  double matrix_size = static_cast<double>(info.M * info.N * info.K);
  double hardware_factor = hardware.has_avx512 ? 1.0 : (hardware.has_avx2 ? 2.0 : 4.0);

  double score = alpha * info.aspect_ratio + beta * std::sqrt(matrix_size) + gamma * hardware_factor;

  return score;
}

} // namespace adaptive_matmul
